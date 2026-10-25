"""
Study 4 - Replication of the Study 3 rank x learning-rate result
================================================================
Pre-registered in PREREG_study4.txt. Three arms, one factor changed at a time:

    A   Qwen2.5-1.5B  x  Financial PhraseBank   (scale changes, task fixed)
    B   Qwen2.5-0.5B  x  SST-2                  (general sentiment, binary)
    C   Qwen2.5-0.5B  x  AG News                (topic classification, 4 classes)

Every arm uses the Study 3 protocol unchanged: LoRA on q_proj/v_proj with
lora_alpha = 2r, n = 500 stratified training examples, 5 epochs (160 steps),
12 validation checkpoints, patience 4, OneCycle schedule, AdamW wd 0.01,
fp16 autocast + GradScaler, validation macro-F1 for every choice, test read once
at the end of each run. Grid: r in {1,2,4,8,16} x lr in {1e-4,...,1e-2} x 5 seeds.

Run order is SEED-MAJOR: all 25 cells for seed 42, then all 25 for seed 43, ...
If compute runs out part-way, the data already collected is a complete, balanced
design with fewer seeds, rather than a grid with holes.

Timing pilot: pilot(arm) runs two throw-away runs (seeds 990/991, flagged
pilot=True) and prints ONLY time and memory. Pilot runs are excluded from every
analysis by the frozen analysis script.

progress(arm) prints counts only - no accuracies - so the grid stays blind until
it is complete and the frozen analysis is run.
"""

import gc
import json
import math
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

RESULTS_FILE = 'study4_results.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = DEVICE == 'cuda'
AMP_DTYPE = torch.float16          # T4 has no usable bf16

ARMS = {
    'A': {'model': 'Qwen/Qwen2.5-1.5B', 'task': 'fpb'},
    'B': {'model': 'Qwen/Qwen2.5-0.5B', 'task': 'sst2'},
    'C': {'model': 'Qwen/Qwen2.5-0.5B', 'task': 'agnews'},
}
RANKS = [1, 2, 4, 8, 16]
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS = [42, 43, 44, 45, 46]
PILOT_SEEDS = [990, 991]

N_TRAIN = 500
VAL_SIZE = 518                     # same as Study 3
AGNEWS_TEST_SIZE = 2000            # stratified fixed subset of the 7,600 test set
SPLIT_SEED = 0                     # fixes val/test for every run of an arm
BATCH = 16
EPOCHS = 5
MIN_STEPS = 150
EVALS_PER_RUN = 12
PATIENCE = 4
WD = 0.01
MAX_LEN = {'fpb': 96, 'sst2': 96, 'agnews': 128}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

FPB_LABELS = ['negative', 'neutral', 'positive']


def _load_fpb():
    """Financial PhraseBank, 75% agreement, parsed from the original archive."""
    import io
    import zipfile

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id='takala/financial_phrasebank',
                           filename='data/FinancialPhraseBank-v1.0.zip',
                           repo_type='dataset')
    texts, labels = [], []
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith('Sentences_75Agree.txt'))
        for line in z.read(name).decode('latin-1').splitlines():
            line = line.strip()
            if '@' not in line:
                continue
            sent, _, lab = line.rpartition('@')
            lab = lab.strip().lower()
            if lab in FPB_LABELS:
                texts.append(sent.strip())
                labels.append(FPB_LABELS.index(lab))
    if not texts:
        raise RuntimeError('parsed zero sentences from Financial PhraseBank')
    return texts, labels


def _parquet(repo, filename):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download(repo_id=repo, filename=filename,
                                           repo_type='dataset'))


def _text_col(df):
    for c in ('sentence', 'text'):
        if c in df.columns:
            return c
    raise RuntimeError(f'no text column in {list(df.columns)}')


def stratified_indices(labels, n, rng):
    """Indices of a stratified sample of size n (class proportions preserved)."""
    labels = np.asarray(labels)
    if n >= len(labels):
        return rng.permutation(len(labels)).tolist()
    classes, counts = np.unique(labels, return_counts=True)
    quota = np.maximum(np.floor(counts / counts.sum() * n).astype(int), 1)
    while quota.sum() > n:
        quota[np.argmax(quota)] -= 1
    while quota.sum() < n:
        quota[np.argmax(counts / counts.sum() * n - quota)] += 1
    idx = []
    for c, q in zip(classes, quota):
        pool = np.flatnonzero(labels == c)
        idx.extend(rng.choice(pool, size=min(q, len(pool)), replace=False).tolist())
    rng.shuffle(idx)
    return idx


def _carve(texts, labels, n, seed):
    """Split (texts, labels) into a stratified sample of n and the remainder."""
    take = set(stratified_indices(labels, n, np.random.default_rng(seed)))
    a = [i for i in range(len(texts)) if i in take]
    b = [i for i in range(len(texts)) if i not in take]
    pick = lambda ix: ([texts[i] for i in ix], [labels[i] for i in ix])
    return pick(a), pick(b)


def load_task(task):
    """Return dict(pool, val, test, n_classes, majority_rate); each split is
    (texts, labels). Val and test are FIXED for the arm (SPLIT_SEED)."""
    if task == 'fpb':
        # Identical to Study 3: 70 / 15 / 15 stratified, val 518, test 518.
        from datasets import ClassLabel, Dataset
        t, y = _load_fpb()
        ds = Dataset.from_dict({'t': t, 'labels': y}).cast_column(
            'labels', ClassLabel(names=FPB_LABELS))
        ds = ds.train_test_split(test_size=0.3, seed=SPLIT_SEED,
                                 stratify_by_column='labels')
        tmp = ds['test'].train_test_split(test_size=0.5, seed=SPLIT_SEED,
                                          stratify_by_column='labels')
        # list(): datasets 4.x returns lazy Column objects, not lists
        pool = (list(ds['train']['t']), list(ds['train']['labels']))
        val = (list(tmp['train']['t']), list(tmp['train']['labels']))
        test = (list(tmp['test']['t']), list(tmp['test']['labels']))
        k = 3
    elif task == 'sst2':
        # GLUE test labels are hidden, so the labelled GLUE validation set
        # (872 sentences) is the TEST set, and VAL is carved from train.
        tr = _parquet('stanfordnlp/sst2', 'data/train-00000-of-00001.parquet')
        te = _parquet('stanfordnlp/sst2', 'data/validation-00000-of-00001.parquet')
        c1, c2 = _text_col(tr), _text_col(te)
        val, pool = _carve(tr[c1].tolist(), tr['label'].astype(int).tolist(),
                           VAL_SIZE, SPLIT_SEED)
        test = (te[c2].tolist(), te['label'].astype(int).tolist())
        k = 2
    elif task == 'agnews':
        tr = _parquet('fancyzhx/ag_news', 'data/train-00000-of-00001.parquet')
        te = _parquet('fancyzhx/ag_news', 'data/test-00000-of-00001.parquet')
        c1, c2 = _text_col(tr), _text_col(te)
        val, pool = _carve(tr[c1].tolist(), tr['label'].astype(int).tolist(),
                           VAL_SIZE, SPLIT_SEED)
        test, _ = _carve(te[c2].tolist(), te['label'].astype(int).tolist(),
                         AGNEWS_TEST_SIZE, SPLIT_SEED)
        k = 4
    else:
        raise ValueError(task)
    for name, (tx, lb) in (('pool', pool), ('val', val), ('test', test)):
        if len(tx) != len(lb) or not tx:
            raise RuntimeError(f'{task}: bad {name} split')
        if set(lb) - set(range(k)):
            raise RuntimeError(f'{task}: unexpected labels in {name}: {set(lb)}')
    counts = np.bincount(pool[1], minlength=k)
    return {'pool': pool, 'val': val, 'test': test, 'n_classes': k,
            'majority_rate': float(counts.max() / counts.sum())}


class _DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        d['labels'] = int(self.labels[i])
        return d


def encode(tokenizer, split, max_len):
    texts, labels = split
    return _DS(tokenizer(list(texts), truncation=True, max_length=max_len), list(labels))


# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------

def host_ram_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:                                             # noqa: BLE001
        return 0.0


def build_model(model_name, n_classes, rank, seed, pad_id):
    from peft import LoraConfig, TaskType, get_peft_model
    torch.manual_seed(seed)
    kw = {'device_map': DEVICE} if DEVICE == 'cuda' else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, dtype=torch.float32, **kw)
    model.config.pad_token_id = pad_id
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=2 * rank,
                     lora_dropout=0.05, bias='none',
                     target_modules=['q_proj', 'v_proj'], modules_to_save=['score'])
    return get_peft_model(model, cfg).to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, gold = [], []
    for batch in loader:
        labels = batch.pop('labels')
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(**batch).logits
        preds.append(logits.float().argmax(-1).cpu())
        gold.append(labels)
    p, g = torch.cat(preds).numpy(), torch.cat(gold).numpy()
    return accuracy_score(g, p), f1_score(g, p, average='macro')


def budget():
    per_epoch = math.ceil(N_TRAIN / BATCH)
    return max(EPOCHS * per_epoch, MIN_STEPS), per_epoch


def run(arm, rank, lr, seed, data, tokenizer, collator, pilot=False):
    cfg = ARMS[arm]
    np.random.seed(seed)
    idx = stratified_indices(data['pool'][1], N_TRAIN, np.random.default_rng(seed))
    train_split = ([data['pool'][0][i] for i in idx], [data['pool'][1][i] for i in idx])
    ml = MAX_LEN[cfg['task']]
    train = encode(tokenizer, train_split, ml)

    t_load = time.time()
    model = build_model(cfg['model'], data['n_classes'], rank, seed, tokenizer.pad_token_id)
    load_secs = time.time() - t_load
    max_steps, per_epoch = budget()
    eval_every = max(1, max_steps // EVALS_PER_RUN)

    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(train, batch_size=BATCH, shuffle=True, collate_fn=collator, generator=g)
    tr_eval = DataLoader(train, batch_size=64, collate_fn=collator)
    va = DataLoader(data['val_enc'], batch_size=64, collate_fn=collator)
    te = DataLoader(data['test_enc'], batch_size=64, collate_fn=collator)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max_steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)

    best = {'val_f1': -1.0, 'val_acc': 0.0, 'step': 0}
    since, step = 0, 0
    t0 = time.time()
    while step < max_steps:
        model.train()
        for batch in tr:
            if step >= max_steps:
                break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % eval_every == 0 or step == max_steps:
                va_acc, va_f1 = evaluate(model, va)
                if va_f1 > best['val_f1']:
                    best = {'val_acc': va_acc, 'val_f1': va_f1, 'step': step}
                    since = 0
                else:
                    since += 1
                model.train()
        if since >= PATIENCE:
            break

    tr_acc, _ = evaluate(model, tr_eval)
    te_acc, te_f1 = evaluate(model, te)
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == 'cuda' else 0.0
    n_adapter = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and 'score' not in n)

    del model, params, opt, sched, scaler, tr, tr_eval, va, te
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    return dict(study='study4', arm=arm, model=cfg['model'], task=cfg['task'],
                method='lora', rank=rank, lr=lr, wd=WD, seed=seed, pilot=pilot,
                n_train=N_TRAIN, n_classes=data['n_classes'],
                majority_rate=data['majority_rate'],
                n_val=len(data['val'][1]), n_test=len(data['test'][1]),
                adapter_params=n_adapter, max_steps=max_steps, steps_run=step,
                eval_every=eval_every, epochs_run=step / per_epoch,
                best_val_f1=best['val_f1'], best_val_acc=best['val_acc'],
                best_step=best['step'], train_acc=tr_acc, test_acc=te_acc,
                test_f1=te_f1, gap=tr_acc - te_acc, seconds=secs,
                load_seconds=load_secs, peak_gb=peak, host_ram_gb=host_ram_gb(),
                finished_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def load_records():
    try:
        with open(RESULTS_FILE) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return []


def save_records(recs):
    """Atomic write: a disconnect mid-write can never corrupt the results file."""
    tmp = RESULTS_FILE + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(recs, fh, indent=1)
    os.replace(tmp, RESULTS_FILE)


def done(recs, arm, rank, lr, seed, pilot=False):
    return any(r['arm'] == arm and r['rank'] == rank and r['seed'] == seed
               and abs(r['lr'] - lr) <= 1e-12 * max(1.0, lr)
               and bool(r.get('pilot', False)) == pilot for r in recs)


def todo_list(recs, arm, seeds=None):
    """Seed-major order: complete replicate sets first."""
    return [(s, rk, lr) for s in (seeds or SEEDS) for rk in RANKS for lr in LRS
            if not done(recs, arm, rk, lr, s)]


def _prepare(arm):
    cfg = ARMS[arm]
    tok = AutoTokenizer.from_pretrained(cfg['model'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data = load_task(cfg['task'])
    ml = MAX_LEN[cfg['task']]
    data['val_enc'] = encode(tok, data['val'], ml)
    data['test_enc'] = encode(tok, data['test'], ml)
    print(f'arm {arm}: {cfg["model"]} x {cfg["task"]} | pool {len(data["pool"][1])} '
          f'| val {len(data["val"][1])} | test {len(data["test"][1])} '
          f'| classes {data["n_classes"]} | majority {data["majority_rate"]:.3f}')
    return tok, data, DataCollatorWithPadding(tok)


def main(arm, seeds=None):
    """Run (or resume) the full grid of one arm. Safe to re-run at any time."""
    tok, data, coll = _prepare(arm)
    recs = load_records()
    todo = todo_list(recs, arm, seeds)
    print(f'{len(todo)} runs to go for arm {arm}\n')
    chunk = []
    for i, (s, rk, lr) in enumerate(todo, 1):
        r = run(arm, rk, lr, s, data, tok, coll)
        recs = load_records() + [r]          # re-read: never clobber other cells' writes
        save_records(recs)
        chunk.append(r)
        # Progress line deliberately shows no accuracy: the grid stays blind.
        print(f'  [{i}/{len(todo)}] arm {arm} seed={s} r={rk:<2} lr={lr:.0e} | '
              f'{r["seconds"]:.0f}s (+{r["load_seconds"]:.0f}s load) | '
              f'GPU {r["peak_gb"]:.1f} GB | RAM {r["host_ram_gb"]:.1f} GB')
        if len(chunk) == 10 or i == len(todo):
            print('RECOVERY_JSON ' + json.dumps(chunk, separators=(',', ':')))
            chunk = []
    progress(arm)


def pilot(arm):
    """Two throw-away runs for timing/memory. Never analysed."""
    tok, data, coll = _prepare(arm)
    recs = load_records()
    times = []
    for s in PILOT_SEEDS:
        if done(recs, arm, 16, 1e-3, s, pilot=True):
            r = next(x for x in recs if x['arm'] == arm and x['seed'] == s and x.get('pilot'))
        else:
            r = run(arm, 16, 1e-3, s, data, tok, coll, pilot=True)
            recs = load_records() + [r]
            save_records(recs)
        times.append(r['seconds'] + r['load_seconds'])
        print(f'  pilot seed {s}: train+eval {r["seconds"]:.0f}s, load {r["load_seconds"]:.0f}s, '
              f'GPU peak {r["peak_gb"]:.1f} GB, host RAM {r["host_ram_gb"]:.1f} GB')
    per = float(np.mean(times))
    print(f'\n  estimated time for arm {arm}: {per * 125 / 3600:.1f} h for 125 runs '
          f'({per * 25 / 3600:.1f} h per seed)')


def progress(arm=None):
    """Counts only. No accuracies are printed until the frozen analysis is run."""
    recs = [r for r in load_records() if not r.get('pilot')]
    for a in ([arm] if arm else list(ARMS)):
        mine = [r for r in recs if r['arm'] == a]
        per_seed = {s: sum(1 for r in mine if r['seed'] == s) for s in SEEDS}
        print(f'arm {a}: {len(mine)}/125 runs | per seed (of 25): {per_seed}')
