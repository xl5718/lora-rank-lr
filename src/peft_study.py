"""
Route B - the same study on a real pretrained LLM and a real dataset
===================================================================
Run this in Google Colab (Runtime -> Change runtime type -> T4 GPU). Results are
written to colab_results.json after every single run, and main() skips anything
already finished - so if the session disconnects, just run the cell again.
It reproduces the comparison in the report on:

    model    Qwen/Qwen2.5-0.5B          (a real pretrained decoder-only LLM)
    data     financial_phrasebank       (sentences_75agree, 3-way sentiment)
    methods  LoRA (r = 1, 2, 4, 8, 16), full fine-tuning, linear probe

Total runtime on a free T4: roughly 25-40 minutes for the whole grid.

Paste the printed RESULTS TABLE back into the chat and the numbers will replace
the small-scale results in the report.

--------------------------------------------------------------------------
Cell 1 - install (run once, then restart the runtime if Colab asks)
--------------------------------------------------------------------------
!pip install -q "transformers>=4.44" "peft>=0.12" "datasets>=2.20" "accelerate>=0.33" scikit-learn

--------------------------------------------------------------------------
Cell 2 - paste everything below
--------------------------------------------------------------------------
"""

import gc
import json
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

MODEL_NAME = 'Qwen/Qwen2.5-0.5B'
MAX_LEN = 96
BATCH = 16
EPOCHS = 3
SEEDS = [42, 43, 44]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Mixed precision. The T4 is a Turing card: it has fast fp16 tensor cores but no
# usable bf16, so we use fp16 + GradScaler. Master weights stay in fp32; only the
# forward/backward maths runs in half precision, which is numerically safe here
# and roughly halves the time per run.
USE_AMP = DEVICE == 'cuda'
AMP_DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

LABEL_NAMES = ['negative', 'neutral', 'positive']       # -> 0, 1, 2
AGREEMENT_FILE = 'Sentences_75Agree.txt'


def _parse_phrasebank_txt(raw_bytes):
    """The corpus is plain text, one 'sentence@label' pair per line, latin-1."""
    from datasets import ClassLabel, Dataset, Features, Value

    sentences, labels = [], []
    for line in raw_bytes.decode('latin-1').splitlines():
        line = line.strip()
        if not line or '@' not in line:
            continue
        sent, _, lab = line.rpartition('@')
        lab = lab.strip().lower()
        if lab not in LABEL_NAMES:
            continue
        sentences.append(sent.strip())
        labels.append(LABEL_NAMES.index(lab))

    if not sentences:
        raise ValueError('parsed zero sentences from the corpus file')

    features = Features({'sentence': Value('string'),
                         'labels': ClassLabel(names=LABEL_NAMES)})
    return Dataset.from_dict({'sentence': sentences, 'labels': labels},
                             features=features)


def load_phrasebank():
    """Financial PhraseBank, 75% annotator agreement. Labels: 0 neg, 1 neu, 2 pos.

    `datasets` 4.x removed support for loading scripts, which is exactly how this
    corpus used to be served, so `load_dataset("financial_phrasebank", ...)` now
    fails. The raw archive is still hosted in the same repository, so the primary
    path here downloads that file and parses it directly - no script machinery
    involved. Mirrors that were re-uploaded in a modern format are tried after,
    and a locally uploaded file last.
    """
    errors = []

    # 1) Canonical data, fetched as a plain file from the original repo.
    try:
        import io
        import zipfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id='takala/financial_phrasebank',
                               filename='data/FinancialPhraseBank-v1.0.zip',
                               repo_type='dataset')
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(AGREEMENT_FILE))
            ds = _parse_phrasebank_txt(z.read(name))
        print(f'loaded {len(ds)} sentences from takala/financial_phrasebank '
              f'({AGREEMENT_FILE})')
        return ds
    except Exception as e:                                        # noqa: BLE001
        errors.append(f'  original archive: {type(e).__name__}: {e}')

    # 2) Community mirrors already converted to a modern (script-free) format.
    for repo, config in [('FinanceMTEB/financial_phrasebank', None),
                         ('dohonba/financial_phrasebank', 'sentences_75agree'),
                         ('dohonba/financial_phrasebank', None)]:
        try:
            ds = load_dataset(repo, config) if config else load_dataset(repo)
            split = ds['train'] if 'train' in ds else ds[list(ds.keys())[0]]
            print(f'loaded mirror {repo} ({len(split)} rows)')
            return split
        except Exception as e:                                    # noqa: BLE001
            errors.append(f'  {repo} [{config}]: {type(e).__name__}: {e}')

    # 3) A file the user uploaded into the Colab session.
    import os
    for cand in (AGREEMENT_FILE, f'/content/{AGREEMENT_FILE}'):
        if os.path.exists(cand):
            with open(cand, 'rb') as fh:
                ds = _parse_phrasebank_txt(fh.read())
            print(f'loaded {len(ds)} sentences from uploaded {cand}')
            return ds

    raise RuntimeError(
        'Could not load Financial PhraseBank. Attempts:\n' + '\n'.join(errors)
        + f'\n\nManual fallback: download FinancialPhraseBank-v1.0.zip from\n'
          f'  https://huggingface.co/datasets/takala/financial_phrasebank/'
          f'blob/main/data/FinancialPhraseBank-v1.0.zip\n'
          f'unzip it, and upload {AGREEMENT_FILE} into the Colab file browser '
          f'(folder icon on the left), then re-run this cell.')


def get_data(tokenizer, seed=42):
    ds = load_phrasebank()

    # Mirrors use different column names; normalise to sentence / labels.
    for src in ('label', 'gold_label', 'target'):
        if src in ds.column_names and 'labels' not in ds.column_names:
            ds = ds.rename_column(src, 'labels')
    for src in ('text', 'sent', 'input'):
        if src in ds.column_names and 'sentence' not in ds.column_names:
            ds = ds.rename_column(src, 'sentence')
    missing = {'sentence', 'labels'} - set(ds.column_names)
    if missing:
        raise RuntimeError(f'dataset is missing {missing}; '
                           f'columns present: {ds.column_names}')
    ds = ds.remove_columns([c for c in ds.column_names
                            if c not in ('sentence', 'labels')])

    # stratified splitting requires labels to be a ClassLabel feature
    from datasets import ClassLabel
    if not isinstance(ds.features['labels'], ClassLabel):
        if ds.features['labels'].dtype == 'string':
            ds = ds.map(lambda b: {'labels': [LABEL_NAMES.index(x.lower())
                                              for x in b['labels']]}, batched=True)
        ds = ds.cast_column('labels', ClassLabel(names=LABEL_NAMES))
    print('label distribution:',
          {LABEL_NAMES[i]: ds['labels'].count(i) for i in range(3)})

    # PhraseBank ships as a single split, so carve out train / val / test
    ds = ds.train_test_split(test_size=0.3, seed=seed, stratify_by_column='labels')
    tmp = ds['test'].train_test_split(test_size=0.5, seed=seed,
                                      stratify_by_column='labels')
    splits = {'train': ds['train'], 'val': tmp['train'], 'test': tmp['test']}

    def tok(b):
        return tokenizer(b['sentence'], truncation=True, max_length=MAX_LEN)

    for k in splits:
        splits[k] = splits[k].map(tok, batched=True,
                                  remove_columns=['sentence'])
    return splits


# ---------------------------------------------------------------------------
# Model construction per method
# ---------------------------------------------------------------------------

def host_ram_gb():
    """Resident set size of this process, in GB. Colab kills the session at ~12 GB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:                                             # noqa: BLE001
        return 0.0


def build_model(method, rank=8, alpha=16, seed=42):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, dtype=torch.float32)
    model.config.pad_token_id = model.config.eos_token_id

    if method == 'head':
        for n, p in model.named_parameters():
            p.requires_grad = 'score' in n

    elif method == 'full':
        for p in model.parameters():
            p.requires_grad = True

    elif method == 'lora':
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha,
                         lora_dropout=0.05, bias='none',
                         target_modules=['q_proj', 'v_proj'],
                         modules_to_save=['score'])
        model = get_peft_model(model, cfg)

    else:
        raise ValueError(method)

    return model.to(DEVICE)


def count_trainable(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    # Exclude the freshly-initialised classification head from the adapter budget
    head = sum(p.numel() for n, p in model.named_parameters()
               if 'score' in n and p.requires_grad)
    return tr - head, head, tot


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

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
    p = torch.cat(preds).numpy()
    g = torch.cat(gold).numpy()
    return accuracy_score(g, p), f1_score(g, p, average='macro')


def run(method, rank, lr, seed, splits, collator):
    model = build_model(method, rank=rank, seed=seed)
    n_adapter, n_head, n_total = count_trainable(model)

    tr = DataLoader(splits['train'], batch_size=BATCH, shuffle=True,
                    collate_fn=collator)
    va = DataLoader(splits['val'], batch_size=64, collate_fn=collator)
    te = DataLoader(splits['test'], batch_size=64, collate_fn=collator)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = EPOCHS * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)

    scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)

    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        for batch in tr:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            # Unscale before clipping, otherwise the norm is computed on the
            # scaled gradients and the clip threshold means nothing.
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
    secs = time.time() - t0

    va_acc, va_f1 = evaluate(model, va)
    te_acc, te_f1 = evaluate(model, te)
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == 'cuda' else 0.0

    # Tear the run down completely. `del model` alone is NOT enough: the
    # optimizer holds references to every parameter (and, for full fine-tuning,
    # two AdamW moment buffers per parameter), and the scheduler holds the
    # optimizer. Miss any one of them and a ~2 GB fp32 copy of the model stays
    # reachable, so 20 runs exhaust the 12 GB Colab gives you.
    del model, params, opt, sched, scaler, tr, va, te
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return dict(method=method, rank=rank, lr=lr, seed=seed,
                adapter_params=n_adapter, head_params=n_head, total_params=n_total,
                pct=100 * n_adapter / n_total, val_acc=va_acc, val_f1=va_f1,
                test_acc=te_acc, test_f1=te_f1, seconds=secs, peak_gb=peak,
                host_ram_gb=host_ram_gb())


# ---------------------------------------------------------------------------
# Main grid
# ---------------------------------------------------------------------------

RESULTS_FILE = 'colab_results.json'

# (method, rank, candidate learning rates)
# Ranks 1 / 4 / 16 span two orders of magnitude of adapter capacity, which is
# enough to see the shape of the rank curve. Dropping r=2 and r=8 costs 8 runs.
GRID = [
    ('head', 0, [1e-3, 3e-3]),
    ('lora', 1, [1e-4, 3e-4]),
    ('lora', 4, [1e-4, 3e-4]),
    ('lora', 16, [1e-4, 3e-4]),
    ('full', 0, [5e-6, 2e-5]),
]


def load_records():
    try:
        return json.load(open(RESULTS_FILE))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_records(records):
    json.dump(records, open(RESULTS_FILE, 'w'), indent=1)


def done(records, method, rank, lr, seed):
    return any(r['method'] == method and r['rank'] == rank
               and abs(r['lr'] - lr) < 1e-12 and r['seed'] == seed
               for r in records)


def main(only=None):
    """Run the grid. Safe to re-run: finished jobs are skipped, not repeated.

    `only` filters by method, e.g. main(only='lora') or main(only=['head', 'full']).
    """
    if isinstance(only, str):
        only = [only]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    splits = get_data(tokenizer)
    collator = DataCollatorWithPadding(tokenizer)
    print({k: len(v) for k, v in splits.items()})

    records = load_records()
    if records:
        print(f'resuming: {len(records)} finished runs already in {RESULTS_FILE}')

    for method, rank, lrs in GRID:
        if only and method not in only:
            continue
        tag = method + (f'-r{rank}' if method == 'lora' else '')

        # 1) pick the learning rate on VALIDATION using one seed
        for lr in lrs:
            if done(records, method, rank, lr, SEEDS[0]):
                continue
            r = run(method, rank, lr, SEEDS[0], splits, collator)
            records.append(r)
            save_records(records)
            print(f'    {tag} lr={lr:.0e} seed={SEEDS[0]} done | '
                  f'{r["seconds"]:.0f}s | RAM {r["host_ram_gb"]:.1f} GB')

        probes = [r for r in records if r['method'] == method and r['rank'] == rank
                  and r['seed'] == SEEDS[0]]
        best_lr = max(probes, key=lambda r: r['val_f1'])['lr']

        # 2) repeat the winning setting on the remaining seeds
        for seed in SEEDS[1:]:
            if done(records, method, rank, best_lr, seed):
                continue
            r = run(method, rank, best_lr, seed, splits, collator)
            records.append(r)
            save_records(records)
            print(f'    {tag} lr={best_lr:.0e} seed={seed} done | '
                  f'{r["seconds"]:.0f}s | RAM {r["host_ram_gb"]:.1f} GB')

        # Also dump the raw records into the cell output. If Colab recycles the
        # runtime, /content is wiped and colab_results.json goes with it - but
        # the notebook output survives, so the numbers can still be recovered.
        print(f'  [{tag}] complete (lr* = {best_lr:.0e})')
        print('RECOVERY_JSON ' + json.dumps(records, separators=(',', ':')))

    print_table(records)


def print_table(records=None):
    """Print whatever results exist so far. Safe to call at any point."""
    records = records if records is not None else load_records()
    if not records:
        print('no results yet')
        return
    print('\n' + '=' * 96)
    print('RESULTS TABLE  (paste this back into the chat)')
    print('=' * 96)
    print(f'{"method":<12}{"adapter":>12}{"% total":>9}{"lr*":>9}'
          f'{"test acc":>18}{"macro-F1":>18}{"sec":>8}{"GB":>7}{"n":>3}')
    print('-' * 96)
    for method, rank, _ in GRID:
        tag = method + (f'-r{rank}' if method == 'lora' else '')
        rs = [r for r in records if r['method'] == method and r['rank'] == rank]
        if not rs:
            print(f'{tag:<12}{"(not run yet)":>20}')
            continue
        best_lr = max(rs, key=lambda r: r['val_f1'])['lr']
        rs = [r for r in rs if r['lr'] == best_lr]
        a = np.array([r['test_acc'] for r in rs])
        f = np.array([r['test_f1'] for r in rs])
        r0 = rs[0]
        sa = a.std(ddof=1) if len(a) > 1 else 0.0
        sf = f.std(ddof=1) if len(f) > 1 else 0.0
        print(f'{tag:<12}{r0["adapter_params"]:>12,}{r0["pct"]:>9.3f}{best_lr:>9.1e}'
              f'{a.mean():>12.4f} +/-{sa:<5.4f}{f.mean():>12.4f} +/-{sf:<5.4f}'
              f'{np.mean([r["seconds"] for r in rs]):>8.0f}'
              f'{max(r["peak_gb"] for r in rs):>7.1f}{len(rs):>3}')
    print('-' * 96)
    print(f'model {MODEL_NAME} | {records[0]["total_params"]:,} total parameters | '
          f'{EPOCHS} epochs | seeds {SEEDS}')


if __name__ == '__main__':
    main()
