"""
Study 2 - Is LoRA's low rank acting as regularisation in low-resource financial NLP?
====================================================================================
Motivated by an unresolved confound in Study 1. There, rank mattered on a 558k
synthetic setup and did not on Qwen2.5-0.5B - but model size, corpus and training-set
size all changed at once, so the effect could not be attributed to any of them.

This study holds the model (Qwen2.5-0.5B) and the corpus (Financial PhraseBank)
fixed and varies ONLY the amount of labelled training data.

    H1  low-rank adaptation is more beneficial the scarcer the supervision
    H2  rank sensitivity decreases as training data grows
    H3  the benefit is specific to the low-rank structure, not to merely having
        fewer trainable parameters   (tested by the `fullreg` control arm)

Design decisions that the conclusions depend on:

  * TRAIN is subsampled; VAL and TEST stay at full size for every run, so
    evaluation noise does not grow as the training set shrinks.
  * Subsampling is stratified. PhraseBank is 62% neutral and 12% negative; an
    unstratified draw of 100 would leave ~12 negatives and the n=100 condition
    would be measuring class imbalance rather than data scarcity.
  * Budget: EPOCHS are held constant (see BUDGET_MODE below). Because
    steps = epochs x n / batch, holding epochs constant necessarily lets the
    number of gradient updates vary with n, and holding steps constant
    necessarily lets the number of exposures per example vary. The two cannot
    both be fixed. Equal epochs is the primary setting, on the grounds that it
    lets each condition actually use its full training set and matches how a
    practitioner with n labels would really train; the fixed-step variant is
    retained as a robustness check, and the qualitative conclusion should
    survive both.
  * The generalisation gap (train accuracy - test accuracy) is recorded, because
    "LoRA wins when data is scarce" is consistent with better optimisation as
    well as with regularisation. Only the gap distinguishes the two.
  * `fullreg` = full fine-tuning with heavy weight decay. If LoRA's advantage is
    just capacity control, a capacity-controlled full fine-tune should match it.

Usage:
    python study2_data_scaling.py                 # whole grid, resumable
    python study2_data_scaling.py --only lora     # one arm at a time
    python study2_data_scaling.py --sizes 100 250 # one slice
"""

import argparse
import gc
import json
import math
import os
import time

import numpy as np
import torch
from datasets import ClassLabel
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

# Reuse the audited data loading from Study 1.
from peft_study import (LABEL_NAMES, MODEL_NAME, MAX_LEN, AMP_DTYPE, USE_AMP,
                        DEVICE, load_phrasebank, host_ram_gb)

RESULTS_FILE = 'study2_results.json'

BATCH = 16

# --- training budget -------------------------------------------------------
# steps = epochs x n / batch, so with batch and n fixed you cannot hold BOTH
# the number of updates and the number of exposures per example constant. The
# two modes trade one confound for the other:
#
#   'epochs' (primary, per supervisor's request)
#       every example is seen EPOCHS times in every condition. Comparable
#       exposure per example; unequal number of gradient updates (n=100 gets
#       50 updates, n=2417 gets 1,209).
#
#   'steps'  (secondary, robustness check)
#       every condition gets the same number of gradient updates. Comparable
#       optimisation budget; unequal exposure per example (n=100 sees each
#       example 48 times, n=2417 sees each twice).
#
# Neither is "correct". Running the small grid under both and checking that the
# qualitative conclusion survives is what actually settles it.
BUDGET_MODE = 'epochs'
# 5 epochs: Study 1 converged on the full training set within 3 epochs, so 5
# plus early stopping leaves headroom while keeping the grid inside a few hours
# of single-GPU time.
EPOCHS = 5               # used when BUDGET_MODE == 'epochs'
FIXED_STEPS = 300        # used when BUDGET_MODE == 'steps'
STEP_CEILING = 2000      # safety cap only; should not bind at these sizes

# Floor on the number of gradient updates, in 'epochs' mode only.
#
# Measured, not assumed. At n=100, 5 epochs is 5 x ceil(100/16) = 35 updates, and
# 35 updates train nothing: a pilot run reached 0.486 test accuracy with 0.54
# train accuracy - below the 0.62 majority-class baseline, and still improving
# when the budget ran out. The same condition at 150 steps reached 0.718 test
# with 1.000 train accuracy and a 0.282 generalisation gap.
#
# Without the floor the small-n columns would measure update starvation rather
# than data scarcity, and H1 could not be tested at all. The floor binds only at
# n=100 and n=250; from n=500 up, the epoch budget already exceeds it, so equal
# epochs is preserved wherever it is meaningful. The cost is that those two
# conditions see each example more often than the larger ones - recorded in
# `exposures_per_example` and reported.
MIN_STEPS = 150

# Validation checkpoints are spaced so that EVERY run gets the same NUMBER of
# them, rather than the same interval in steps. A fixed interval would give the
# n=100 condition 2 checkpoints and the n=2417 condition 48, so early stopping
# would be far more sensitive at large n than at small n - and the evaluation
# overhead would grow without bound (at n=2417 it exceeded the training time).
EVALS_PER_RUN = 12
PATIENCE = 4             # early stop after this many evals without val-F1 gain
SEEDS = [42, 43, 44]
TRAIN_SIZES = [100, 250, 500, 1000, 2417]     # 2417 = the full training split

# method, rank, weight_decay, learning rate
ARMS = [
    ('lora', 1, 0.01, 3e-4),
    ('lora', 2, 0.01, 3e-4),
    ('lora', 4, 0.01, 3e-4),
    ('lora', 8, 0.01, 3e-4),
    ('lora', 16, 0.01, 3e-4),
    ('full', 0, 0.01, 5e-6),
    ('fullreg', 0, 0.50, 5e-6),   # H3 control: capacity-limited full fine-tuning
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_splits(tokenizer, seed=0):
    """Full-size val/test, plus the full training pool to subsample from."""
    ds = load_phrasebank()
    if 'label' in ds.column_names and 'labels' not in ds.column_names:
        ds = ds.rename_column('label', 'labels')
    if not isinstance(ds.features['labels'], ClassLabel):
        ds = ds.cast_column('labels', ClassLabel(names=LABEL_NAMES))

    ds = ds.train_test_split(test_size=0.3, seed=seed, stratify_by_column='labels')
    tmp = ds['test'].train_test_split(test_size=0.5, seed=seed,
                                      stratify_by_column='labels')
    splits = {'train': ds['train'], 'val': tmp['train'], 'test': tmp['test']}

    def tok(b):
        return tokenizer(b['sentence'], truncation=True, max_length=MAX_LEN)

    for k in splits:
        splits[k] = splits[k].map(tok, batched=True, remove_columns=['sentence'])
    return splits


def stratified_subsample(train, n, seed):
    """Draw n examples keeping the class proportions of the full training pool."""
    if n >= len(train):
        return train
    labels = np.array(train['labels'])
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    quota = np.floor(counts / counts.sum() * n).astype(int)
    quota = np.maximum(quota, 1)                    # never drop a class entirely
    while quota.sum() > n:                          # trim the largest class first
        quota[np.argmax(quota)] -= 1
    while quota.sum() < n:
        quota[np.argmax(counts - quota)] += 1

    idx = []
    for c, q in zip(classes, quota):
        pool = np.flatnonzero(labels == c)
        idx.extend(rng.choice(pool, size=min(q, len(pool)), replace=False).tolist())
    rng.shuffle(idx)
    return train.select(idx)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(method, rank, seed):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, dtype=torch.float32)
    model.config.pad_token_id = model.config.eos_token_id

    if method in ('full', 'fullreg'):
        for p in model.parameters():
            p.requires_grad = True
    elif method == 'lora':
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=2 * rank,
                         lora_dropout=0.05, bias='none',
                         target_modules=['q_proj', 'v_proj'],
                         modules_to_save=['score'])
        model = get_peft_model(model, cfg)
    else:
        raise ValueError(method)
    return model.to(DEVICE)


def count_trainable(model):
    """(adapter params, classification-head params, total).

    The head is randomly initialised and therefore trained under every arm, so
    it is reported separately and excluded from the adapter budget.
    """
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for n, p in model.named_parameters()
               if 'score' in n and p.requires_grad)
    return tr - head, head, tot


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


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def budget_for(n_examples):
    """Total optimisation steps for a training set of this size."""
    per_epoch = math.ceil(n_examples / BATCH)
    if BUDGET_MODE == 'epochs':
        steps = max(EPOCHS * per_epoch, MIN_STEPS)
    elif BUDGET_MODE == 'steps':
        steps = FIXED_STEPS
    else:
        raise ValueError(BUDGET_MODE)
    return min(steps, STEP_CEILING), per_epoch


def run(method, rank, wd, lr, n_train, seed, splits, collator):
    train = stratified_subsample(splits['train'], n_train, seed)
    model = build_model(method, rank, seed)
    n_adapter, n_head, n_total = count_trainable(model)
    max_steps, steps_per_epoch = budget_for(len(train))
    eval_every = max(1, max_steps // EVALS_PER_RUN)

    tr = DataLoader(train, batch_size=BATCH, shuffle=True, collate_fn=collator,
                    drop_last=False)
    tr_eval = DataLoader(train, batch_size=64, collate_fn=collator)
    va = DataLoader(splits['val'], batch_size=64, collate_fn=collator)
    te = DataLoader(splits['test'], batch_size=64, collate_fn=collator)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=max_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)

    best = {'val_f1': -1.0, 'val_acc': 0.0, 'step': 0}
    since_improve, step = 0, 0
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
                # Only VALIDATION is evaluated during training. Touching the
                # test set at every checkpoint would triple the evaluation cost
                # (a forward pass over 518 examples of a 0.5B model is ~4 s) and
                # buys nothing: early stopping decides when to stop, and the
                # reported model is the one training stopped at.
                va_acc, va_f1 = evaluate(model, va)
                if va_f1 > best['val_f1']:
                    best = {'val_acc': va_acc, 'val_f1': va_f1, 'step': step}
                    since_improve = 0
                else:
                    since_improve += 1
                model.train()
        if since_improve >= PATIENCE:
            break

    # Final-state metrics: the generalisation gap must be read at the END of
    # training, not at the early-stopped optimum, or it is not a gap at all.
    fin_tr_acc, _ = evaluate(model, tr_eval)
    fin_te_acc, fin_te_f1 = evaluate(model, te)
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == 'cuda' else 0.0

    del model, params, opt, sched, scaler, tr, tr_eval, va, te
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    return dict(method=method, rank=rank, wd=wd, lr=lr, n_train=len(train),
                n_requested=n_train, seed=seed,
                adapter_params=n_adapter, head_params=n_head, total_params=n_total,
                pct=100 * n_adapter / n_total,
                budget_mode=BUDGET_MODE, max_steps=max_steps,
                floor_applied=bool(BUDGET_MODE == 'epochs'
                                   and EPOCHS * steps_per_epoch < MIN_STEPS),
                steps_per_epoch=steps_per_epoch, eval_every=eval_every,
                n_evals=step // eval_every,
                epochs_run=step / steps_per_epoch,
                exposures_per_example=step * BATCH / len(train),
                best_val_acc=best['val_acc'], best_val_f1=best['val_f1'],
                best_step=best['step'], steps_run=step,
                # Reported at the point training stopped, which is the early-
                # stopped model. train/test are measured on the same state, so
                # the gap between them is meaningful.
                test_acc=fin_te_acc, test_f1=fin_te_f1,
                train_acc=fin_tr_acc,
                gap=fin_tr_acc - fin_te_acc,
                seconds=secs, peak_gb=peak, host_ram_gb=host_ram_gb())


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def load_records():
    try:
        return json.load(open(RESULTS_FILE))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def done(recs, method, rank, n, seed, lr=None, wd=None):
    """Has this exact configuration already been run?

    The identity of a run includes the budget mode, the learning rate and the
    weight decay - not just the arm. Omitting lr made a learning-rate sweep
    silently skip every point after the first, since all of them share
    (method, rank, n, seed).
    """
    for r in recs:
        if (r['method'] == method and r['rank'] == rank
                and r['n_requested'] == n and r['seed'] == seed
                and r.get('budget_mode', 'steps') == BUDGET_MODE
                and (lr is None or abs(r['lr'] - lr) < 1e-12)
                and (wd is None or abs(r['wd'] - wd) < 1e-12)):
            return True
    return False


def tag_of(method, rank):
    return f'lora-r{rank}' if method == 'lora' else method


def main(only=None, sizes=None, budget=None):
    global BUDGET_MODE
    if budget:
        BUDGET_MODE = budget
    if isinstance(only, str):
        only = [only]
    sizes = sizes or TRAIN_SIZES

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    splits = build_splits(tokenizer)
    collator = DataCollatorWithPadding(tokenizer)
    budget_desc = (f'{EPOCHS} epochs' if BUDGET_MODE == 'epochs'
                   else f'{FIXED_STEPS} steps')
    print({k: len(v) for k, v in splits.items()},
          f'| budget: {budget_desc} ({BUDGET_MODE} mode), patience {PATIENCE}')
    if BUDGET_MODE == 'epochs':
        print('  steps by size:',
              {n: budget_for(min(n, len(splits['train'])))[0] for n in (sizes or TRAIN_SIZES)})

    recs = load_records()
    if recs:
        print(f'resuming: {len(recs)} runs already done')

    todo = [(m, rk, wd, lr, n, s)
            for n in sizes
            for (m, rk, wd, lr) in ARMS
            for s in SEEDS
            if (not only or m in only) and not done(recs, m, rk, n, s, lr, wd)]
    print(f'{len(todo)} runs to go\n')

    for i, (m, rk, wd, lr, n, s) in enumerate(todo, 1):
        r = run(m, rk, wd, lr, n, s, splits, collator)
        recs.append(r)
        json.dump(recs, open(RESULTS_FILE, 'w'), indent=1)
        print(f'  [{i}/{len(todo)}] {tag_of(m, rk):<10} n={r["n_train"]:<5} '
              f'seed={s} | test {r["test_acc"]:.4f} | gap {r["gap"]:+.4f} | '
              f'{r["seconds"]:.0f}s | RAM {r["host_ram_gb"]:.1f} GB')
        if i % 6 == 0 or i == len(todo):
            print('RECOVERY_JSON ' + json.dumps(recs, separators=(',', ':')))

    summarize(recs)


def summarize(recs=None):
    recs = recs if recs is not None else load_records()
    if not recs:
        print('no results yet')
        return
    tags = [tag_of(m, rk) for (m, rk, _, _) in ARMS]
    sizes = sorted({r['n_requested'] for r in recs})

    for metric, label in [('test_acc', 'TEST ACCURACY'), ('gap', 'GENERALISATION GAP')]:
        print(f'\n{"=" * 78}\n{label}  (mean over seeds)\n{"=" * 78}')
        print(f'{"method":<11}' + ''.join(f'{"n=" + str(n):>13}' for n in sizes))
        print('-' * 78)
        for tag in tags:
            row = f'{tag:<11}'
            for n in sizes:
                sel = [r[metric] for r in recs
                       if tag_of(r['method'], r['rank']) == tag and r['n_requested'] == n]
                row += f'{np.mean(sel):>13.4f}' if sel else f'{"-":>13}'
            print(row)
    print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--sizes', nargs='*', type=int, default=None)
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        main(only=a.only, sizes=a.sizes)
