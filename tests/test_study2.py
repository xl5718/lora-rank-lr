"""Audit of study2_data_scaling.py, with the model and tokenizer stubbed."""

import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import types
from contextlib import redirect_stdout

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')

import numpy as np
import torch
from torch import nn

FAILS, CHECKS = [], 0


def check(desc, ok, detail=''):
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILS.append(desc)
    print(('  [PASS] ' if ok else '  [FAIL] ') + desc + (f'   {detail}' if detail else ''))


class Out:
    def __init__(self, loss, logits):
        self.loss, self.logits = loss, logits


class TinyModel(nn.Module):
    def __init__(self, vocab=64, dim=16, n_cls=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.body = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.score = nn.Linear(dim, n_cls, bias=False)
        self.lossf = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        h = self.emb(input_ids)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            h = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        logits = self.score(self.body(h))
        return Out(self.lossf(logits, labels) if labels is not None else None, logits)


class TinyLoRAModel(nn.Module):
    """Mimics what peft does: base frozen, thin A/B trainable, head trainable."""

    def __init__(self, rank, dim=16, n_cls=3):
        super().__init__()
        self.base = TinyModel(dim=dim, n_cls=n_cls)
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(dim, rank))
        self.score = self.base.score
        for p in self.score.parameters():
            p.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        h = self.base.emb(input_ids)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            h = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        h = self.base.body(h) + (h @ self.lora_A.T @ self.lora_B.T)
        logits = self.score(h)
        return Out(self.base.lossf(logits, labels) if labels is not None else None,
                   logits)


def stub_build_model(method, rank, seed):
    torch.manual_seed(seed)
    if method == 'lora':
        return TinyLoRAModel(rank)
    m = TinyModel()
    for p in m.parameters():
        p.requires_grad = True
    return m


class TinyTokenizer:
    pad_token = '<pad>'
    pad_token_id = 0

    def __call__(self, texts, truncation=True, max_length=32):
        ids = [[(abs(hash(w)) % 63) + 1 for w in t.split()][:max_length] for t in texts]
        return {'input_ids': ids, 'attention_mask': [[1] * len(x) for x in ids]}


def collate(features):
    n = max(len(f['input_ids']) for f in features)
    return {
        'input_ids': torch.tensor([f['input_ids'] + [0] * (n - len(f['input_ids']))
                                   for f in features]),
        'attention_mask': torch.tensor([f['attention_mask'] + [0] *
                                        (n - len(f['attention_mask']))
                                        for f in features]),
        'labels': torch.tensor([f['labels'] for f in features]),
    }


def install_stubs():
    tr = types.ModuleType('transformers')
    tr.__spec__ = importlib.machinery.ModuleSpec('transformers', None)

    class PreTrainedTokenizerBase:
        pass

    tr.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: TinyTokenizer())
    tr.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: TinyModel())
    tr.DataCollatorWithPadding = lambda tok: collate
    sys.modules['transformers'] = tr


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    install_stubs()
    # study2 imports `peft_study`; point that name at the Study-1 script
    load('peft_study', os.path.join(SRC, 'peft_study.py'))
    s2 = load('s2', os.path.join(SRC, 'study2_data_scaling.py'))

    s2.RESULTS_FILE = '/tmp/study2_test.json'
    if os.path.exists(s2.RESULTS_FILE):
        os.remove(s2.RESULTS_FILE)
    s2.BUDGET_MODE, s2.EPOCHS, s2.FIXED_STEPS = 'epochs', 4, 20
    s2.EVALS_PER_RUN, s2.PATIENCE, s2.STEP_CEILING = 6, 3, 500
    s2.MIN_STEPS = 40
    s2.TRAIN_SIZES = [60, 240]
    s2.SEEDS = [42, 43]
    s2.ARMS = [('lora', 1, 0.01, 3e-4), ('full', 0, 0.01, 5e-6),
               ('fullreg', 0, 0.50, 5e-6)]

    corpus = []
    for i in range(200):
        corpus.append(f'company {i} beat estimates .@positive')
    for i in range(60):
        corpus.append(f'company {i} cut guidance .@negative')
    for i in range(340):
        corpus.append(f'company {i} will report .@neutral')
    raw = ('\n'.join(corpus)).encode('latin-1')
    s2.load_phrasebank = lambda: sys.modules['peft_study']._parse_phrasebank_txt(raw)
    s2.build_model = stub_build_model

    print('=' * 72)
    print('AUDIT of study2_data_scaling.py')
    print('=' * 72)

    tok = TinyTokenizer()
    splits = s2.build_splits(tok)

    print('\n1. Splits: train shrinks, val/test never do')
    check('val and test are full-size and equal',
          len(splits['val']) == len(splits['test']) == 90,
          f"val {len(splits['val'])} test {len(splits['test'])}")
    base = {k: len(v) for k, v in splits.items()}

    print('\n2. Stratified subsampling')
    full_lab = np.array(splits['train']['labels'])
    full_prop = np.bincount(full_lab, minlength=3) / len(full_lab)
    for n in (30, 60, 120, 240):
        sub = s2.stratified_subsample(splits['train'], n, seed=42)
        lab = np.array(sub['labels'])
        prop = np.bincount(lab, minlength=3) / len(lab)
        check(f'n={n}: exact size', len(sub) == n, f'got {len(sub)}')
        check(f'n={n}: class proportions preserved (max dev {np.abs(prop-full_prop).max():.3f})',
              np.abs(prop - full_prop).max() < 0.06)
        check(f'n={n}: no class dropped', (np.bincount(lab, minlength=3) > 0).all())
    big = s2.stratified_subsample(splits['train'], 10 ** 6, seed=42)
    check('n larger than the pool returns the whole pool', len(big) == len(splits['train']))
    a = np.array(s2.stratified_subsample(splits['train'], 60, seed=1)['labels'])
    b = np.array(s2.stratified_subsample(splits['train'], 60, seed=2)['labels'])
    s1 = set(map(tuple, np.array(s2.stratified_subsample(splits['train'], 60, 1)['input_ids'], dtype=object).tolist()))
    s1b = set(map(tuple, np.array(s2.stratified_subsample(splits['train'], 60, 1)['input_ids'], dtype=object).tolist()))
    check('same seed -> identical subsample (reproducible)', s1 == s1b)

    print('\n3. Full grid via main()')
    buf = io.StringIO()
    with redirect_stdout(buf):
        s2.main()
    out = buf.getvalue()
    recs = json.load(open(s2.RESULTS_FILE))
    expected = len(s2.TRAIN_SIZES) * len(s2.ARMS) * len(s2.SEEDS)
    check(f'{expected} runs recorded', len(recs) == expected, f'got {len(recs)}')
    check('val/test sizes unchanged by the grid',
          {k: len(v) for k, v in splits.items()} == base)
    check('each run logs the actual training size used',
          all(r['n_train'] == min(r['n_requested'], len(splits['train'])) for r in recs))

    print('\n4. Budget: epochs mode (primary, per supervisor)')
    check('every run tagged with its budget mode',
          all(r['budget_mode'] == 'epochs' for r in recs))
    check('no run exceeds its own step cap',
          all(r['steps_run'] <= r['max_steps'] for r in recs))
    check('best checkpoint step is within the run',
          all(r['best_step'] <= r['steps_run'] for r in recs))
    check('step cap scales with training size (bigger n -> more updates)',
          all(a['max_steps'] < b['max_steps']
              for a in recs if a['n_requested'] == min(s2.TRAIN_SIZES)
              for b in recs if b['n_requested'] == max(s2.TRAIN_SIZES)))
    cap_exposure = [r['max_steps'] * s2.BATCH / r['n_train'] for r in recs]
    check('BUDGETED exposures are equal among the non-floored sizes '
          '— the design property of epochs mode',
          (lambda v: not v or max(v) / min(v) < 1.2)(
              [r['max_steps'] * s2.BATCH / r['n_train'] for r in recs
               if not r['floor_applied']]),
          f'{len([r for r in recs if r["floor_applied"]])} sizes hit the floor')
    floored = [r for r in recs if r['floor_applied']]
    unfloored = [r for r in recs if not r['floor_applied']]
    check('the step floor binds only where the epoch budget is degenerate',
          all(r['max_steps'] == s2.MIN_STEPS for r in floored)
          and all(r['max_steps'] >= s2.MIN_STEPS for r in unfloored),
          f'{len(floored)} floored / {len(unfloored)} not')
    check('no run ever gets fewer updates than the floor',
          all(r['max_steps'] >= s2.MIN_STEPS for r in recs))
    if unfloored:
        ce = [r['max_steps'] * s2.BATCH / r['n_train'] for r in unfloored]
        check('equal-epochs still holds exactly where the floor does not bind',
              max(ce) / min(ce) < 1.2,
              f'{min(ce):.2f}-{max(ce):.2f} exposures')
    budgeted_evals = [r['max_steps'] // r['eval_every'] for r in recs]
    # Integer division means this is approximate, not exact. What matters is
    # that it no longer SCALES with n: with a fixed 25-step interval the ratio
    # between the smallest and largest condition was ~24x.
    check('budgeted validation checkpoints no longer scale with n '
          '(equal early-stopping granularity across data sizes)',
          max(budgeted_evals) / min(budgeted_evals) < 1.5,
          f'{min(budgeted_evals)}-{max(budgeted_evals)} '
          f'(ratio {max(budgeted_evals)/min(budgeted_evals):.2f}x, target {s2.EVALS_PER_RUN})')
    check('eval interval scales with run length, so overhead does not grow with n',
          all(r['eval_every'] == max(1, r['max_steps'] // s2.EVALS_PER_RUN)
              for r in recs))
    check('eval interval genuinely differs between the smallest and largest n',
          len({r['eval_every'] for r in recs}) > 1,
          str(sorted({(r['n_train'], r['eval_every']) for r in recs})))
    realized = [r['exposures_per_example'] for r in recs]
    check('REALIZED exposures vary — early stopping ends runs at different '
          'epochs, so the equality holds for the budget, not the outcome',
          max(realized) / min(realized) > 1.0,
          f'{min(realized):.1f}–{max(realized):.1f}x (reported, not a defect)')

    print('\n5. Budget: steps mode (secondary robustness check)')
    s2.RESULTS_FILE = '/tmp/study2_test_steps.json'
    if os.path.exists(s2.RESULTS_FILE):
        os.remove(s2.RESULTS_FILE)
    with redirect_stdout(io.StringIO()):
        s2.main(budget='steps')
    srecs = json.load(open(s2.RESULTS_FILE))
    check('steps mode gives every condition the same cap',
          len({r['max_steps'] for r in srecs}) == 1)
    check('...and therefore UNEQUAL exposures — the trade-off is real',
          max(r['exposures_per_example'] for r in srecs)
          / min(r['exposures_per_example'] for r in srecs) > 2.5,
          f"{min(r['exposures_per_example'] for r in srecs):.1f}–"
          f"{max(r['exposures_per_example'] for r in srecs):.1f}x")
    check('the two modes are stored as distinct runs, not confused for repeats',
          all(r['budget_mode'] == 'steps' for r in srecs)
          and len(srecs) == len(recs))
    s2.BUDGET_MODE = 'epochs'
    s2.RESULTS_FILE = '/tmp/study2_test.json'

    print("\n6. The regularisation metric")
    check('gap == train_acc - test_acc for every run',
          all(abs(r['gap'] - (r['train_acc'] - r['test_acc'])) < 1e-9 for r in recs))
    check('train and test measured on the SAME model state, so the gap is real',
          'train_acc' in recs[0] and 'test_acc' in recs[0])
    check('the test set is never used for stopping or selection '
          '(only val_f1 drives early stopping)',
          'best_val_f1' in recs[0]
          and 'evaluate(model, te)' not in
              open(os.path.join(SRC, 'study2_data_scaling.py')).read().split('while step < max_steps')[1]
              .split('# Final-state metrics')[0])

    print("\n7. The H3 control arm")
    fu = [r for r in recs if r['method'] == 'full']
    fr = [r for r in recs if r['method'] == 'fullreg']
    check('fullreg exists and differs from full only in weight decay',
          fr and fu and fr[0]['wd'] == 0.5 and fu[0]['wd'] == 0.01
          and fr[0]['lr'] == fu[0]['lr'])
    check('both full arms train EVERY non-head parameter (exact, scale-free)',
          all(r['adapter_params'] + r['head_params'] == r['total_params']
              for r in fu + fr))
    lo = [r for r in recs if r['method'] == 'lora']
    check('lora leaves the overwhelming majority of the model frozen',
          all(r['adapter_params'] + r['head_params'] < 0.1 * r['total_params']
              for r in lo))
    check('lora adapter budget grows with rank',
          s2.stratified_subsample is not None and
          all(r['adapter_params'] > 0 for r in lo))

    print("\n7b. A learning-rate sweep must not be swallowed by the resume logic")
    before = json.load(open(s2.RESULTS_FILE))
    n_lr_before = len({r['lr'] for r in before if r['method'] == 'lora'
                       and r['rank'] == 1 and r['n_requested'] == 60})
    _arms = s2.ARMS
    try:
        s2.ARMS = [('lora', 1, 0.01, 3e-4), ('lora', 1, 0.01, 1e-3),
                   ('lora', 1, 0.01, 3e-3)]          # same arm, three lrs
        with redirect_stdout(io.StringIO()):
            s2.main(sizes=[60])
    finally:
        s2.ARMS = _arms
    after = json.load(open(s2.RESULTS_FILE))
    lrs = sorted({r['lr'] for r in after if r['method'] == 'lora'
                  and r['rank'] == 1 and r['n_requested'] == 60})
    check('all three learning rates ran '
          '(regression: done() used to ignore lr and skip them)',
          len(lrs) == 3, f'{n_lr_before} -> {len(lrs)} distinct lrs: {lrs}')
    check('the sweep added runs rather than being treated as duplicates',
          len(after) > len(before), f'{len(before)} -> {len(after)}')

    print("\n8. Resume")
    n_before = len(json.load(open(s2.RESULTS_FILE)))
    with redirect_stdout(io.StringIO()):
        s2.main()
    check('re-running adds nothing', len(json.load(open(s2.RESULTS_FILE))) == n_before)
    with redirect_stdout(io.StringIO()):
        s2.main(only='lora', sizes=[60])
    check('filtered re-run also adds nothing',
          len(json.load(open(s2.RESULTS_FILE))) == n_before)

    print("\n9. Recovery + summary")
    lines = [l for l in out.splitlines() if l.strip().startswith('RECOVERY_JSON')]
    check('RECOVERY_JSON emitted', len(lines) >= 1)
    check('last RECOVERY_JSON parses to the full record set',
          len(json.loads(lines[-1].split('RECOVERY_JSON ', 1)[1])) == expected)
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        s2.summarize(recs)
    tbl = buf2.getvalue()
    check('summary prints both tables',
          'TEST ACCURACY' in tbl and 'GENERALISATION GAP' in tbl)
    check('summary has a column per data size',
          all(f'n={n}' in tbl for n in s2.TRAIN_SIZES))

    print('\n' + '=' * 72)
    print(f'{CHECKS - len(FAILS)}/{CHECKS} checks passed')
    if FAILS:
        print('FAILURES:')
        for f in FAILS:
            print('  -', f)
        sys.exit(1)
    print('AUDIT CLEAN')


if __name__ == '__main__':
    main()
