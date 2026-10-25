"""Tests for study4_replication.py logic that does not need a GPU.
torch / transformers are stubbed; training itself is exercised by the Colab pilot."""
import sys, types, json, os, tempfile
import numpy as np, pandas as pd

# ---- stubs ---------------------------------------------------------------
torch = types.ModuleType('torch'); torch.cuda = types.SimpleNamespace(is_available=lambda: False)
import importlib.machinery as _m
for _x in ('torch',): pass
torch.__spec__ = _m.ModuleSpec('torch', None)
torch.Tensor = type('Tensor', (), {}); torch.float16 = 'fp16'; torch.float32 = 'fp32'
torch.no_grad = lambda: (lambda f: f)
ud = types.ModuleType('torch.utils.data'); ud.__spec__ = _m.ModuleSpec('torch.utils.data', None); ud.DataLoader = object; ud.Dataset = object
tu = types.ModuleType('torch.utils'); tu.data = ud; torch.utils = tu
sys.modules.update({'torch': torch, 'torch.utils': tu, 'torch.utils.data': ud})
tf = types.ModuleType('transformers'); tf.__spec__ = _m.ModuleSpec('transformers', None)
for n in ('AutoModelForSequenceClassification', 'AutoTokenizer', 'DataCollatorWithPadding'):
    setattr(tf, n, object)
sys.modules['transformers'] = tf
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
import study4_replication as S

fails = 0
def check(name, cond):
    global fails
    print(('PASS ' if cond else 'FAIL ') + name); fails += (not cond)

# ---- stratified sampling --------------------------------------------------
rng = np.random.default_rng(0)
lab = rng.choice(3, size=3453, p=[.12, .62, .26])
for n in (500, 518, 37):
    idx = S.stratified_indices(lab, n, np.random.default_rng(1))
    sub = lab[idx]; full = np.bincount(lab) / len(lab); got = np.bincount(sub, minlength=3) / n
    check(f'stratified n={n}: size exact', len(idx) == n)
    check(f'stratified n={n}: no duplicates', len(set(idx)) == n)
    check(f'stratified n={n}: proportions within 1/n+0.002', np.all(np.abs(got - full) <= 1 / n + 0.002))
check('stratified deterministic given seed',
      S.stratified_indices(lab, 500, np.random.default_rng(5)) == S.stratified_indices(lab, 500, np.random.default_rng(5)))
check('different seeds give different training sets',
      S.stratified_indices(lab, 500, np.random.default_rng(42)) != S.stratified_indices(lab, 500, np.random.default_rng(43)))

# ---- carving: disjoint and complete -----------------------------------------
texts = [f't{i}' for i in range(5000)]; labels = list(rng.choice(4, size=5000))
(vt, vl), (pt, pl) = S._carve(texts, labels, 518, 0)
check('carve: val size 518', len(vt) == 518)
check('carve: val and pool disjoint', not (set(vt) & set(pt)))
check('carve: nothing lost', len(vt) + len(pt) == 5000)
check('carve: labels stay aligned', all(int(t[1:]) < 5000 and labels[int(t[1:])] == l for t, l in zip(vt, vl)))
check('carve: fixed seed -> identical val', S._carve(texts, labels, 518, 0)[0] == (vt, vl))

# ---- task loaders with fake parquet ------------------------------------------
def fake_parquet(repo, fn):
    r = np.random.default_rng(len(fn))
    if repo == 'stanfordnlp/sst2':
        n = 67349 if 'train' in fn else 872
        return pd.DataFrame({'idx': range(n), 'sentence': [f's{"tr" if "train" in fn else "te"}{i}' for i in range(n)],
                             'label': r.choice(2, size=n, p=[.44, .56])})
    n = 120000 if 'train' in fn else 7600
    return pd.DataFrame({'text': [f'a{"tr" if "train" in fn else "te"}{i}' for i in range(n)], 'label': r.choice(4, size=n)})
S._parquet = fake_parquet
d = S.load_task('sst2')
check('sst2: test = GLUE validation (872)', len(d['test'][1]) == 872)
check('sst2: val 518', len(d['val'][1]) == 518)
check('sst2: pool = train - val', len(d['pool'][1]) == 67349 - 518)
check('sst2: val disjoint from pool', not (set(d['val'][0]) & set(d['pool'][0])))
check('sst2: 2 classes, majority ~0.56', d['n_classes'] == 2 and 0.54 < d['majority_rate'] < 0.58)
d = S.load_task('agnews')
check('agnews: test 2000 stratified subset', len(d['test'][1]) == 2000)
check('agnews: test balanced (4 classes, each 0.22-0.28)',
      all(0.22 < c / 2000 < 0.28 for c in np.bincount(d['test'][1])))
check('agnews: val disjoint from pool', not (set(d['val'][0]) & set(d['pool'][0])))
check('agnews: test texts come from test file', all(t.startswith('ate') for t in d['test'][0][:50]))
check('agnews: 4 classes', d['n_classes'] == 4)

for _k in ('torch', 'torch.utils', 'torch.utils.data', 'transformers'):
    sys.modules.pop(_k, None)   # let `datasets` see that torch is absent
try:
    import datasets  # noqa
    have_ds = True
except Exception as e:
    print('datasets import failed:', e)
    have_ds = False
if have_ds:
    fpb_t = [f'f{i}' for i in range(3453)]; fpb_y = list(lab)
    S._load_fpb = lambda: (fpb_t, fpb_y)
    d = S.load_task('fpb')
    check('fpb: Study 3 split sizes 2417/518/518',
          (len(d['pool'][1]), len(d['val'][1]), len(d['test'][1])) == (2417, 518, 518))
    check('fpb: splits disjoint', not (set(d['pool'][0]) & set(d['val'][0]) | set(d['pool'][0]) & set(d['test'][0])))
    check('fpb: plain lists (not lazy Columns)', isinstance(d['pool'][0], list))
else:
    print('SKIP fpb split test (datasets not installed here)')

# ---- budget identical to Study 3 --------------------------------------------
check('budget: 160 steps, 32 per epoch (Study 3)', S.budget() == (160, 32))

# ---- bookkeeping ---------------------------------------------------------------
tmp = tempfile.mkdtemp(); S.RESULTS_FILE = os.path.join(tmp, 'r.json')
check('load_records on missing file -> []', S.load_records() == [])
todo = S.todo_list([], 'A')
check('todo: 125 runs per arm', len(todo) == 125)
check('todo: seed-major (first 25 all seed 42)', all(t[0] == 42 for t in todo[:25]) and todo[25][0] == 43)
recs = [dict(arm='A', rank=8, lr=1e-3, seed=42, pilot=False),
        dict(arm='A', rank=16, lr=1e-3, seed=990, pilot=True)]
S.save_records(recs)
check('atomic save leaves no tmp file', not os.path.exists(S.RESULTS_FILE + '.tmp'))
check('save/load round trip', S.load_records() == recs)
check('done: matches arm/rank/lr/seed', S.done(recs, 'A', 8, 1e-3, 42))
check('done: different lr not done', not S.done(recs, 'A', 8, 3e-3, 42))
check('done: different arm not done', not S.done(recs, 'B', 8, 1e-3, 42))
check('done: lr float noise tolerated', S.done(recs, 'A', 8, 0.001 * (1 + 1e-14), 42))
check('done: pilot run does not count as real run', not S.done(recs, 'A', 16, 1e-3, 990))
check('todo excludes finished run', len(S.todo_list(recs, 'A')) == 124)
check('pilot seeds disjoint from analysis seeds', not set(S.PILOT_SEEDS) & set(S.SEEDS))
print(f'\n{"ALL TESTS PASSED" if not fails else f"{fails} FAILED"}')
