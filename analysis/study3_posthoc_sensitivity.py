"""
Study 3 — POST-HOC SENSITIVITY ANALYSIS.  NOT PRE-REGISTERED.

Status, stated plainly:
  The pre-registered analysis is study3_prereg_analysis.py (sha256 in
  PREREG_study3.txt), frozen before any Study 3 result existed. It is the
  primary analysis and is run unchanged.

  On 24 Sep the supervisor proposed three amendments "before running Study 3".
  By then the grid had already finished and its descriptive summary had been
  viewed. The amendments were SPECIFIED without sight of the data, but they are
  IMPLEMENTED here after the data were seen, and the implementation choices
  below (test family, divergence threshold) were made after that point. They are
  therefore reported as sensitivity analyses, alongside - never instead of - the
  pre-registered result.

Amendments implemented:
  A1  Q3 with Holm correction. Each directional comparison "rank a beats rank b
      at lr x" is a one-sided Welch t-test on validation macro-F1 (3 seeds per
      arm). Family = every such comparison: 10 rank pairs x 5 lrs x 2 directions
      = 100 tests. Q3 passes if some pair has a Holm-significant win in EACH
      direction (alpha = 0.05).
  A2  "Unstable" renamed. LOW-FIT = final train accuracy < 0.80. DIVERGED is
      recorded separately as final train accuracy at or below the majority-class
      rate of the training set (~0.62): the model fits no better than a trivial
      constant predictor. The majority rate is a property of the dataset, not of
      any run. This threshold was chosen after the data were seen.
  A3  The Q4 spread, stated explicitly (documentation - the computation is the
      pre-registered one, unchanged):
        tuned spread    = max over ranks minus min over ranks of the mean test
                          accuracy (mean over the 3 seeds), each rank evaluated at
                          its own validation-selected learning rate (selection on
                          mean validation macro-F1 over the 3 seeds);
        fixed spread(x) = the same max-minus-min, every rank at the common lr x;
        MATERIAL        = tuned spread <= 0.5 x median over x of fixed spread(x).
  R1  Q1 robustness: permutation test of the interaction F (residuals permuted
      under the additive model, 10,000 permutations), as suggested for later.
"""

import json
import math
import statistics as st
import sys
from itertools import combinations

import numpy as np
from scipy import stats

PATH = sys.argv[1] if len(sys.argv) > 1 else 'study2_results.json'
N = 500
RANKS = [1, 2, 4, 8, 16]
LR_L = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS = [42, 43, 44]
LOW_FIT = 0.80
MAJORITY = 2146 / 3453          # neutral share of Financial PhraseBank 75%-agree
ALPHA = 0.05

R = json.load(open(PATH))


def close(a, b):
    return abs(a - b) <= 1e-12 * max(1.0, abs(b))


def lora(rank, lr):
    return {r['seed']: r for r in R
            if r['budget_mode'] == 'epochs' and r['method'] == 'lora'
            and r['n_requested'] == N and r['rank'] == rank
            and close(r['lr'], lr) and close(r['wd'], 0.01)}


def lrs(x):
    return f'{x:.0e}'


out = {}
print('=' * 90)
print('STUDY 3 — POST-HOC SENSITIVITY ANALYSIS  (not pre-registered; see header)')
print('=' * 90)

# --------------------------------------------------------------- A2
print('\nA2  LOW-FIT vs DIVERGED, per cell  (L = low-fit < 0.80, D = diverged <= '
      f'{MAJORITY:.3f})')
print(f'{"rank":>6}' + ''.join(f'{lrs(l):>12}' for l in LR_L))
cells = {}
for rk in RANKS:
    row = f'{rk:>6}'
    for lr in LR_L:
        c = lora(rk, lr)
        d = sum(r['train_acc'] <= MAJORITY for r in c.values())
        low = sum(MAJORITY < r['train_acc'] < LOW_FIT for r in c.values())
        cells[(rk, lr)] = (low, d)
        row += f'{f"L{low} D{d}" if (low or d) else "ok":>12}'
    print(row)
diverged_cells = [(rk, lrs(lr)) for (rk, lr), (_, d) in cells.items() if d]
lowfit_low_lr = [(rk, lrs(lr)) for (rk, lr), (l, d) in cells.items()
                 if l and not d and lr <= 3e-4]
print(f'\n  cells with any DIVERGED seed: {diverged_cells}')
print(f'  LOW-FIT cells at the SMALL-lr end (<= 3e-4), i.e. under-trained rather '
      f'than diverged: {lowfit_low_lr}')
print('  Decision impact: none. The pre-registered decision uses the all-lr and '
      'middle-rank\n  interaction tests; the stability label enters only the '
      'descriptive "stable lrs" rows,\n  whose column set is identical under '
      'either label.')
out['A2'] = {'diverged_cells': diverged_cells, 'lowfit_small_lr': lowfit_low_lr}

# --------------------------------------------------------------- A1
print('\n' + '=' * 90)
print('A1  Q3 with Holm correction  (one-sided Welch t, validation macro-F1)')
print('=' * 90)
tests = []
for a, b in combinations(RANKS, 2):
    for lr in LR_L:
        A = [r['best_val_f1'] for r in lora(a, lr).values()]
        B = [r['best_val_f1'] for r in lora(b, lr).values()]
        if len(A) < 2 or len(B) < 2:
            continue
        va, vb = np.var(A, ddof=1) / len(A), np.var(B, ddof=1) / len(B)
        se = math.sqrt(va + vb)
        for w, l, sign in [(a, b, 1), (b, a, -1)]:
            diff = sign * (np.mean(A) - np.mean(B))
            if se == 0:
                p = 0.0 if diff > 0 else 1.0
            else:
                df = (va + vb) ** 2 / ((va ** 2) / (len(A) - 1) + (vb ** 2) / (len(B) - 1))
                p = float(stats.t.sf(diff / se, df))
            tests.append({'winner': w, 'loser': l, 'lr': lr, 'diff': diff, 'p': p})
m = len(tests)
order = sorted(range(m), key=lambda i: tests[i]['p'])
reject = [False] * m
for k, i in enumerate(order):
    if tests[i]['p'] <= ALPHA / (m - k):
        reject[i] = True
    else:
        break
sig = [t for t, rj in zip(tests, reject) if rj]
rev = []
for a, b in combinations(RANKS, 2):
    ab = [lrs(t['lr']) for t in sig if t['winner'] == a and t['loser'] == b]
    ba = [lrs(t['lr']) for t in sig if t['winner'] == b and t['loser'] == a]
    if ab and ba:
        rev.append((a, b, ab, ba))
print(f'  tests in family: {m}   Holm-significant wins: {len(sig)}')
print(f'  pairs with a Holm-significant win in EACH direction: {len(rev)}')
for a, b, ab, ba in rev:
    print(f'    r={a} > r={b} at {ab}   and   r={b} > r={a} at {ba}')
if not rev:
    print('  none - see power note below')
print(f'\n  Power note: with 3 seeds per arm the Welch test has ~2-4 degrees of '
      f'freedom, and\n  Holm over {m} tests requires p < {ALPHA / m:.1e} for the first '
      'rejection. A null here\n  may reflect low power rather than absence of '
      'reversal; the effect sizes are reported regardless.')
a1_pass = len(rev) > 0
out['A1'] = {'n_tests': m, 'n_significant': len(sig), 'reversal_pairs': rev,
             'pass': a1_pass}

# --------------------------------------------------------------- A3
print('\n' + '=' * 90)
print('A3  Q4 spread, computed exactly as now stated in the header')
print('=' * 90)
pref = {rk: max([lr for lr in LR_L if lora(rk, lr)],
                key=lambda lr: st.mean(r['best_val_f1'] for r in lora(rk, lr).values()))
        for rk in RANKS}
tuned = {rk: st.mean(r['test_acc'] for r in lora(rk, pref[rk]).values()) for rk in RANKS}
fixed = {lr: max(st.mean(r['test_acc'] for r in lora(rk, lr).values()) for rk in RANKS)
         - min(st.mean(r['test_acc'] for r in lora(rk, lr).values()) for rk in RANKS)
         for lr in LR_L}
ts = max(tuned.values()) - min(tuned.values())
med = st.median(fixed.values())
print(f'  tuned spread {ts:.4f} | median fixed {med:.4f} | ratio {ts / med:.2f} | '
      f'material: {ts <= 0.5 * med}')
best_fixed = min(fixed, key=fixed.get)
print(f'  smallest fixed-lr spread: {fixed[best_fixed]:.4f} at lr={lrs(best_fixed)}  '
      f'(tuned is {fixed[best_fixed] / ts:.1f}x smaller)')
out['A3'] = {'tuned_spread': ts, 'median_fixed': med, 'fixed': fixed,
             'best_single_lr': best_fixed}

# --------------------------------------------------------------- R1
print('\n' + '=' * 90)
print('R1  Q1 robustness: permutation test of the rank x lr interaction')
print('=' * 90)


def interaction_F(Y):
    a, b, n = Y.shape
    g, cell = Y.mean(), Y.mean(axis=2)
    rm, cm = cell.mean(axis=1), cell.mean(axis=0)
    ss_ab = n * ((cell - rm[:, None] - cm[None, :] + g) ** 2).sum()
    ss_e = ((Y - cell[..., None]) ** 2).sum()
    return (ss_ab / ((a - 1) * (b - 1))) / (ss_e / (a * b * (n - 1)))


rng = np.random.default_rng(0)
for name, ranks in [('all 5 ranks', RANKS), ('middle ranks {2,4,8}', [2, 4, 8])]:
    Y = np.array([[[lora(rk, lr)[s]['best_val_f1'] for s in SEEDS] for lr in LR_L]
                  for rk in ranks])
    F0 = interaction_F(Y)
    cell = Y.mean(axis=2)
    fit = cell.mean(axis=1)[:, None] + cell.mean(axis=0)[None, :] - cell.mean()
    resid = (Y - fit[..., None]).ravel()
    B, ge = 10000, 0
    for _ in range(B):
        Yp = fit[..., None] + rng.permutation(resid).reshape(Y.shape)
        ge += interaction_F(Yp) >= F0
    p = (ge + 1) / (B + 1)
    print(f'  {name:<22} F = {F0:7.2f}   permutation p = {p:.4f}  ({B:,} permutations)')
    out[f'R1_{name}'] = {'F': F0, 'p_perm': p}

json.dump(out, open('study3_posthoc_summary.json', 'w'), indent=1, default=str)
print('\nwrote study3_posthoc_summary.json')
