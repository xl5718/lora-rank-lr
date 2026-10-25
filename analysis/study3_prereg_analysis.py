"""
Study 3 — PRE-REGISTERED analysis of the rank x learning-rate experiment.

Written and frozen BEFORE the Study 3 results were inspected, following the
supervisor's email of 17 Sep 2026. Its sha256 is recorded in PREREG_study3.txt.
Nothing below may be changed after the data are seen; any extra analysis must
be added in a separate, clearly labelled exploratory file.

The four questions (verbatim intent from review):
  Q1  Does rank interact systematically with learning rate across the FULL set
      of ranks, not only r=1 and r=16?
  Q2  Does the preferred / stable learning-rate region shift as rank changes?
  Q3  Do conclusions about which rank is better change with the fixed lr?
  Q4  Once each rank is tuned independently on validation, do the apparent
      rank differences become much smaller?

Rules fixed in advance:
  * Hyperparameters are chosen on VALIDATION macro-F1 only. Q1-Q3 are answered
    entirely on validation. The test set is read only in Q4 and the full-FT
    comparison, after every choice is fixed.
  * Diverged runs are NEVER dropped. A run is labelled UNSTABLE if its final
    training accuracy is below 0.80 - i.e. it failed to fit a training set on
    which every stable configuration so far exceeds 0.87, and on which the
    majority-class rate is ~0.62. Unstable runs stay in every average; the
    stability map is reported as an outcome in its own right.
  * Evidence is GRADED, not pass/fail (per review):
      Q2 strong    preferred lr non-increasing in rank across all five ranks
         moderate  one local reversal of a single grid step, overall trend down
         weak      otherwise
      Q4 material  tuned spread <= 0.5 x median fixed-lr spread
  * Jason's "confined to the two extreme ranks" is operationalised directly:
    the Q1 interaction test is rerun on r in {2,4,8} only.

Usage:  python study3_prereg_analysis.py study2_results.json
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
LR_F = [2e-6, 5e-6, 2e-5, 5e-5, 2e-4]
SEEDS = [42, 43, 44]
UNSTABLE = 0.80

R = json.load(open(PATH))


def close(a, b):
    return abs(a - b) <= 1e-12 * max(1.0, abs(b))


def lora(rank, lr, n=N):
    return {r['seed']: r for r in R
            if r['budget_mode'] == 'epochs' and r['method'] == 'lora'
            and r['n_requested'] == n and r['rank'] == rank
            and close(r['lr'], lr) and close(r['wd'], 0.01)}


def full(lr, n):
    return {r['seed']: r for r in R
            if r['budget_mode'] == 'epochs' and r['method'] == 'full'
            and r['n_requested'] == n and close(r['lr'], lr)
            and close(r['wd'], 0.01)}


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float('nan')


def sd(xs):
    xs = list(xs)
    return st.stdev(xs) if len(xs) > 1 else 0.0


def lrs(x):
    return f'{x:.0e}'


out = {'path': PATH, 'n_records': len(R)}
print('=' * 90)
print('STUDY 3 — PRE-REGISTERED ANALYSIS  (rank x learning rate, n = 500)')
print('=' * 90)

# ------------------------------------------------------------- completeness
missing = [(rk, lr) for rk in RANKS for lr in LR_L if len(lora(rk, lr)) < 3]
missing_f = [(n, lr) for n in (250, 500) for lr in LR_F if len(full(lr, n)) < 3]
print(f'\nrecords in file: {len(R)}')
print(f'LoRA grid cells complete: {25 - len(missing)}/25', '' if not missing
      else f'   MISSING: {[(r, lrs(l)) for r, l in missing]}')
print(f'full-FT cells complete:   {10 - len(missing_f)}/10', '' if not missing_f
      else f'   MISSING: {[(n, lrs(l)) for n, l in missing_f]}')
balanced = not missing
out['complete'] = {'lora_missing': missing, 'full_missing': missing_f}

# -------------------------------------------------------------- stability map
print('\n' + '=' * 90)
print('STABILITY  (unstable seeds / 3;  final train acc < 0.80)  — an OUTCOME')
print('=' * 90)
print(f'{"rank":>6}' + ''.join(f'{lrs(l):>10}' for l in LR_L) + '   stable upper edge')
stab = {}
for rk in RANKS:
    row, edge = f'{rk:>6}', None
    for lr in LR_L:
        c = lora(rk, lr)
        u = sum(r['train_acc'] < UNSTABLE for r in c.values())
        stab[(rk, lr)] = (u, len(c))
        row += f'{f"{u}/{len(c)}":>10}'
        if c and u == 0:
            edge = lr
    print(row + f'   {lrs(edge) if edge else "none"}')
out['stable_upper_edge'] = {
    rk: max([lr for lr in LR_L if stab[(rk, lr)][1] and stab[(rk, lr)][0] == 0],
            default=None) for rk in RANKS}


# ------------------------------------------------------------ Q1: interaction
def anova(ranks, lr_list, key='best_val_f1'):
    Y = np.array([[[lora(rk, lr)[s][key] for s in SEEDS] for lr in lr_list]
                  for rk in ranks])
    a, b, n = Y.shape
    g = Y.mean()
    cell = Y.mean(axis=2)
    rm, cm = cell.mean(axis=1), cell.mean(axis=0)
    ss_a = b * n * ((rm - g) ** 2).sum()
    ss_b = a * n * ((cm - g) ** 2).sum()
    ss_ab = n * ((cell - rm[:, None] - cm[None, :] + g) ** 2).sum()
    ss_e = ((Y - cell[..., None]) ** 2).sum()
    df_ab, df_e = (a - 1) * (b - 1), a * b * (n - 1)
    F = (ss_ab / df_ab) / (ss_e / df_e) if ss_e > 0 else float('inf')
    return {'F': F, 'df': (df_ab, df_e), 'p': float(stats.f.sf(F, df_ab, df_e)),
            'partial_eta2': ss_ab / (ss_ab + ss_e),
            'share_between': ss_ab / (ss_a + ss_b + ss_ab)}


print('\n' + '=' * 90)
print('Q1  Does rank interact with lr across the full set of ranks?  (validation)')
print('=' * 90)
q1 = {}
if balanced:
    stable_cols = [lr for lr in LR_L if all(stab[(rk, lr)][0] == 0 for rk in RANKS)]
    tests = [('all 5 ranks, all lrs', RANKS, LR_L),
             ('all 5 ranks, stable lrs only', RANKS, stable_cols),
             ('middle ranks {2,4,8}, all lrs', [2, 4, 8], LR_L),
             ('middle ranks {2,4,8}, stable lrs', [2, 4, 8], stable_cols)]
    for name, rk, cols in tests:
        if len(cols) < 2:
            print(f'  {name:<34} n/a (fewer than 2 stable lr columns)')
            continue
        r = anova(rk, cols)
        q1[name] = r
        print(f'  {name:<34} F({r["df"][0]},{r["df"][1]}) = {r["F"]:7.2f}   '
              f'p = {r["p"]:.2e}   partial eta2 = {r["partial_eta2"]:.2f}   '
              f'interaction share = {r["share_between"]:.2f}')
    print(f'\n  stable lr columns used: {[lrs(x) for x in stable_cols]}')
    full_p = q1.get('all 5 ranks, all lrs', {}).get('p', 1.0)
    mid = q1.get('middle ranks {2,4,8}, all lrs')
    q1['any_interaction'] = full_p < 0.05
    q1['confined_to_extremes'] = bool(full_p < 0.05 and mid and mid['p'] >= 0.05)
    if full_p >= 0.05:
        verdict = 'NO interaction detected across the five ranks at all'
    elif q1['confined_to_extremes']:
        verdict = 'CONFINED to the extremes - not detectable among r in {2,4,8}'
    else:
        verdict = 'NOT confined - interaction present among the middle ranks too'
    print(f'  Verdict: {verdict}')
else:
    print('  grid incomplete - interaction test not run')
out['Q1'] = q1

# -------------------------------------------------------- Q2: preferred lr
print('\n' + '=' * 90)
print('Q2  Does the preferred lr shift with rank?  (validation macro-F1)')
print('=' * 90)
pref = {}
for rk in RANKS:
    avail = [lr for lr in LR_L if lora(rk, lr)]
    pref[rk] = max(avail, key=lambda lr: mean(r['best_val_f1']
                                              for r in lora(rk, lr).values()))
per_seed = [(rk, max([lr for lr in LR_L if s in lora(rk, lr)],
                     key=lambda lr: lora(rk, lr)[s]['best_val_f1']))
            for rk in RANKS for s in SEEDS
            if any(s in lora(rk, lr) for lr in LR_L)]
idx = {lr: i for i, lr in enumerate(LR_L)}
steps = [idx[pref[b]] - idx[pref[a]] for a, b in zip(RANKS, RANKS[1:])]
reversals = [(a, b, d) for (a, b), d in zip(zip(RANKS, RANKS[1:]), steps) if d > 0]
trend_down = idx[pref[RANKS[0]]] > idx[pref[RANKS[-1]]]
if not reversals and trend_down:
    grade = 'STRONG'
elif (len(reversals) == 1 and reversals[0][2] == 1 and trend_down):
    grade = 'MODERATE'
else:
    grade = 'WEAK'
_x = [r for r, _ in per_seed]
_y = [math.log10(l) for _, l in per_seed]
if len(set(_y)) > 1 and len(set(_x)) > 1:
    rho, rho_p = stats.spearmanr(_x, _y)
else:
    rho, rho_p = float('nan'), float('nan')   # every seed picked the same lr
print('  pooled preferred lr:  ' + '   '.join(f'r={rk}: {lrs(pref[rk])}' for rk in RANKS))
print(f'  local reversals: {len(reversals)} '
      + (str([(f"r{a}->r{b}", f"+{d} step") for a, b, d in reversals]) if reversals else ''))
print('  per-seed Spearman(rank, log lr*) = ' +
      (f'{rho:+.2f}  (p = {rho_p:.3f}, {len(per_seed)} seed-level picks)'
       if not math.isnan(rho) else 'undefined - every seed picked the same lr'))
print('  stable upper edge:    ' + '   '.join(
    f'r={rk}: {lrs(out["stable_upper_edge"][rk]) if out["stable_upper_edge"][rk] else "none"}'
    for rk in RANKS))
print(f'  EVIDENCE: {grade}')
out['Q2'] = {'preferred_lr': pref, 'reversals': reversals, 'grade': grade,
             'spearman': [rho, rho_p]}

# ---------------------------------------------- Q3: does the winner change?
print('\n' + '=' * 90)
print('Q3  Does "which rank is better" depend on the fixed lr?  (validation)')
print('=' * 90)
order = {}
for lr in LR_L:
    sc = {rk: mean(r['best_val_f1'] for r in lora(rk, lr).values())
          for rk in RANKS if lora(rk, lr)}
    order[lr] = sc
    best = max(sc, key=sc.get)
    print(f'  lr={lrs(lr)}  winner r={best:<3} ranking: '
          + ' > '.join(f'r{k}' for k in sorted(sc, key=sc.get, reverse=True)))
winners = {lr: max(order[lr], key=order[lr].get) for lr in LR_L if order[lr]}
taus = []
for a, b in combinations(LR_L, 2):
    common = [rk for rk in RANKS if rk in order[a] and rk in order[b]]
    if len(common) >= 3:
        t, _ = stats.kendalltau([order[a][k] for k in common],
                                [order[b][k] for k in common])
        taus.append((a, b, t))
neg = [(lrs(a), lrs(b), round(t, 2)) for a, b, t in taus if t < 0]
print(f'\n  distinct winning ranks across lrs: {sorted(set(winners.values()))}'
      '   (descriptive only - satisfiable by noise)')
print(f'  lr pairs with Kendall tau < 0: {len(neg)} of {len(taus)}'
      '   (descriptive only)')


# Decision criterion. A changing "winner" is NOT used: when ranks are truly equal
# the winner at each lr is a coin toss, so it changes under pure noise (verified
# on a synthetic null world before freezing). What counts is a SIGNIFICANT
# reversal: ranks a, b and learning rates x, y such that a beats b at x AND
# b beats a at y, each by more than 2 standard errors of the seed-mean difference.
def gap(a, b, lr):
    A = [r['best_val_f1'] for r in lora(a, lr).values()]
    B = [r['best_val_f1'] for r in lora(b, lr).values()]
    if len(A) < 2 or len(B) < 2:
        return None
    se = math.sqrt(sd(A) ** 2 / len(A) + sd(B) ** 2 / len(B))
    return mean(A) - mean(B), se


sig_rev = []
for a, b in combinations(RANKS, 2):
    wins_a = [lr for lr in LR_L if (g := gap(a, b, lr)) and g[0] > 2 * g[1] > 0]
    wins_b = [lr for lr in LR_L if (g := gap(b, a, lr)) and g[0] > 2 * g[1] > 0]
    if wins_a and wins_b:
        sig_rev.append((a, b, [lrs(x) for x in wins_a], [lrs(x) for x in wins_b]))
print(f'\n  SIGNIFICANT rank reversals (each direction > 2 SE): {len(sig_rev)} '
      f'of {len(list(combinations(RANKS, 2)))} rank pairs')
for a, b, wa, wb in sig_rev:
    print(f'    r={a} > r={b} at lr {wa}   but   r={b} > r={a} at lr {wb}')
out['Q3'] = {'winners': winners, 'n_distinct_winners': len(set(winners.values())),
             'reversed_pairs_descriptive': neg, 'significant_reversals': sig_rev}

# --------------------------------- Q4: does tuning remove the rank effect?
print('\n' + '=' * 90)
print('Q4  After per-rank tuning on validation, do rank differences shrink?  (TEST)')
print('    [first point at which the test set is read]')
print('=' * 90)
tuned = {rk: mean(r['test_acc'] for r in lora(rk, pref[rk]).values()) for rk in RANKS}
tuned_seed = {}
for rk in RANKS:
    vals = []
    for s in SEEDS:
        opts = [lr for lr in LR_L if s in lora(rk, lr)]
        if opts:
            b = max(opts, key=lambda lr: lora(rk, lr)[s]['best_val_f1'])
            vals.append(lora(rk, b)[s]['test_acc'])
    tuned_seed[rk] = mean(vals)
spread_t = max(tuned.values()) - min(tuned.values())
spread_ts = max(tuned_seed.values()) - min(tuned_seed.values())
fixed = {}
for lr in LR_L:
    v = [mean(r['test_acc'] for r in lora(rk, lr).values()) for rk in RANKS if lora(rk, lr)]
    if len(v) == len(RANKS):
        fixed[lr] = max(v) - min(v)
med = st.median(fixed.values()) if fixed else float('nan')
noise = st.median(sd(r['test_acc'] for r in lora(rk, lr).values())
                  for rk in RANKS for lr in LR_L
                  if lora(rk, lr) and stab[(rk, lr)][0] == 0)
print('  tuned test acc per rank:  ' + '   '.join(f'r={rk}: {tuned[rk]:.4f}' for rk in RANKS))
print(f'  spread after tuning:  {spread_t:.4f}  (pooled selection)   '
      f'{spread_ts:.4f}  (per-seed selection)')
print('  spread at each fixed lr: ' + '   '.join(f'{lrs(l)}: {s:.4f}' for l, s in fixed.items()))
print(f'  median fixed-lr spread: {med:.4f}   ratio tuned/median = {spread_t / med:.2f}')
print(f'  fixed lrs with a LARGER spread than tuned: '
      f'{sum(s > spread_t for s in fixed.values())} of {len(fixed)}')
print(f'  reference: median seed s.d. in stable cells = {noise:.4f}')
material = spread_t <= 0.5 * med
print(f'  EVIDENCE: {"MATERIAL reduction" if material else "NOT a material reduction"}'
      f'  (pre-set bar: tuned <= 0.5 x median fixed)')
out['Q4'] = {'tuned': tuned, 'spread_tuned': spread_t, 'spread_tuned_per_seed': spread_ts,
             'fixed_spreads': fixed, 'median_fixed': med, 'material': material,
             'seed_noise': noise}

# ------------------------------------------------- full FT, now tuned too
print('\n' + '=' * 90)
print('FULL FINE-TUNING vs LoRA, both tuned on validation  (TEST, paired by seed)')
print('=' * 90)
cmp_ = {}
for n in (250, 500):
    favail = [lr for lr in LR_F if full(lr, n)]
    if not favail:
        print(f'  n={n}: no full-FT sweep data'); continue
    f_lr = max(favail, key=lambda lr: mean(r['best_val_f1'] for r in full(lr, n).values()))
    f_uns = sum(r['train_acc'] < UNSTABLE for lr in favail for r in full(lr, n).values())
    cfgs = sorted({(r['rank'], r['lr']) for r in R
                   if r['budget_mode'] == 'epochs' and r['method'] == 'lora'
                   and r['n_requested'] == n and close(r['wd'], 0.01)})
    l_rk, l_lr = max(cfgs, key=lambda c: mean(r['best_val_f1']
                                              for r in lora(c[0], c[1], n).values()))
    F, L = full(f_lr, n), lora(l_rk, l_lr, n)
    d = [F[s]['test_acc'] - L[s]['test_acc'] for s in SEEDS if s in F and s in L]
    cmp_[n] = {'full_lr': f_lr, 'lora': (l_rk, l_lr), 'diff': mean(d), 'sd': sd(d),
               'seeds_full_wins': sum(x > 0 for x in d), 'lora_configs_searched': len(cfgs)}
    print(f'  n={n}: full lr*={lrs(f_lr)} ({f_uns} unstable runs in sweep)   '
          f'LoRA r*={l_rk}, lr*={lrs(l_lr)}  [{len(cfgs)} LoRA configs searched]')
    print(f'         full - LoRA = {mean(d):+.4f} ± {sd(d):.4f}   '
          f'full wins {sum(x > 0 for x in d)}/{len(d)} seeds')
print('  Note: at n=250 the LoRA search covers fewer configurations than at n=500.')
out['full_vs_lora'] = cmp_

# ------------------------------------------------------------------ verdict
print('\n' + '=' * 90)
print('DECISION  (supervisor, 17 Sep: proceed only if consistently positive)')
print('=' * 90)
q1_ok = bool(q1 and q1.get('all 5 ranks, all lrs', {}).get('p', 1) < 0.05
             and not q1.get('confined_to_extremes', True))
q2_ok = grade in ('STRONG', 'MODERATE')
q3_ok = len(sig_rev) > 0
q4_ok = material
for name, ok in [('Q1 systematic interaction, not confined to extremes', q1_ok),
                 ('Q2 preferred lr shifts with rank', q2_ok),
                 ('Q3 winning rank depends on fixed lr', q3_ok),
                 ('Q4 tuning materially shrinks rank differences', q4_ok)]:
    print(f'  [{"yes" if ok else " no"}] {name}')
all_ok = q1_ok and q2_ok and q3_ok and q4_ok
print('\n  ' + ('CONSISTENTLY POSITIVE -> next: replicate on one more model and '
                '1-2 more tasks' if all_ok else
                'NOT consistently positive -> discuss with supervisor before any '
                'further experiments'))
out['decision'] = {'Q1': q1_ok, 'Q2': q2_ok, 'Q3': q3_ok, 'Q4': q4_ok, 'proceed': all_ok}

json.dump(out, open('study3_prereg_summary.json', 'w'), indent=1, default=str)
print('\nwrote study3_prereg_summary.json')
