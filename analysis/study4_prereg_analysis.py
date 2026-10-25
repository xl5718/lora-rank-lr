"""
Study 4 - PRE-REGISTERED analysis of the replication (arms A, B, C).

Frozen BEFORE any Study 4 run (other than the timing pilot, which is excluded
below) was performed. sha256 recorded in PREREG_study4.txt. Nothing below may
change once data exist; extra analyses go in a separate file labelled
EXPLORATORY.

Roles of the four studies (supervisor, Oct 2026): Study 3 was the DISCOVERY /
confirmatory stage on Qwen2.5-0.5B x Financial PhraseBank. Study 4 asks only
whether its two main findings GENERALISE.

PRIMARY OUTCOMES (per arm; the only outcomes that drive the decision)
  P1  rank x lr interaction on validation macro-F1 over all 5 ranks x all
      5 lrs, alpha 0.05, tested with a HETEROSCEDASTICITY-ROBUST, SEED-BLOCKED
      test (version 2):
        model      y[rank, lr, seed] = mu[rank, lr] + b[seed] + e
                   Every seed uses the SAME training subsample (and the same
                   initialisation seed) in all 25 cells, so seed is a block.
        statistic  Wald statistic for the 16 interaction contrasts of the cell
                   means, with cell-specific variances estimated from the
                   block-adjusted residuals e = y - cell mean - seed mean + grand.
        reference  wild bootstrap under the additive null WITH the seed effects
                   held fixed: y* = additive fit + b[seed] + e * (Rademacher),
                   B = 9,999, rng seed 0.
      Sensitivity (pre-specified, reported, not decision-relevant): wild
      CLUSTER bootstrap that flips whole seeds (Webb 6-point weights), allowing
      any within-seed correlation across cells.
      Why not the Study 3 F-test: in simulations built from the Study 3 cell
      means and standard deviations, cell variances differ by more than 10x
      (diverging cells are very noisy) and the classical F-test rejected a
      TRUE null 16-18% of the time at alpha 0.05. Why blocked (v2) rather than
      v1 (which treated the 125 runs as independent): v1 kept the error rate
      but lost power when seed effects exist (0.44 vs 0.73 at half the Study 3
      effect with a seed effect of s.d. 0.03). Simulated error rates of v2:
      0.03-0.05 with no block effect, a seed effect, or a seed x rank effect.
      The classical F is still printed for comparability with Study 3.
  P2  per-rank validation tuning substantially reduces the apparent rank
      differences on TEST accuracy: tuned spread <= 0.5 x median fixed-lr
      spread (the Study 3 rule, unchanged; spreads defined below). The
      CONTINUOUS ratio is reported with a 95% interval from a bootstrap that
      resamples whole seeds (all 25 cells of a seed together), so a
      near-threshold result remains interpretable.

SECONDARY CONFIRMATORY (per arm; reported, do not drive the decision)
  S1  Two reversal contrasts fixed in advance from Study 3 (discovery):
        C1  r=16 beats r=1 at lr 3e-4   AND   r=1 beats r=16 at lr 3e-3
        C2  r=8  beats r=2 at lr 3e-4   AND   r=2 beats r=8  at lr 3e-3
      Each direction: one-sided PAIRED t-test on validation macro-F1, pairing
      the two ranks by seed (same training subsample).
      Contrast p = max of its two one-sided p (intersection-union test: a
      reversal needs BOTH directions). Holm over the 2 contrasts, alpha 0.05.

EXPLORATORY (descriptive only, never decision-relevant)
  preferred lr per rank and per-seed Spearman; low-fit / diverged map;
  middle-rank {2,4,8} interaction and stable-lr-only interaction (same robust
  blocked test).

Revision history (all before the first non-pilot run)
  v1  frozen 2026-10-12.
  v2  supervisor (16 Oct) asked that the seed/block structure be preserved.
      v1 treated cells as independent; v2 blocks on seed in P1, bootstraps
      whole seeds for the P2 interval, pairs by seed in S1, and adds the
      cluster-bootstrap sensitivity. Outcomes, thresholds, contrasts and the
      decision rule are unchanged.

FIXED DEFINITIONS
  * Analysis seeds: every seed in {42..46} that has all 25 cells of the arm.
    Fewer than 5 complete seeds is reported; fewer than 3 -> arm not analysed.
    Pilot runs (pilot = True) are excluded.
  * Preferred lr of a rank = argmax over lrs of mean validation macro-F1 over
    the analysis seeds. Test accuracy is never used for any choice.
  * tuned spread    = max - min over ranks of mean test accuracy at each rank's
                      preferred lr.
    fixed spread(x) = max - min over ranks of mean test accuracy at common lr x.
    ratio           = tuned spread / median over x of fixed spread(x).
  * LOW-FIT  = final training accuracy < 0.80.
    DIVERGED = final training accuracy <= majority-class rate of the training
               pool (the trivial constant predictor).  Both exploratory.
  * No run is ever dropped.

DECISION (supervisor, Oct 2026)
  An arm REPLICATES if P1 and P2 both hold. If all three arms replicate:
  stop expanding the main experiments and write the paper. Otherwise: discuss
  with the supervisor before any further experiment.
  The overall claim is a conjunction over arms, so no correction across arms
  is applied (intersection-union logic).

WORDING RULES (supervisor, 28 Sep 2026), applied to every printed conclusion:
  a reversal contrast that fails is "inconclusive", never "underpowered";
  nothing is called a "tie".

Usage:  python study4_prereg_analysis.py study4_results.json
"""

import json
import math
import statistics as st
import sys

import numpy as np
from scipy import stats

PATH = sys.argv[1] if len(sys.argv) > 1 else 'study4_results.json'
ARMS = ['A', 'B', 'C']
RANKS = [1, 2, 4, 8, 16]
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS = [42, 43, 44, 45, 46]
ALPHA = 0.05
LOW_FIT = 0.80
CONTRASTS = [('C1', 1, 16), ('C2', 2, 8)]      # (name, low rank, high rank)
LR_LOW, LR_HIGH = 3e-4, 3e-3
N_BOOT = 10000

ALL = [r for r in json.load(open(PATH)) if not r.get('pilot', False)]


def close(a, b):
    return abs(a - b) <= 1e-12 * max(1.0, abs(b))


def lrs(x):
    return f'{x:.0e}'


def cells(arm):
    """{(rank, lr): {seed: record}} for one arm."""
    out = {(rk, lr): {} for rk in RANKS for lr in LRS}
    for r in ALL:
        if r.get('arm') != arm:
            continue
        for lr in LRS:
            if close(r['lr'], lr) and r['rank'] in RANKS:
                out[(r['rank'], lr)][r['seed']] = r
    return out


def tensor(C, seeds, key, ranks=RANKS, lr_list=LRS):
    return np.array([[[C[(rk, lr)][s][key] for s in seeds] for lr in lr_list]
                     for rk in ranks])


def interaction(Y):
    a, b, n = Y.shape
    g, cell = Y.mean(), Y.mean(axis=2)
    rm, cm = cell.mean(axis=1), cell.mean(axis=0)
    ss_ab = n * ((cell - rm[:, None] - cm[None, :] + g) ** 2).sum()
    ss_e = ((Y - cell[..., None]) ** 2).sum()
    d1, d2 = (a - 1) * (b - 1), a * b * (n - 1)
    F = (ss_ab / d1) / (ss_e / d2) if ss_e > 0 else float('inf')
    return F, (d1, d2), float(stats.f.sf(F, d1, d2)), ss_ab / (ss_ab + ss_e)


def _block_parts(Y):
    """Cell means, seed (block) means, grand mean, block-adjusted residuals.
    Y: (..., a, b, n) with the last axis = seed."""
    cell = Y.mean(-1, keepdims=True)
    seed = Y.mean((-3, -2), keepdims=True)
    grand = Y.mean((-3, -2, -1), keepdims=True)
    return cell, seed, grand, Y - cell - seed + grand


def wald(Y):
    """Seed-blocked heteroscedastic Wald statistic for the (a-1)(b-1)
    interaction contrasts. Cell variances come from block-adjusted residuals."""
    a, b, n = Y.shape[-3:]
    cell, _, _, r = _block_parts(Y)
    mu = cell[..., 0].reshape(*Y.shape[:-3], a * b)
    v = np.maximum((r ** 2).sum(-1) / (n - 1) / n, 1e-12).reshape(*Y.shape[:-3], a * b)
    L = np.kron((np.eye(a) - 1 / a)[:-1], (np.eye(b) - 1 / b)[:-1])
    Lmu = mu @ L.T
    S = np.einsum('ij,...j,kj->...ik', L, v, L)
    return np.einsum('...i,...ij,...j->...', Lmu, np.linalg.inv(S), Lmu)


WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])


def robust_p(Y, rng, B=9999, chunk=1000, cluster=False):
    """Wild bootstrap p-value of the blocked Wald statistic under the additive
    model, with the seed effects held fixed in every bootstrap sample.
    cluster=False: Rademacher sign per observation (primary).
    cluster=True : one Webb weight per SEED, shared by all its cells (sensitivity)."""
    a, b, n = Y.shape
    cell, seed, grand, r = _block_parts(Y)
    fit = cell.mean(1, keepdims=True) + cell.mean(0, keepdims=True) - grand
    base = fit + seed - grand
    r = r * math.sqrt(n / (n - 1))
    w0, ge, done_ = wald(Y), 0, 0
    while done_ < B:
        k = min(chunk, B - done_)
        if cluster:
            w = rng.choice(WEBB, size=(k, 1, 1, n))
        else:
            w = rng.choice([-1.0, 1.0], size=(k, a, b, n))
        ge += int((wald(base[None] + r[None] * w) >= w0).sum())
        done_ += k
    return w0, (ge + 1) / (B + 1)


def spreads(V, T):
    """V, T: (rank, lr, seed) validation F1 and test acc. Returns tuned, fixed, ratio."""
    vm, tm = V.mean(axis=2), T.mean(axis=2)
    pref = vm.argmax(axis=1)
    tuned_acc = np.array([tm[i, pref[i]] for i in range(len(RANKS))])
    fixed = tm.max(axis=0) - tm.min(axis=0)
    tuned = tuned_acc.max() - tuned_acc.min()
    med = float(np.median(fixed))
    return tuned, fixed, med, tuned / med if med > 0 else float('inf'), pref, tuned_acc


def paired_one_sided(A, B):
    """p for H1: mean(A - B) > 0, A and B paired by seed (same order)."""
    d = np.asarray(A, float) - np.asarray(B, float)
    sd_ = d.std(ddof=1)
    if sd_ == 0:
        return 0.0 if d.mean() > 0 else 1.0
    return float(stats.t.sf(d.mean() / (sd_ / math.sqrt(len(d))), len(d) - 1))


def ratio_interval(V, T, rng, B=N_BOOT):
    """95% interval for the P2 ratio, resampling WHOLE seeds (block bootstrap)."""
    n = V.shape[2]
    boot = []
    for _ in range(B):
        pick = rng.integers(0, n, size=n)
        boot.append(spreads(V[:, :, pick], T[:, :, pick])[3])
    boot = np.array(boot)
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5), float(np.mean(boot <= 0.5))


summary = {'path': PATH, 'n_records_excluding_pilot': len(ALL), 'arms': {}}
print('=' * 92)
print('STUDY 4 — PRE-REGISTERED REPLICATION ANALYSIS')
print('=' * 92)

for arm in ARMS:
    C = cells(arm)
    seeds = [s for s in SEEDS if all(s in C[k] for k in C)]
    arm_recs = [r for r in ALL if r.get('arm') == arm]
    head = arm_recs[0] if arm_recs else {}
    print(f'\n{"#" * 92}\nARM {arm}:  {head.get("model", "?")}  x  {head.get("task", "?")}')
    print(f'{"#" * 92}')
    print(f'  runs on file: {len(arm_recs)}   complete seeds: {seeds}  ({len(seeds)}/5)')
    res = {'seeds': seeds, 'model': head.get('model'), 'task': head.get('task')}
    summary['arms'][arm] = res
    if len(seeds) < 3:
        print('  fewer than 3 complete seeds -> arm NOT analysed (pre-registered rule)')
        res['analysed'] = False
        continue
    res['analysed'] = True
    if len(seeds) < 5:
        print(f'  NOTE: only {len(seeds)} complete seeds; reported as a reduced replication')
    V = tensor(C, seeds, 'best_val_f1')
    T = tensor(C, seeds, 'test_acc')
    TR = tensor(C, seeds, 'train_acc')
    maj = head.get('majority_rate', float('nan'))

    # ---------------------------------------------------------------- P1
    rng = np.random.default_rng(0)
    w0, p_rob = robust_p(V, rng)
    _, p_clu = robust_p(V, rng, cluster=True)
    F, df, p, eta = interaction(V)
    p1 = p_rob < ALPHA
    print(f'\n  P1  rank x lr interaction (validation macro-F1, all ranks x all lrs)')
    print(f'      robust seed-blocked test: Wald = {w0:.1f} on 16 contrasts, '
          f'wild-bootstrap p = {p_rob:.4f}   ->  {"REPLICATES" if p1 else "does NOT replicate"}')
    print(f'      sensitivity, whole-seed cluster bootstrap: p = {p_clu:.4f}')
    print(f'      (classical F({df[0]},{df[1]}) = {F:.2f}, p = {p:.2e}, partial eta2 = '
          f'{eta:.2f}; for comparison with Study 3 only)')
    res['P1'] = {'wald': w0, 'p_robust': p_rob, 'p_cluster': p_clu, 'F': F, 'df': df,
                 'p_classical': p, 'partial_eta2': eta, 'pass': p1}

    # ---------------------------------------------------------------- P2
    tuned, fixed, med, ratio, pref, tuned_acc = spreads(V, T)
    p2 = tuned <= 0.5 * med
    print(f'\n  P2  per-rank validation tuning vs fixed lr (TEST accuracy)')
    print('      preferred lr:   ' + '   '.join(f'r={rk}: {lrs(LRS[pref[i]])}'
                                                for i, rk in enumerate(RANKS)))
    print('      tuned test acc: ' + '   '.join(f'r={rk}: {tuned_acc[i]:.4f}'
                                                for i, rk in enumerate(RANKS)))
    print('      fixed-lr spreads: ' + '   '.join(f'{lrs(l)}: {fixed[j]:.4f}'
                                                  for j, l in enumerate(LRS)))
    lo95, hi95, share = ratio_interval(V, T, np.random.default_rng(1))
    print(f'      tuned spread {tuned:.4f} | median fixed {med:.4f}')
    print(f'      RATIO = {ratio:.3f}   95% interval (whole-seed bootstrap) [{lo95:.2f}, {hi95:.2f}]'
          f'   share of resamples <= 0.5: {share:.2f}')
    print(f'      ->  {"REPLICATES" if p2 else "does NOT replicate"}  (bar: ratio <= 0.50)')
    res['P2'] = {'tuned_spread': tuned, 'fixed_spreads': dict(zip(map(lrs, LRS), fixed.tolist())),
                 'median_fixed': med, 'ratio': ratio, 'ratio_ci95': [lo95, hi95],
                 'share_le_half': share, 'pass': p2,
                 'preferred_lr': {rk: LRS[pref[i]] for i, rk in enumerate(RANKS)}}

    # ---------------------------------------------------------------- S1
    print(f'\n  S1  pre-registered reversal contrasts (secondary; IUT, Holm over 2)')
    ri = {rk: i for i, rk in enumerate(RANKS)}
    li = {lr: j for j, lr in enumerate(LRS)}
    cps = []
    for name, lo, hi in CONTRASTS:
        p_low = paired_one_sided(V[ri[hi], li[LR_LOW]], V[ri[lo], li[LR_LOW]])
        p_high = paired_one_sided(V[ri[lo], li[LR_HIGH]], V[ri[hi], li[LR_HIGH]])
        d_low = V[ri[hi], li[LR_LOW]].mean() - V[ri[lo], li[LR_LOW]].mean()
        d_high = V[ri[lo], li[LR_HIGH]].mean() - V[ri[hi], li[LR_HIGH]].mean()
        cps.append((name, lo, hi, max(p_low, p_high), p_low, p_high, d_low, d_high))
    order = sorted(range(len(cps)), key=lambda k: cps[k][3])
    rej = [False] * len(cps)
    for k, i in enumerate(order):
        if cps[i][3] <= ALPHA / (len(cps) - k):
            rej[i] = True
        else:
            break
    res['S1'] = {}
    for (name, lo, hi, pc, pl, ph, dl, dh), rj in zip(cps, rej):
        print(f'      {name}: r{hi} > r{lo} at {lrs(LR_LOW)} by {dl:+.4f} (p={pl:.4f});  '
              f'r{lo} > r{hi} at {lrs(LR_HIGH)} by {dh:+.4f} (p={ph:.4f})')
        print(f'          contrast p = {pc:.4f}  ->  '
              f'{"CONFIRMED (Holm)" if rj else "INCONCLUSIVE after correction"}')
        res['S1'][name] = {'p': pc, 'p_low': pl, 'p_high': ph, 'd_low': dl,
                           'd_high': dh, 'confirmed': rj}

    # ------------------------------------------------------- exploratory
    print('\n  EXPLORATORY (descriptive; not decision-relevant)')
    rng = np.random.default_rng(2)
    _, mid_p = robust_p(V[1:4], rng, B=1999)
    print(f'      middle ranks {{2,4,8}}, all lrs: robust p = {mid_p:.4f}')
    trm = TR.min(axis=(0, 2))
    stable_cols = [LRS[j] for j in range(len(LRS)) if trm[j] >= LOW_FIT]
    s_all_p = s_mid_p = None
    if len(stable_cols) >= 2:
        js = [li[x] for x in stable_cols]
        _, s_all_p = robust_p(V[:, js, :], rng, B=1999)
        _, s_mid_p = robust_p(V[1:4][:, js, :], rng, B=1999)
        print(f'      stable lrs {[lrs(x) for x in stable_cols]}: all ranks robust p = '
              f'{s_all_p:.4f}; middle ranks robust p = {s_mid_p:.4f}'
              f'  (low power: see study4_power_check)')
    else:
        print('      fewer than 2 fully stable lr columns')
    print(f'      fit map (L = low-fit < {LOW_FIT}, D = diverged <= majority {maj:.3f}):')
    print('        rank' + ''.join(f'{lrs(l):>10}' for l in LRS))
    for i, rk in enumerate(RANKS):
        row = f'{rk:>8}'
        for j in range(len(LRS)):
            d = int((TR[i, j] <= maj).sum())
            low = int(((TR[i, j] > maj) & (TR[i, j] < LOW_FIT)).sum())
            row += f'{(f"L{low} D{d}" if (low or d) else "ok"):>10}'
        print(row)
    rho = stats.spearmanr([rk for rk in RANKS for s in seeds],
                          [math.log10(LRS[int(V[i, :, k].argmax())])
                           for i, rk in enumerate(RANKS) for k in range(len(seeds))])
    print(f'      per-seed Spearman(rank, log preferred lr) = {rho[0]:+.2f} (p = {rho[1]:.3f})')
    res['exploratory'] = {'ratio_ci95': [lo95, hi95], 'middle_p_robust': mid_p,
                          'stable_cols': stable_cols, 'stable_p_robust': [s_all_p, s_mid_p],
                          'spearman': [rho[0], rho[1]]}
    res['replicates'] = bool(p1 and p2)

# ----------------------------------------------------------------- decision
print('\n' + '=' * 92)
print('DECISION  (supervisor, Oct 2026)')
print('=' * 92)
analysed = [a for a in ARMS if summary['arms'][a].get('analysed')]
for a in ARMS:
    r = summary['arms'][a]
    if not r.get('analysed'):
        print(f'  arm {a}: not analysed')
        continue
    print(f'  arm {a}: P1 [{"yes" if r["P1"]["pass"] else " no"}]  '
          f'P2 [{"yes" if r["P2"]["pass"] else " no"}]  ->  '
          f'{"REPLICATES" if r["replicates"] else "does not replicate"}')
all_rep = len(analysed) == 3 and all(summary['arms'][a]['replicates'] for a in ARMS)
print('\n  ' + ('ALL THREE ARMS REPLICATE -> stop expanding the main experiments; '
                'write the paper.' if all_rep else
                'NOT all arms replicate (or not all analysed) -> discuss with the '
                'supervisor before any further experiment.'))
summary['all_replicate'] = all_rep

json.dump(summary, open('study4_prereg_summary.json', 'w'), indent=1, default=str)
print('\nwrote study4_prereg_summary.json')
