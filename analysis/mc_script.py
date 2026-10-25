"""Monte Carlo check of the FROZEN script's own functions (v2): error rate and power.
Worlds are built from Study 3 cell means and s.d.; 'blocks' adds a seed effect
(s.d. 0.03, shared training subsample) and a seed x rank effect (s.d. 0.02)."""
import numpy as np, math
from scipy import stats
body = open('study4_prereg_analysis.py').read()
ns = {'np': np, 'math': math, 'stats': stats, 'N_BOOT': 200, 'RANKS': [1, 2, 4, 8, 16]}
exec(body[body.index('def interaction'):body.index('summary = {')], ns)
import make_synth4 as W
def gen(scale, rng, blocks, n=5):
    V = np.empty((5, 5, n)); T = np.empty((5, 5, n))
    b = rng.normal(0, .03, n) if blocks else np.zeros(n)
    br = rng.normal(0, .02, (5, n)) if blocks else np.zeros((5, n))
    for i in range(5):
        for j in range(5):
            z = rng.standard_normal(n)
            for X, k in ((V, 'best_val_f1'), (T, 'test_acc')):
                m = W.additive(W.M[k]) + scale * (W.M[k] - W.additive(W.M[k]))
                X[i, j] = np.clip(m[i, j] + b + br[i] + W.S[k][i, j] * (0.7 * z + 0.71 * rng.standard_normal(n)), 0, 1)
    return V, T
def s1(V):
    out = []
    for lo, hi in ((0, 4), (1, 3)):
        out.append(max(ns['paired_one_sided'](V[hi, 1], V[lo, 1]), ns['paired_one_sided'](V[lo, 3], V[hi, 3])))
    o = np.argsort(out); rej = [False, False]
    if out[o[0]] <= .025:
        rej[o[0]] = True
        if out[o[1]] <= .05: rej[o[1]] = True
    return rej
rng = np.random.default_rng(2026); N = 300
print(f'{"world":<22}{"P1 v2":>8}{"P1 cluster":>12}{"P1 classicF":>13}{"P2":>7}{"replicates":>12}{"C1":>6}{"C2":>6}')
for label, sc, bl in [('null', 0, False), ('null + seed blocks', 0, True), ('half', .5, False),
                      ('half + seed blocks', .5, True), ('full', 1, False), ('full + seed blocks', 1, True)]:
    r = np.zeros(7)
    for _ in range(N):
        V, T = gen(sc, rng, bl)
        a = ns['robust_p'](V, rng, B=499)[1] < .05
        c = ns['robust_p'](V, rng, B=499, cluster=True)[1] < .05
        f = ns['interaction'](V)[2] < .05
        s = ns['spreads'](V, T); p2 = s[0] <= .5 * s[2]
        r += [a, c, f, p2, a and p2] + s1(V)
    r /= N
    print(f'{label:<22}{r[0]:>8.3f}{r[1]:>12.3f}{r[2]:>13.3f}{r[3]:>7.3f}{r[4]:>12.3f}{r[5]:>6.2f}{r[6]:>6.2f}')
