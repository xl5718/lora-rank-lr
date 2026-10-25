"""Synthetic worlds for validating study4_prereg_analysis.py BEFORE freezing.
Cell means / s.d. are taken from Study 3 (0.5B x FPB, 3 seeds)."""
import json, numpy as np
R = json.load(open('study2_results.json'))
RANKS = [1, 2, 4, 8, 16]; LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
cl = lambda a, b: abs(a - b) <= 1e-12 * max(1, abs(b))
def c(rk, lr):
    return [r for r in R if r['budget_mode'] == 'epochs' and r['method'] == 'lora'
            and r['n_requested'] == 500 and r['rank'] == rk and cl(r['lr'], lr) and cl(r['wd'], 0.01)]
KEYS = ['best_val_f1', 'test_acc', 'train_acc']
M = {k: np.array([[np.mean([r[k] for r in c(rk, l)]) for l in LRS] for rk in RANKS]) for k in KEYS}
S = {k: np.array([[np.std([r[k] for r in c(rk, l)], ddof=1) for l in LRS] for rk in RANKS]) for k in KEYS}
for k in KEYS:
    S[k] = np.maximum(S[k], np.median(S[k]))

def additive(X):
    return X.mean(1, keepdims=True) + X.mean(0, keepdims=True) - X.mean()

def world(scale, seed, seeds=(42, 43, 44, 45, 46), arms='ABC', drop=()):
    rng = np.random.default_rng(seed); out = []
    for a in arms:
        for k_s, s in enumerate(seeds):
            for i, rk in enumerate(RANKS):
                for j, lr in enumerate(LRS):
                    if (a, s, rk, lr) in drop:
                        continue
                    rec = dict(arm=a, model='synthetic', task='synthetic', rank=rk, lr=lr,
                               seed=s, pilot=False, majority_rate=0.6215)
                    z = rng.standard_normal()      # shared within a run (val/test correlated)
                    for k in KEYS:
                        m = additive(M[k]) + scale * (M[k] - additive(M[k]))
                        v = m[i, j] + S[k][i, j] * (0.7 * z + 0.71 * rng.standard_normal())
                        rec[k] = float(np.clip(v, 0, 1))
                    out.append(rec)
    return out

if __name__ == '__main__':
    json.dump(world(1.0, 1), open('w_interaction.json', 'w'))
    json.dump(world(0.0, 2), open('w_null.json', 'w'))
    json.dump(world(0.5, 3), open('w_half.json', 'w'))
    w = world(1.0, 4, drop={('C', 46, 1, 1e-4), ('C', 46, 16, 1e-2)})
    w += [dict(arm='A', model='synthetic', task='synthetic', rank=16, lr=1e-3, seed=990,
               pilot=True, best_val_f1=0.0, test_acc=0.0, train_acc=0.0, majority_rate=.62)]
    json.dump(w, open('w_incomplete_pilot.json', 'w'))
    # Study 3 itself, re-expressed as arm B with its 3 seeds
    s3 = [dict(arm='B', model='Qwen/Qwen2.5-0.5B', task='fpb (Study 3)', rank=r['rank'], lr=r['lr'],
               seed=r['seed'], pilot=False, majority_rate=2146 / 3453,
               **{k: r[k] for k in KEYS})
          for rk in RANKS for l in LRS for r in c(rk, l)]
    json.dump(s3, open('w_study3_as_B.json', 'w'))
    print('worlds written')
