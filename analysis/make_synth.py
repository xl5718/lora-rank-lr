"""Synthetic Study-3 records: one world WITH a rank x lr interaction, one WITHOUT."""
import json, math, random, sys
RANKS=[1,2,4,8,16]; LR_L=[1e-4,3e-4,1e-3,3e-3,1e-2]; LR_F=[2e-6,5e-6,2e-5,5e-5,2e-4]
SEEDS=[42,43,44]
def rec(method, rank, lr, n, seed, val, test, train):
    return dict(method=method, rank=rank, lr=lr, wd=0.01, n_requested=n, n_train=n,
                seed=seed, budget_mode='epochs', best_val_f1=val, best_val_acc=val,
                test_acc=test, test_f1=test-0.05, train_acc=train, max_steps=160)
def world(interaction, extremes_only=False):
    rnd=random.Random(0); R=[]
    for rk in RANKS:
        # preferred log-lr falls with rank when interaction=True, fixed otherwise
        if extremes_only:
            opt = {1: -2.3, 16: -3.7}.get(rk, -3.0)   # only the two ends move
        else:
            opt = (-2.5 - 0.35*math.log2(rk)) if interaction else -3.0
        for lr in LR_L:
            for s in SEEDS:
                d = math.log10(lr) - opt
                q = 0.92 - 0.10*d*d + rnd.gauss(0, 0.008)
                diverge = (interaction or extremes_only) and d > 0.9
                train = 0.60 if diverge else min(1.0, 0.95 + 0.05*(q > 0.85))
                if diverge: q = 0.55 + rnd.gauss(0,0.03)
                R.append(rec('lora', rk, lr, 500, s, q-0.05, q, train))
    for n in (250,500):
        for lr in LR_F:
            for s in SEEDS:
                q = 0.91 - 0.08*(math.log10(lr)+5)**2 + rnd.gauss(0,0.008)
                R.append(rec('full', 0, lr, n, s, q-0.05, q, 1.0))
    for s in SEEDS:  # a LoRA config at n=250 so the comparison has something to pick
        R.append(rec('lora', 16, 1e-3, 250, s, 0.80, 0.85, 1.0))
    # noise records that must be FILTERED OUT (other budget / other n / fullreg)
    R.append(dict(rec('lora', 4, 3e-4, 500, 42, 0.99, 0.99, 1.0), budget_mode='steps'))
    R.append(dict(rec('full', 0, 5e-6, 500, 42, 0.99, 0.99, 1.0), wd=0.5, method='fullreg'))
    return R
json.dump(world(True),  open('synth_interaction.json','w'))
json.dump(world(False), open('synth_null.json','w'))
json.dump(world(False, extremes_only=True), open('synth_extremes.json','w'))
print('synthetic worlds written')
