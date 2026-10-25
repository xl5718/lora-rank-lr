"""
Study 2, analysis v4 — after the two follow-up experiments.

Both follow-ups overturn conclusions drawn from the main grid. This file records
what survives and what does not, and is deliberately blunt about the latter.

  3B  learning-rate sweep    -> the "low-rank underfits" mechanism was an
                               OPTIMISATION artefact, not a capacity limit.
  3A  matched-budget rerun   -> the "rank sensitivity declines as data grows"
                               pattern was an artefact of the step budget
                               growing with n. At matched budget it does not
                               decline.

Consequence: the headline of the previous write-up - a LoRA penalty at
n=250-500 - does not survive, because LoRA was run at a single learning rate
that is too small for low ranks.
"""

import json
import statistics as st

R = json.load(open('study2_results.json'))
RANKS = [1, 2, 4, 8, 16]
LRS = [3e-4, 1e-3, 3e-3]


def ms(xs):
    xs = list(xs)
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def g(**kw):
    return [r for r in R if all(
        (abs(r[k] - v) < 1e-12 if isinstance(v, float) else r[k] == v)
        for k, v in kw.items())]


def acc(**kw):
    return ms(x['test_acc'] for x in g(**kw))


def tracc(**kw):
    return ms(x['train_acc'] for x in g(**kw))


print('=' * 88)
print('STUDY 2 v4 — after the learning-rate and matched-budget follow-ups')
print(f'{len(R)} runs total')
print('=' * 88)

# ---------------------------------------------------------------- 3B
print("""
================================================================================
3B  LEARNING-RATE SWEEP — the decisive experiment
================================================================================
The main grid ran every LoRA arm at lr = 3e-4. If low-rank adapters simply need
a larger step size, the low training accuracy at n = 250-500 is an optimisation
artefact and the capacity reading is wrong.
""")
print(f'{"n":>5}{"rank":>6}' + ''.join(f'{f"lr={l:.0e}":>22}' for l in LRS))
print(f'{"":11}' + ''.join(f'{"train / test":>22}' for _ in LRS))
print('-' * 88)
for n in (250, 500):
    for rk in (1, 16):
        row = f'{n:>5}{rk:>6}'
        for lr in LRS:
            tr, _ = tracc(budget_mode='epochs', method='lora', n_requested=n,
                          rank=rk, lr=lr)
            te, _ = acc(budget_mode='epochs', method='lora', n_requested=n,
                        rank=rk, lr=lr)
            row += f'{tr:>13.3f} /{te:>7.4f}'
        print(row)
    print()

print('  FINDING 1 — the underfitting disappears with a larger learning rate.')
for n in (250, 500):
    a, _ = tracc(budget_mode='epochs', method='lora', n_requested=n, rank=1, lr=3e-4)
    b, _ = tracc(budget_mode='epochs', method='lora', n_requested=n, rank=1, lr=3e-3)
    print(f'    n={n}: r=1 train accuracy {a:.3f} at lr=3e-4  ->  {b:.3f} at lr=3e-3')
print('  The capacity-limitation reading of the main grid is REFUTED.')

print('\n  FINDING 2 — the rank ordering REVERSES with the learning rate.')
for n in (250, 500):
    lo4, _ = acc(budget_mode='epochs', method='lora', n_requested=n, rank=1, lr=3e-4)
    hi4, _ = acc(budget_mode='epochs', method='lora', n_requested=n, rank=16, lr=3e-4)
    lo3, _ = acc(budget_mode='epochs', method='lora', n_requested=n, rank=1, lr=3e-3)
    hi3, _ = acc(budget_mode='epochs', method='lora', n_requested=n, rank=16, lr=3e-3)
    print(f'    n={n}: at lr=3e-4  r=1 {lo4:.4f} < r=16 {hi4:.4f}   '
          f'(r=16 better by {hi4 - lo4:+.4f})')
    print(f'    {"":9}at lr=3e-3  r=1 {lo3:.4f} > r=16 {hi3:.4f}   '
          f'(r=1  better by {lo3 - hi3:+.4f})')
print('  A rank ablation at fixed learning rate measures LR mismatch, not rank.')

print('\n  FINDING 3 — tuned per rank, rank barely matters.')
print(f'    {"n":>5}{"r=1 best":>22}{"r=16 best":>22}{"difference":>14}')
for n in (250, 500):
    b1, l1 = max((acc(budget_mode='epochs', method='lora', n_requested=n,
                      rank=1, lr=l)[0], l) for l in LRS)
    b16, l16 = max((acc(budget_mode='epochs', method='lora', n_requested=n,
                        rank=16, lr=l)[0], l) for l in LRS)
    print(f'    {n:>5}{f"{b1:.4f} @ {l1:.0e}":>22}{f"{b16:.4f} @ {l16:.0e}":>22}'
          f'{b1 - b16:>+14.4f}')
print('    Consistent with Hu et al., who report performance largely flat in r.')

print('\n  FINDING 4 — the previous headline does not survive.')
print(f'    {"n":>5}{"full FT":>12}{"LoRA @3e-4":>14}{"LoRA tuned":>14}'
      f'{"old claim":>14}{"now":>16}')
for n in (250, 500):
    fu, _ = acc(budget_mode='epochs', method='full', n_requested=n)
    old, _ = acc(budget_mode='epochs', method='lora', n_requested=n, rank=16, lr=3e-4)
    best = max(acc(budget_mode='epochs', method='lora', n_requested=n, rank=rk, lr=l)[0]
               for rk in (1, 16) for l in LRS)
    print(f'    {n:>5}{fu:>12.4f}{old:>14.4f}{best:>14.4f}'
          f'{fu - old:>+14.4f}{fu - best:>+16.4f}')
print("""    The n=250-500 "LoRA penalty" was a comparison between a tuned full
    fine-tune and an untuned LoRA. At n=250 LoRA now leads; at n=500 they tie.

    IMPORTANT ASYMMETRY: full fine-tuning was NOT given the same sweep, so
    "LoRA beats full fine-tuning" is equally unsupported. Neither direction can
    be claimed until both sides are tuned.""")

# ---------------------------------------------------------------- 3A
print("""
================================================================================
3A  MATCHED BUDGET — does rank sensitivity really fall as data grows?
================================================================================""")
print(f'{"n":>6}{"steps":>8}' + ''.join(f'{"r=" + str(k):>10}' for k in RANKS)
      + f'{"spread":>10}')
print('-' * 88)
print('  main grid (budget grows with n):')
for n, stp in [(500, 160), (1000, 315), (2417, 760)]:
    a = [acc(budget_mode='epochs', method='lora', n_requested=n, rank=k, lr=3e-4)[0]
         for k in RANKS]
    print(f'{n:>6}{stp:>8}' + ''.join(f'{x:>10.4f}' for x in a)
          + f'{max(a) - min(a):>10.4f}')
print('  matched budget (160 steps everywhere):')
matched = {}
for n in (500, 1000, 2417):
    mode = 'epochs' if n == 500 else 'steps'
    a = [acc(budget_mode=mode, method='lora', n_requested=n, rank=k, lr=3e-4,
             max_steps=160)[0] for k in RANKS]
    matched[n] = max(a) - min(a)
    print(f'{n:>6}{160:>8}' + ''.join(f'{x:>10.4f}' for x in a)
          + f'{max(a) - min(a):>10.4f}')
print(f"""
  At matched budget the spread is {matched[500]:.4f}, {matched[1000]:.4f},
  {matched[2417]:.4f} for n = 500, 1000, 2417 — it does NOT fall with data size;
  if anything it rises slightly.

  So the decline seen in the main grid (0.095 -> 0.034 -> 0.014) was produced by
  the larger step budget at larger n, not by the extra data. The "rank
  sensitivity fades as data grows" pattern is REFUTED as stated.

  Note this is also consistent with Finding 2: at 160 steps the larger ranks
  have simply had more effective optimisation at lr=3e-4, and the low ranks are
  still undertrained.""")

# ---------------------------------------------------------------- summary
print("""
================================================================================
WHAT SURVIVES
================================================================================
  Survives:
    - The mechanics of the pipeline: stratified subsampling, fixed-size val/test,
      validation-only model selection, paired seed analysis.
    - A genuine methodological finding: LoRA rank comparisons are not meaningful
      at a fixed learning rate. The rank ordering reverses between lr=3e-4 and
      lr=3e-3, and per-rank tuning removes most of the apparent rank effect.
      Published rank ablations that hold lr fixed are open to the same objection.

  Does NOT survive:
    - "Full fine-tuning outperforms LoRA at n=250-500."  Confounded by lr.
    - "Low rank underfits, so this is a capacity limitation."  It was
      optimisation; train accuracy reaches ~1.0 at a larger lr.
    - "Rank sensitivity fades as training data grows."  Artefact of the budget.

  Required before any comparative claim:
    - The same learning-rate sweep for full fine-tuning.
    - The sweep extended to all five ranks, not just r=1 and r=16.
    - Ideally a per-(rank, n) tuned grid, which is the only configuration in
      which "rank versus data size" is a well-posed question.
""")
