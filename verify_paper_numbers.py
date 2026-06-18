"""
Ad-hoc verifier: cross-check every numeric claim in paper/study0 against the
committed result JSON artifacts (what the code actually emits).

Run: python verify_paper_numbers.py
Not part of the repo pipeline; safe to delete.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
R = ROOT / "results" / "study0"

def load(p):
    with open(R / p) as f:
        return json.load(f)

B = load("baseline_results.json")          # default baselines (outcome->model)
F = load("final_benchmark.json")           # model->outcome ; gate_* ; bootstrap
S = load("ssl_experiment_results.json")    # baseline/pretrained/comparison/loss
C = load("corrected/benchmark_results.json")   # tuned: gate + outcomes->model
T = load("corrected/ft_transformer_tuning.json")
M = load("corrected/multiseed_results.json")
W = load("corrected/walk_forward_results.json")

OUT = ["stock_decline", "earnings_restate", "sec_enforcement", "bankruptcy"]

fails = []
def chk(label, paper, source, tol=6e-4):
    ok = abs(paper - source) < tol
    tag = "ok " if ok else "FAIL"
    if not ok:
        fails.append((label, paper, source, paper - source))
    print(f"  [{tag}] {label:52s} paper={paper:+.4f}  code={source:+.4f}  d={paper-source:+.5f}")
    return ok

print("="*90)
print("TABLE 1  tab:auroc  (default baselines + FT test-selected)  source: final_benchmark.json")
print("="*90)
t1 = {"lr_trad":[0.644,0.619,0.606,0.646],"lr_full":[0.682,0.660,0.400,0.640],
      "xgboost":[0.653,0.651,0.530,0.615],"gbt_raw":[0.677,0.670,0.602,0.561],
      "ft_scratch":[0.667,0.685,0.606,0.648]}
for m,vals in t1.items():
    for o,pv in zip(OUT,vals):
        chk(f"T1 {m}/{o}", pv, F[m][o]["auroc"])

print("="*90)
print("TABLE 2  tab:full_metrics  source: final_benchmark.json")
print("="*90)
# FT(tuned)=ft_scratch ; best baselines: stock=lr_full, earn=gbt_raw, sec=lr_trad, bank=lr_trad
t2 = [
 ("stock_decline","ft_scratch",[0.667,0.742,0.234,0.089]),
 ("stock_decline","lr_full",   [0.682,0.735,0.226,0.081]),
 ("earnings_restate","ft_scratch",[0.685,0.199,0.255,0.392]),
 ("earnings_restate","gbt_raw",   [0.670,0.190,0.227,0.359]),
 ("sec_enforcement","ft_scratch",[0.606,0.011,0.073,0.220]),
 ("sec_enforcement","lr_trad",   [0.606,0.010,0.229,0.456]),
 ("bankruptcy","ft_scratch",[0.648,0.017,0.163,0.381]),
 ("bankruptcy","lr_trad",   [0.646,0.014,0.220,0.437]),
]
for o,m,(au,ap,br,ec) in t2:
    d=F[m][o]
    chk(f"T2 {m}/{o} AUROC",au,d["auroc"]); chk(f"T2 {m}/{o} AUPRC",ap,d["auprc"])
    chk(f"T2 {m}/{o} Brier",br,d["brier"]); chk(f"T2 {m}/{o} ECE",ec,d["ece"])

print("="*90)
print("TABLE 3  tab:auroc_tuned  source: corrected/benchmark_results.json")
print("="*90)
t3 = {"lr_traditional":[0.644,0.608,0.630,0.646],"lr_full":[0.683,0.642,0.391,0.654],
      "xgboost":[0.668,0.645,0.513,0.617],"gbt_raw":[0.682,0.649,0.595,0.540],
      "ft_transformer":[0.677,0.682,0.529,0.658]}
for m,vals in t3.items():
    for o,pv in zip(OUT,vals):
        chk(f"T3 {m}/{o}", pv, C["outcomes"][o][m]["auroc"])

print("="*90)
print("TABLE 4  tab:ssl  PAPER vs ssl_experiment_results.json (the claimed source)")
print("="*90)
t4 = {"baseline":[0.664,0.671,0.481,0.668],"0.15":[0.672,0.658,0.554,0.684],
      "0.30":[0.668,0.651,0.565,0.672],"0.50":[0.647,0.663,0.480,0.654]}
for key,vals in t4.items():
    for o,pv in zip(OUT,vals):
        src = S[key][o]["auroc"] if key=="baseline" else S["pretrained"][key][o]["auroc"]
        chk(f"T4 {key}/{o}", pv, src)

print("="*90)
print("TABLE 5  tab:multiseed  source: corrected/multiseed_results.json")
print("="*90)
t5 = {"stock_decline":[0.677,0.672,0.667,0.672,0.004],
      "earnings_restate":[0.682,0.653,0.658,0.664,0.013],
      "sec_enforcement":[0.529,0.486,0.593,0.536,0.044],
      "bankruptcy":[0.658,0.681,0.658,0.666,0.011]}
for o,(s42,s123,s456,mean,std) in t5.items():
    d=M["multiseed"][o]
    chk(f"T5 {o} seed42",s42,d["per_seed_auroc"]["42"])
    chk(f"T5 {o} seed123",s123,d["per_seed_auroc"]["123"])
    chk(f"T5 {o} seed456",s456,d["per_seed_auroc"]["456"])
    chk(f"T5 {o} mean",mean,d["mean_auroc"])
    chk(f"T5 {o} std",std,d["std_auroc"])

print("="*90)
print("TABLE 6  tab:gate_tuned  source: corrected/benchmark_results.json gate")
print("="*90)
t6 = {"stock_decline":(0.668,0.677,0.009),"earnings_restate":(0.645,0.682,0.037),
      "sec_enforcement":(0.513,0.529,0.016),"bankruptcy":(0.617,0.658,0.041)}
for o,(xg,ft,dl) in t6.items():
    g=C["gate"]["detail"][o]
    chk(f"T6 {o} xgb",xg,g["xgb_auroc"]); chk(f"T6 {o} ft",ft,g["ft_auroc"])
    chk(f"T6 {o} delta",dl,g["ft_auroc"]-g["xgb_auroc"])
print(f"  gate passed={C['gate']['passed']} n_wins={C['gate']['n_wins']} (paper: 3/4)")

print("="*90)
print("TABLE 7  tab:gate_default  source: final_benchmark.json gate_*")
print("="*90)
# SSL column dropped from tab:gate_default (now scratch-only; SSL lives in tab:ssl)
t7 = {"stock_decline":(0.653,0.667,0.015),
      "earnings_restate":(0.651,0.685,0.033),
      "sec_enforcement":(0.530,0.606,0.075),
      "bankruptcy":(0.615,0.648,0.034)}
for o,(xg,fs,ds) in t7.items():
    xa=F["gate_xgb"][o]["auroc"]; sa=F["gate_scratch"][o]["auroc"]
    chk(f"T7 {o} xgb",xg,xa)
    chk(f"T7 {o} ft_scratch",fs,sa); chk(f"T7 {o} d_scratch",ds,sa-xa)

print("="*90)
print("TABLE 8  tab:bootstrap  source: final_benchmark.json bootstrap")
print("="*90)
# SSL rows dropped from tab:bootstrap (now scratch-only)
t8 = {"stock_decline":(0.015,0.008,0.021),
      "earnings_restate":(0.033,0.020,0.046),
      "sec_enforcement":(0.075,-0.013,0.162),
      "bankruptcy":(0.034,-0.022,0.092)}
for o,(sm,sl,sh) in t8.items():
    b=F["bootstrap"][o]
    chk(f"T8 {o} scr mean",sm,b["scratch_vs_xgb"]["mean"])
    chk(f"T8 {o} scr ci_lo",sl,b["scratch_vs_xgb"]["ci_low"])
    chk(f"T8 {o} scr ci_hi",sh,b["scratch_vs_xgb"]["ci_high"])

print("="*90)
print("TABLE 9  tab:walkforward  source: corrected/walk_forward_results.json")
print("="*90)
folds = W["walk_forward_folds"]
covid_idx = {3,4}  # test 2019-2020, 2020-2021
t9 = {"earnings_restate":(7,0.014,0.004,0.024),"bankruptcy":(6,0.037,-0.025,0.074),
      "stock_decline":(5,0.013,-0.004,0.040),"sec_enforcement":(4,0.024,-0.031,0.110)}
for o,(wins,mean,rlo,rhi) in t9.items():
    deltas=[fd["outcomes"][o]["ft_auroc"]-fd["outcomes"][o]["xgb_auroc"] for fd in folds]
    nwin=sum(1 for d in deltas if d>0)
    cm=sum(deltas)/len(deltas)
    print(f"  {o}: per-fold deltas = "+", ".join(f"{d:+.4f}" for d in deltas))
    print(f"    wins paper={wins} code={nwin} {'ok' if wins==nwin else 'FAIL'}")
    chk(f"T9 {o} mean",mean,cm)
    chk(f"T9 {o} range_lo",rlo,min(deltas))
    chk(f"T9 {o} range_hi",rhi,max(deltas))
    covid_wins=sum(1 for i,d in enumerate(deltas) if i in covid_idx and d>0)
    print(f"    COVID-fold(3,4) wins code={covid_wins}")

print("="*90)
print("FT TUNING  (method sec)  source: corrected/ft_transformer_tuning.json")
print("="*90)
print(f"  best_params: lr={T['best_params']['learning_rate']:.6g} n_layers={T['best_params']['n_layers']} d_token={T['best_params']['d_token']}  (paper: d=128,L=3,eta~1e-3)")
print(f"  n_trials={T['_meta']['n_trials']} len(all_trials)={len(T['all_trials'])}  (paper: 30)")

print("="*90)
print("SSL config / loss curves  source: ssl_experiment_results.json")
print("="*90)
for k in ["0.15","0.30","0.50"]:
    print(f"  loss_curves[{k}] length = {len(S['loss_curves'][k])}  (paper: 200 epochs)")
print(f"  comparison best_mask_ratio per outcome (CODE): "
      + ", ".join(f"{o}={S['comparison'][o].get('best_mask_ratio')}" for o in OUT))
print( "  paper tab:ssl best per outcome (bold)     : stock=0.15, earn=scratch, sec=0.30, bank=0.15")

print("="*90)
print("INTERNAL CONSISTENCY: SSL now reported ONLY in tab:ssl (C2 resolved)")
print("="*90)
print("  tab:gate_default and tab:bootstrap are now scratch-only; the benchmark-notebook SSL run")
print("  (final_benchmark.json ft_ssl / gate_ssl / bootstrap.ssl_vs_xgb) is no longer cited in the paper.")
print(f"  Canonical SSL (tab:ssl) r=0.50 = " + ", ".join(f"{o[:4]}:{t4['0.50'][OUT.index(o)]:.3f}" for o in OUT))

print("\n"+"="*90)
print(f"TOTAL HARD FAILURES (|paper-code| > 6e-4): {len(fails)}")
print("="*90)
for label,p,s,d in fails:
    print(f"  {label:40s} paper={p:+.4f} code={s:+.4f} diff={d:+.5f}")
