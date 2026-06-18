# Study 0 Paper — Reproduction & Consistency Audit

**Date:** 2026-06-17
**Scope:** Every numeric claim in `paper/study0/main.tex` + included sections, cross-checked
against the committed result artifacts under `results/study0/` (i.e. *what the code emits*),
against the source data parquets, and for internal self-consistency. Reproduction path per
`REPRODUCE.md`.

**Method:** values were checked programmatically, not by eye. The table cross-check script is
`verify_paper_numbers.py` (repo root; safe to delete). Data-section counts were recomputed from
`data/raw/xbrl_features.parquet ⋈ data/processed/label_database.parquet` and from the actual
`build_feature_matrix` pipeline. Tolerance for a "match": `|paper − code| < 6e-4` (i.e. agrees to
the 3 decimals printed in the paper).

**Bottom line:** Tables 1, 2, 3, 5, 7, 8 and the gate (Table tab:gate_tuned) reproduce **exactly**;
the walk-forward table and all data-section counts reproduce (one 0.001 mis-round). **One table — the
SSL table (`tab:ssl`, Table 6) — does not reproduce at all** and drives several headline SSL claims
in the abstract/results/discussion. There is also a real reproducibility-pipeline gap on
`final_benchmark.json`. Details and severities below.

**Update (2026-06-17) — corrections applied.** The four deterministic, single-correct-value errors
(**M2–M5**) have been fixed directly in the paper (marked ✅ below).

**Update (2026-06-18) — SSL rerun complete; C1 RESOLVED.** Per the author's decision, the SSL pipeline
was re-run (`scripts/run_ssl_experiment.py`, seed 42, d=192, 200 epochs × {0.15,0.30,0.50},
~18 h CPU), regenerating `ssl_experiment_results.json`. `tab:ssl`, the abstract, §5.2, and §6.2/§6.3
were rewritten from the fresh numbers, and `ssl_loss_curves.png` was regenerated. **`verify_paper_numbers.py`
now reports 0 hard failures across all tables** (including `tab:ssl`). The fresh SSL story is materially
weaker than the orphaned one (see C1 resolution). **C2 is now also RESOLVED** — the redundant
benchmark-notebook SSL run was removed from the head-to-head default benchmark (`tab:gate_default`,
`tab:bootstrap`, `fig:final_benchmark`, `fig:bootstrap_ci`, and the "six models" framing), so SSL is
reported in exactly one place: the canonical ablation `tab:ssl`/§5.2. Paper rebuilds (21 pp, exit 0)
with **0 verifier failures** and no undefined references.

---

## CRITICAL — must fix before release

### C1. ✅ RESOLVED — SSL results table (`tab:ssl`, §5.2) was not reproducible from the committed artifact
**Resolution (2026-06-18):** re-ran the SSL pipeline and rewrote `tab:ssl` + all SSL claims from the
fresh `ssl_experiment_results.json`. New (reproduced) values, with the honest, weaker story they tell:

| init | stock | earn | sec | bank |
|---|---|---|---|---|
| Scratch | 0.664 | **0.671** | 0.481 | 0.668 |
| r=0.15 | **0.672** | 0.658 | 0.554 | **0.684** |
| r=0.30 | 0.668 | 0.651 | **0.565** | 0.672 |
| r=0.50 | 0.647 | 0.663 | 0.480 | 0.654 |

SSL gain (best vs scratch): sec **+0.085** (r=0.30), bank **+0.016** (r=0.15), stock **+0.009** (r=0.15),
earnings **none** (all ratios below scratch). Scratch `sec_enforcement` is 0.481 (below chance, single
seed, 96 positives); even boosted (0.565) it stays below the LR-traditional baseline (0.606). The old
table's headline (+0.093 at r=0.50, "substantially exceeds LR-traditional 0.606") was unreproducible and
has been removed. The original-finding analysis below is retained for the record.

**Original finding —** Claim location: `sections/results.tex` Table `tab:ssl`; propagated to the abstract,
`sections/discussion.tex` §6.2 / §6.3, and the SSL deltas in §5.2.
Claimed source (REPRODUCE.md & artifact map): `results/study0/ssl_experiment_results.json`.

15 of the 16 AUROC cells in `tab:ssl` disagree with that JSON (the 1 "match" is coincidence).
The `sec_enforcement` column is off by up to **0.206 AUROC**:

| cell | paper | code (ssl_experiment_results.json) | Δ |
|---|---|---|---|
| Scratch / sec_enforcement | 0.552 | 0.482 | +0.070 |
| Scratch / bankruptcy | 0.627 | 0.662 | −0.035 |
| r=0.15 / stock_decline | 0.672 | 0.658 | +0.014 |
| r=0.30 / sec_enforcement | 0.613 | 0.407 | +0.206 |
| r=0.50 / sec_enforcement | 0.645 | 0.481 | +0.164 |
| r=0.50 / stock_decline | 0.662 | 0.667 | −0.005 |
| …(15/16 cells fail)… | | | |

The **qualitative conclusion is reversed**, not just the digits:
- Paper (§5.2): *"the strongest SSL result is on `sec_enforcement`, where r=0.50 (AUROC 0.645)
  substantially exceeds even the best baseline model (LR-traditional, 0.606)."*
- Committed artifact: best SSL `sec_enforcement` is **r=0.15 at AUROC 0.555**, which **does not**
  exceed LR-traditional (0.606). `ssl_experiment_results.json["comparison"]` reports
  `best_mask_ratio = 0.15` for `sec_enforcement` **and** `bankruptcy`; the paper bolds r=0.50 for
  `sec_enforcement`. `generate_final_benchmark.py` likewise hard-recommends `mask_ratio=0.15`.

**Downstream claims that inherit this and are therefore unsupported by the committed data:**
- Abstract: *"+0.093 AUROC on SEC enforcement actions at mask ratio 0.50."*
- §5.2: *"largest gains on `sec_enforcement` (+0.092 … at r=0.50)"*, *"bankruptcy (+0.052 at r=0.15)"*,
  *"`stock_decline`, r=0.15 … (+0.009)"*, *"`earnings_restate`, r=0.50 … (+0.005)."*
- §6.2: *"most dramatic improvement on `sec_enforcement` (+0.093 AUROC at r=0.50)."*
- §6.3: *"r=0.50 yields the strongest gains on the most challenging outcomes."*

**Root cause (from git history):** the SSL-table values were written in the **first** paper commit
`6fb758d` and never updated. The SSL code was later changed in `f4073ad`
(*"fix(ATS-217, ATS-218): reconcile SSL pipeline …"*) whose own message says
**"Requires rerun to regenerate corrected numbers."** That rerun was apparently never committed:
`ssl_experiment_results.json` was last written at `81f84b3` (before `f4073ad`), and the value
`0.645` does not appear as a test-set AUROC in *any* committed artifact. So the paper's SSL table is
an orphaned old-run snapshot.

**Remediation (pick one):**
1. Re-run `scripts/run_ssl_experiment.py` with the reconciled pipeline, regenerate
   `ssl_experiment_results.json`, and rewrite `tab:ssl` + all SSL deltas/claims from it; **or**
2. If the d=192 SSL numbers in `final_benchmark.json` (`ft_ssl`, the r=0.50 row) are the intended
   canonical SSL result, rebuild `tab:ssl` from a single regenerated per-mask-ratio artifact and make
   the r=0.50 column consistent with `tab:gate_default` (see C2).

Either way the SSL narrative (which mask ratio wins, and whether SSL beats the baseline on
`sec_enforcement`) must be re-derived, because the committed data currently contradicts it.

### C2. ✅ RESOLVED — "FT SSL (r=0.50)" disagreed across tables (two different SSL runs)
**Resolution (2026-06-18):** removed the redundant benchmark-notebook SSL variant (`final_benchmark.json`
`ft_ssl`) from the head-to-head default benchmark — dropped the FT(SSL) column from `tab:gate_default`,
the FT-SSL rows from `tab:bootstrap`, regenerated `fig:bootstrap_ci` (scratch-only) and
`fig:final_benchmark` (5 models), and changed the "six models" framing to five (`method.tex` §3.5,
`results.tex` §5.3). SSL is now reported only in the canonical ablation (`tab:ssl`/§5.2). Nothing
load-bearing was lost: the gate rests on the equal-budget `tab:gate_tuned`, and the abstract's
"+0.015 to +0.075 / significant on two" is the FT-**scratch** vs default-XGB result (unchanged). The
original analysis is retained below.

**Original finding —** `tab:ssl` (r=0.50 row, now the **fresh** dedicated SSL ablation) and `tab:gate_default` /
`tab:bootstrap` (FT (SSL) column, from `final_benchmark.json` — a **separate** notebook-04 SSL run)
both claim to be the SSL-pretrained FT-Transformer at r=0.50, d=192, but report different AUROCs. The
gap **widened** after the C1 rerun (the fresh run is the weaker, more honest one):

| outcome | `tab:ssl` r=0.50 (fresh) | `tab:gate_default` FT(SSL) (final_benchmark) | Δ |
|---|---|---|---|
| stock_decline | 0.647 | 0.670 | −0.023 |
| earnings_restate | 0.663 | 0.675 | −0.012 |
| sec_enforcement | 0.480 | 0.602 | −0.122 |
| bankruptcy | 0.654 | 0.672 | −0.018 |

Each table individually reproduces from its own committed source, so this is an **internal
two-run inconsistency**, not a paper-vs-code error. But it is a credibility risk: a reader sees the
same nominal model with sec_enforcement = 0.480 in one table and 0.602 in another.

**Recommended fix (author decision — not auto-applied):** the dedicated ablation (`tab:ssl`, §5.2) is
now the canonical SSL result. The SSL column in `tab:gate_default` and the SSL rows in `tab:bootstrap`
come from a *superseded* notebook-04 run that **overstates** the SSL benefit vs the fresh ablation
(it would show SSL beating default XGB 4/4, whereas the fresh r=0.50 vs default XGB is only ~2/4).
Cleanest: **drop the FT(SSL) column from `tab:gate_default`, the FT-SSL rows from `tab:bootstrap`, and
the SSL series from `fig:bootstrap_ci`** — the gate rests on the equal-budget `tab:gate_tuned`, and the
"+0.015 to +0.075 / significant on two" abstract claim is about FT-**scratch** vs default XGB (unaffected).
Alternative (more work): re-run notebook 04's SSL with per-sample score capture to recompute consistent
paired-bootstrap CIs. I can do the drop-and-regenerate on request; it touches two tables, one figure,
and one sentence in §5.3.

### C3. The committed artifacts themselves conflict — there is no single "code value" for the FT model
This is *why* C1/C2 cannot be mechanically auto-fixed. The same nominal FT model has **three different**
committed AUROCs. From-scratch FT (d=192), `sec_enforcement`:

| from-scratch FT | `final_benchmark.json` ft_scratch | `ssl_experiment_results.json` baseline | paper `tab:ssl` |
|---|---|---|---|
| stock_decline | 0.667 | 0.667 | 0.663 |
| earnings_restate | 0.685 | 0.677 | 0.679 |
| **sec_enforcement** | **0.606** | **0.482** | **0.552** |
| bankruptcy | 0.648 | 0.662 | 0.627 |

**Post-rerun correction (2026-06-18):** the 2026-06-18 rerun **reproduced** the sub-chance scratch
`sec_enforcement` (0.481, vs the old 0.482) — so `ssl_experiment_results.json` was **not** stale/buggy
as first suspected; the below-chance value is a *genuine* single-seed result for this extremely unstable
96-positive outcome (recall multi-seed std = 0.044, range 0.486–0.593). The real divergence is therefore
that the **SSL experiment's scratch run** (seed 42, via `_train_and_evaluate`) and the **benchmark
notebook's scratch run** (notebook 04) are two different single-seed draws of a high-variance setup,
landing at **0.481 vs 0.606** on `sec_enforcement`. Both are legitimately "what the code returns" — they
just disagree. This is the root of C2: `tab:ssl` (SSL-experiment run) and `tab:gate_default`/`tab:bootstrap`
(benchmark-notebook run) report different numbers for nominally the same model. It was the **paper's old
`tab:ssl`** (0.552 / 0.645) that was orphaned and matched neither run.

**The only reproducible SSL evidence** is `final_benchmark.json` (`ft_ssl` = r=0.50 vs `ft_scratch`),
which Tables 1/7/8 already match. Its verdict on SSL is the opposite of the paper's headline:

| outcome | scratch | SSL r=0.50 | Δ (validated) |
|---|---|---|---|
| stock_decline | 0.667 | 0.670 | **+0.002** |
| earnings_restate | 0.685 | 0.675 | **−0.010** |
| sec_enforcement | 0.606 | 0.602 | **−0.004** |
| bankruptcy | 0.648 | 0.672 | **+0.023** |

i.e. by the reproducible artifact, SSL (r=0.50) gives a meaningful lift **only on bankruptcy
(+0.023)**, is flat on stock, and slightly **hurts** earnings and `sec_enforcement` — directly
contradicting the abstract/§6.2 "+0.093 on sec_enforcement" claim. **No committed artifact supports
the paper's SSL headline.**

---

## HIGH — reproducibility pipeline does not match the docs

### H1. `generate_final_benchmark.py` does not (and refuses to) produce the committed `final_benchmark.json`
`REPRODUCE.md` step 3 and the artifact map credit `scripts/generate_final_benchmark.py` with writing
`final_benchmark.json` (→ Table 1, default-baseline gate, bootstrap CIs). In fact the committed script:
- **Refuses to overwrite** the file by default — it raises
  *"Refusing to overwrite … it contains the paper's paired-bootstrap results, which this (analytical)
  script does NOT reproduce."*
- Emits a **different schema** (`metrics` / `bootstrap_cis` / `gate.outcome_detail`) than the committed
  artifact (`lr_trad…ft_ssl` / `gate_scratch|ssl|xgb` / `bootstrap`).
- Computes **Hanley–McNeil analytical CIs**, not the **paired bootstrap (n=1000, seed 42)** the paper
  reports (`tab:bootstrap`) and that the committed file records (`_meta.ci_method="paired_bootstrap"`).
- Sources FT from `ssl_experiment_results.json["baseline"]` (e.g. stock_decline 0.6666), which **differs**
  from the committed `final_benchmark.json["ft_scratch"]` (0.6674).

**Consequence:** a reader following `REPRODUCE.md` cannot regenerate the Table 1/2/7/8 numbers; they get
an error, or (with `--force`) a different artifact. The paper's numbers match the *committed* JSON, but
that JSON is not reproducible from the documented script. The actual producer appears to be a notebook
(`experiments/study0/notebooks_archive/04_final_benchmark.ipynb`).
**Remediation:** either point `REPRODUCE.md` at the real producer, restore a script that reproduces the
paired-bootstrap artifact, or commit the prediction arrays (`--scores-dir`) so the bootstrap is
reproducible. The same script's internal gate logic (`ft_ssl_0.15` vs XGB, ≥3/5, with skip rules) is a
stale criterion that differs from the paper's gate (FT vs XGB, ≥3/4) — worth removing/aligning to avoid
confusion.

---

## MINOR — rounding / wording / internal inconsistencies

### M1. §5.2 SSL deltas inconsistent with abstract/discussion (rounding)
Even taking `tab:ssl` at face value: §5.2 says `sec_enforcement` "+0.092" while the abstract and §6.2
say "+0.093" (0.645−0.552 = 0.093). §5.2 says `earnings_restate` "+0.005" while the table gives
0.685−0.679 = 0.006. (Subsumed by C1, but the internal rounding disagreement should not survive the C1
fix.)

### M2. ✅ FIXED — Walk-forward range lower bound mis-rounded (`tab:walkforward`)
`stock_decline` "Range of Δ" lower bound is printed **−0.005**; the actual minimum fold delta is
**−0.00447** (fold 3, test 2019–2020), which rounds to **−0.004**. Off by 0.001. Every other
walk-forward number reproduces: win counts 7/6/5/4, means +0.014/+0.037/+0.013/+0.024, other range
bounds, and the COVID-fold splits all match `walk_forward_results.json`.

### M3. ✅ FIXED — `fig:walkforward` caption overstated
Caption: *"the `earnings_restate` advantage is flat and consistently above the margin."* Fold 5
(test 2021–2022) Δ = **+0.0037**, below the 0.01 gate margin, and the table's own range lower bound
(+0.004) is below the margin. "FT wins all seven folds" (Δ>0) is correct; "consistently above the
[0.01] margin" is not. Suggest: *"…consistently positive, dipping toward the margin only in 2021–2022."*

### M4. ✅ FIXED — §5.1 "AUPRC < 0.01 across all models" (sec_enforcement) is false
Default-benchmark `sec_enforcement` AUPRC: LR-trad **0.0102** and FT-scratch **0.0112** exceed 0.01
(the paper's own `tab:full_metrics` lists FT sec AUPRC = 0.011). Reword to e.g. "AUPRC ≈ 0.01" or
"AUPRC < 0.02 across all models" (max is 0.0112).

### M5. ✅ FIXED — `sections/release.tex` "5 YoY changes"
Release bullet 2 lists *"16 raw features, 12 financial ratios, **5 YoY changes**."* The data section
(`§2.3`) defines a year-over-year change for **each of the 16 raw features** (16 YoY features; component
of the 45-feature pre-pruning count). "5" is wrong — should be **16**. (5 is the income-statement raw
count; likely a copy error.)

### M6. Data: "approximately 8,000 unique companies" not supported by committed data
`§2.1` states the universe *"contains approximately 8,000 unique companies."* The committed
`data/raw/company_universe.parquet` has **13,521** unique CIKs; the modeling set (after the
inner XBRL⋈label merge) has **4,463** unique CIKs; exchange-listed only ≈ 4,615. None is ≈ 8,000.
Does not affect any result (the modeling counts below all reproduce), but the descriptive figure is
~69% below the committed universe size. Recommend recomputing the stated count or clarifying which
subset "≈8,000" refers to.

### M7. "10 traditional ratios" / "16 raw" are nominal, not effective (coverage pruning)
`§3.2` describes LR-traditional as *"10 traditional financial ratios only"* and GBT-raw as *"raw XBRL
features"* (16). After the documented 50%-coverage pruning, `build_feature_matrix` retains **9** of the
10 traditional ratios and **15** of the 16 raw features in the matrices the models actually fit. The
stated sets are the definitions; the effective counts are 9/15. Low severity (pruning is documented),
but a careful reader reconciling code to text will notice.

### M8. Promised sector-stratified reporting not shown
`§4.2` states *"we report AUROC and AUPRC stratified by Fama–French 12-industry sector."* No sector
table or figure appears in the Results sections, although the data exists
(`results/study0/corrected/sector_stratified_results.json`, and embedded in
`corrected/benchmark_results.json`). Either add the table/figure or soften the sentence. (Separately,
four committed figures are never `\includegraphics`'d by the paper: `multiseed_variance.png`,
`baseline_auroc_bar.png`, `hero_chart_all_models.png`, `tuned_baseline_auroc.png` — housekeeping only.)

---

## VERIFIED — reproduces with no discrepancy

| Item | Source artifact | Result |
|---|---|---|
| Table 1 `tab:auroc` (20 AUROC cells) | `final_benchmark.json` | ✅ all match |
| Table 2 `tab:full_metrics` (32 metric cells) | `final_benchmark.json` | ✅ all match |
| Table 3 `tab:auroc_tuned` (20 cells) | `corrected/benchmark_results.json` | ✅ all match |
| Table `tab:multiseed` (per-seed, mean, std) | `corrected/multiseed_results.json` | ✅ all match |
| Gate `tab:gate_tuned` (AUROCs, Δ, 3/4 pass) | `corrected/benchmark_results.json` | ✅ all match (`passed=true, n_wins=3`) |
| Table `tab:gate_default` (AUROCs + Δ, 4/4) | `final_benchmark.json` gate_* | ✅ all match |
| Table `tab:bootstrap` (means + 95% CIs) | `final_benchmark.json` bootstrap (paired, n=1000) | ✅ all match |
| Table `tab:walkforward` (wins/means/ranges) | `corrected/walk_forward_results.json` | ✅ matches (one 0.001 mis-round, M2) |
| FT Optuna params d=128, L=3, η≈9.89e-4, 30 trials | `corrected/ft_transformer_tuning.json` | ✅ match |
| SSL loss curves: 200 epochs; masks {0.15,0.30,0.50} | `ssl_experiment_results.json` | ✅ match (the AUROC table is the problem, not the loss curves) |
| 33,737 modeling obs; splits 13,920 / 5,657 / 14,160 | parquets (recomputed) | ✅ exact |
| Test label counts 8334/5821 (+5 NaN), 2952/11208, 96/14064, 113/14047 | parquets (recomputed) | ✅ exact |
| Prevalences 0.68% / 0.80% / 20.8% / 58.9% | derived | ✅ consistent |
| Feature counts: 16 raw / 12 ratios / 16 YoY / 45 pre-prune / 83 cont + 1 cat (`sector_idx`) | `build_feature_matrix` (run) | ✅ exact |
| Multi-seed std ≤ 0.013 (×3) and 0.044 (sec) | `multiseed_results.json` | ✅ match |

**Note on 33,737:** the raw inner merge yields 33,751 rows; exactly 14 have `period_end > 2023-12-31`
and are excluded by the test cap, giving 33,737 = the modeling set (≤2023) = sum of the three splits.
Consistent with the paper's "modeling uses 2012–2023" framing.

---

## Provenance pointers (for the fix)
- SSL table values first appear: commit `6fb758d`; never updated since.
- SSL pipeline changed (needs rerun): commit `f4073ad` ("Requires rerun to regenerate corrected numbers").
- `ssl_experiment_results.json` last regenerated: commit `81f84b3` (predates `f4073ad`).
- Committed `final_benchmark.json` `_meta`: `spec=ATS-176`, `ci_method=paired_bootstrap`,
  `n_bootstrap=1000`, `best_ssl_mr=0.50`, `generated_at=2026-03-26T22:39:29` — not produced by the
  current `generate_final_benchmark.py`.
