"""One-shot runner for ATS-167 SSL experiment (no Hydra CLI required)."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
Path(ROOT / "results/study0").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "results/study0/ssl_experiment_run.log", mode="w"),
    ],
)

# Load config from YAML via OmegaConf, fall back to plain dict if not available.
cfg_path = ROOT / "configs/study0/ssl_experiment.yaml"
try:
    from omegaconf import OmegaConf
    config = OmegaConf.load(cfg_path)
    print(f"Loaded config from {cfg_path}")
except ImportError:
    import yaml
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    print(f"Loaded config (plain dict) from {cfg_path}")


from fin_jepa.training.pretrain_ssl import run_ssl_experiment
results = run_ssl_experiment(config)

print("\n=== SSL EXPERIMENT RESULTS ===")
print(f"Recommendation: {results['recommendation']}")
print("\nPer-outcome comparison:")
for outcome, comp in results["comparison"].items():
    if comp.get("skipped"):
        print(f"  {outcome}: SKIPPED")
    else:
        print(
            f"  {outcome}: baseline={comp['baseline_auroc']:.4f}  "
            f"best_pretrained={comp['best_pretrained_auroc']:.4f}  "
            f"delta={comp['delta']:+.4f}  best_ratio={comp['best_mask_ratio']}"
        )
print("\nDone. Full results in results/study0/ssl_experiment_results.json")
