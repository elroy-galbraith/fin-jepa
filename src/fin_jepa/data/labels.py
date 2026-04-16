"""Distress event label database.

Constructs binary distress labels per (cik, period_end) for five adverse
outcome types used by Study 0 (supervised distress prediction) and
Studies 1–2 (coherence signal evaluation).

Outcome types
-------------
1. stock_decline      — >20 % market-adjusted price decline within 12 months
2. earnings_restate   — earnings restatement (10-K/A amendment proxy)
3. audit_qualification — going-concern or adverse audit opinion
4. sec_enforcement    — SEC enforcement action (AAERs, litigation releases)
5. bankruptcy         — Chapter 7 or Chapter 11 filing

Data sources
------------
* **stock_decline** is computed from ``market_aligned.parquet`` (ATS-163) by
  thresholding the 252-day market-adjusted forward log return.
* **earnings_restate** uses the EDGAR filing index (ATS-161) as a proxy:
  the presence of a 10-K/A amendment filed within ``horizon_days`` of the
  original period end signals a potential restatement.  This is a *noisy*
  proxy — not all 10-K/A filings are restatements.  For ground-truth data,
  use Audit Analytics and set ``restatement_source="external_csv"``.
* **audit_qualification** requires external data (Audit Analytics or
  equivalent).  Loaded from CSV when available; all-NaN otherwise.
* **sec_enforcement** uses the Dechow et al. AAER database (freely
  available for academic use) loaded from CSV.  All-NaN if not provided.
* **bankruptcy** uses Compustat delist codes (dlrsn="03") when available
  via :func:`~fin_jepa.data.compustat.merge_compustat`, with an external
  CSV fallback.  All-NaN if neither source is available.

Missing-data semantics
----------------------
NaN = "data unavailable", **not** "no event".  Downstream training must
mask NaN labels in the loss function.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome type enum
# ---------------------------------------------------------------------------

class OutcomeType(StrEnum):
    STOCK_DECLINE = "stock_decline"
    EARNINGS_RESTATE = "earnings_restate"
    AUDIT_QUALIFICATION = "audit_qualification"
    SEC_ENFORCEMENT = "sec_enforcement"
    BANKRUPTCY = "bankruptcy"


ALL_OUTCOMES: list[str] = list(OutcomeType)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LabelConfig:
    """Configuration for distress label construction."""

    #: Market-adjusted 252-day log return threshold for stock_decline.
    #: Default −0.20 corresponds to a >20 % decline (market-adjusted).
    decline_threshold: float = -0.20

    #: Treat delisted companies (insufficient post-filing price data) as
    #: stock_decline = 1.  Delisting often accompanies distress events.
    treat_delisted_as_decline: bool = True

    #: Source for earnings restatement labels.
    #: ``"reconciled"`` *(default)* — OR together the widened EDGAR 10-K/A
    #:   match and the XBRL amendment registry (any ``(cik, period_end)`` with
    #:   ≥2 distinct 10-K filings).  Writes an ``earnings_restate_source``
    #:   provenance column.
    #: ``"edgar_amendments"`` — use 10-K/A filings from the EDGAR index only.
    #:   Strict form-type match, 365-day forward window.  Kept for tests and
    #:   backwards-compatible rebuilds.
    #: ``"external_csv"`` — load from ``external_label_dir/earnings_restate.csv``.
    restatement_source: str = "reconciled"

    #: Forward-looking window (calendar days) used specifically for matching
    #: 10-K/A amendments in the ``"reconciled"`` path.  Restatements are often
    #: filed 1–3 years after the original period_end, so the default here is
    #: wider than the general ``horizon_days``.
    restatement_horizon_days: int = 1095

    #: Form-type strings that count as restatement amendments under the
    #: ``"reconciled"`` path.  ``"NT 10-K/A"`` = notification of late
    #: amendment; ``"10-KT/A"`` = transition-period amendment.
    restatement_form_types: tuple[str, ...] = (
        "10-K/A", "10-KT/A", "NT 10-K/A",
    )

    #: Source for audit qualification labels.
    #: Only ``"external_csv"`` is supported (no free structured source).
    audit_qualification_source: str = "external_csv"

    #: Source for SEC enforcement labels.
    #: ``"external_csv"`` — load from ``external_label_dir/sec_enforcement.csv``.
    sec_enforcement_source: str = "external_csv"

    #: Source for bankruptcy labels.
    #: ``"compustat"`` — derive from Compustat delist codes in the universe.
    #: ``"external_csv"`` — load from ``external_label_dir/bankruptcy.csv``.
    bankruptcy_source: str = "compustat"

    #: Directory containing external label CSV/Parquet files.
    external_label_dir: Path | None = None

    #: Forward-looking horizon in calendar days for label assignment.
    horizon_days: int = 365


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_label(grid: pd.DataFrame, fill=pd.NA) -> pd.Series:
    """Create a nullable Int8 Series filled with *fill*, aligned to *grid*."""
    return pd.Series(
        pd.array([fill] * len(grid), dtype="Int8"), index=grid.index
    )


def _match_events_to_grid(
    grid: pd.DataFrame,
    events: pd.DataFrame,
    date_col: str,
    horizon_days: int,
) -> pd.Series:
    """Vectorized event-matching: for each grid row, check if any event
    with the same CIK falls within [period_end, period_end + horizon].

    Args:
        grid: Must have ``cik`` (str, zero-padded) and ``period_end`` columns.
        events: Must have ``cik`` (str, zero-padded) and *date_col* columns.
        date_col: Name of the date column in *events* to match against.
        horizon_days: Forward-looking window in calendar days.

    Returns:
        Int8 Series aligned to *grid* index (1 = event in window, 0 = no event).
    """
    label = _empty_label(grid, fill=0)

    if events.empty or date_col not in events.columns:
        return label

    grid_work = grid[["cik", "period_end"]].copy()
    grid_work["period_end"] = pd.to_datetime(grid_work["period_end"])
    grid_work["_grid_idx"] = grid.index

    events_work = events[["cik", date_col]].copy()
    events_work[date_col] = pd.to_datetime(events_work[date_col])

    merged = grid_work.merge(events_work, on="cik", how="inner")
    if merged.empty:
        return label

    horizon = pd.Timedelta(days=horizon_days)
    in_window = merged[date_col].between(
        merged["period_end"], merged["period_end"] + horizon
    )
    hit_indices = merged.loc[in_window, "_grid_idx"].unique()
    label.loc[hit_indices] = 1

    return label


# ---------------------------------------------------------------------------
# Generic external label loader
# ---------------------------------------------------------------------------

def _load_external_label(
    label_name: str,
    external_dir: Path | None,
) -> pd.DataFrame | None:
    """Load an external label file if it exists.

    Looks for ``{label_name}.csv``, ``{label_name}.csv.gz``, or
    ``{label_name}.parquet`` inside *external_dir*.

    Returns:
        DataFrame with at least ``cik`` and ``period_end`` columns,
        or ``None`` if no file is found.
    """
    if external_dir is None:
        return None

    external_dir = Path(external_dir)
    for suffix in (".parquet", ".csv", ".csv.gz"):
        candidate = external_dir / f"{label_name}{suffix}"
        if candidate.exists():
            if suffix == ".parquet":
                df = pd.read_parquet(candidate)
            else:
                df = pd.read_csv(candidate, dtype=str, low_memory=False)
            df.columns = [c.lower().strip() for c in df.columns]
            if "cik" in df.columns:
                df["cik"] = df["cik"].astype(str).str.strip().str.zfill(10)
            if "period_end" in df.columns:
                df["period_end"] = pd.to_datetime(
                    df["period_end"], errors="coerce"
                ).dt.date
            logger.info(
                "Loaded external label '%s' from %s (%d rows)",
                label_name, candidate, len(df),
            )
            return df

    return None


# ---------------------------------------------------------------------------
# Per-label builders
# ---------------------------------------------------------------------------

def _compute_stock_decline(
    market_df: pd.DataFrame,
    config: LabelConfig,
) -> pd.Series:
    """Compute stock_decline labels from market-adjusted forward returns.

    Args:
        market_df: ``market_aligned.parquet`` DataFrame with columns
            ``mkt_adj_252d`` (float) and ``delisted`` (bool).
        config: Label configuration.

    Returns:
        Nullable Int8 Series aligned to *market_df* index.
    """
    label = _empty_label(market_df)

    has_return = market_df["mkt_adj_252d"].notna()
    label[has_return] = (
        market_df.loc[has_return, "mkt_adj_252d"] < config.decline_threshold
    ).astype("Int8")

    if config.treat_delisted_as_decline and "delisted" in market_df.columns:
        delisted_mask = market_df["delisted"].fillna(False).astype(bool)
        label[delisted_mask] = pd.array([1] * delisted_mask.sum(), dtype="Int8")

    return label


def _build_earnings_restate(
    grid: pd.DataFrame,
    raw_dir: Path,
    config: LabelConfig,
    *,
    return_source: bool = False,
) -> pd.Series | tuple[pd.Series, pd.Series | None]:
    """Build earnings_restate labels.

    Sources (selected via ``config.restatement_source``):
      * ``"reconciled"`` *(default)* — OR of widened EDGAR 10-K/A match and
        XBRL amendment registry.  See :func:`_build_restate_reconciled`.
      * ``"edgar_amendments"`` — strict 10-K/A proxy with the narrow
        365-day window.  Kept for reproducibility of pre-v1.1 label DBs.
      * ``"external_csv"`` — load from
        ``{external_label_dir}/earnings_restate.csv``.

    Args:
        grid: Observation grid with ``cik`` and ``period_end`` columns.
        raw_dir: Root raw-data directory.
        config: Label configuration.
        return_source: If True, also return a source-provenance Series
            with values in ``{"edgar", "xbrl", "both", "none", None}``.
            ``None`` means no data source was available for that row.

    Returns:
        A nullable ``Int8`` Series aligned to *grid*.  If ``return_source``
        is True, a ``(label, source)`` tuple where ``source`` is an object
        Series or ``None`` if provenance is not tracked for this source.
    """
    if config.restatement_source == "external_csv":
        ext = _load_external_label("earnings_restate", config.external_label_dir)
        if ext is not None and "earnings_restate" in ext.columns:
            label = _merge_external_label(grid, ext, "earnings_restate")
        else:
            logger.warning(
                "earnings_restate: external CSV requested but not found in %s",
                config.external_label_dir,
            )
            label = _empty_label(grid)
        return (label, None) if return_source else label

    if config.restatement_source == "reconciled":
        return _build_restate_reconciled(
            grid, raw_dir, config, return_source=return_source
        )

    # Legacy "edgar_amendments": strict form-type match, narrow horizon.
    label = _build_restate_from_edgar_index(
        grid,
        raw_dir,
        horizon_days=config.horizon_days,
        form_types=("10-K/A",),
    )
    if label is None:
        label = _empty_label(grid)
    return (label, None) if return_source else label


def _build_restate_from_edgar_index(
    grid: pd.DataFrame,
    raw_dir: Path,
    *,
    horizon_days: int,
    form_types: tuple[str, ...],
) -> pd.Series | None:
    """Match grid rows against 10-K/A-family filings in the EDGAR index.

    Returns ``None`` if the index cache is unavailable (so the caller can
    distinguish "data missing" from "no amendments found"), otherwise a
    0/1 Int8 Series.
    """
    index_dir = raw_dir / "edgar_index"
    if not index_dir.exists():
        logger.warning(
            "earnings_restate: EDGAR index cache not found at %s.",
            index_dir,
        )
        return None

    idx_files = sorted(index_dir.glob("*.parquet"))
    if not idx_files:
        logger.warning(
            "earnings_restate: no parquet files in %s.",
            index_dir,
        )
        return None

    filings = pd.concat(
        [pd.read_parquet(f) for f in idx_files], ignore_index=True
    )

    form_type_stripped = filings["form_type"].astype(str).str.strip()
    amendments = filings.loc[form_type_stripped.isin(form_types)].copy()
    if amendments.empty:
        logger.info(
            "earnings_restate: no %s filings in index; edgar signal = 0.",
            list(form_types),
        )
        return _empty_label(grid, fill=0)

    amendments["cik"] = amendments["cik"].astype(str).str.zfill(10)
    amendments["date_filed"] = pd.to_datetime(
        amendments["date_filed"], errors="coerce"
    )

    return _match_events_to_grid(grid, amendments, "date_filed", horizon_days)


def _build_restate_from_xbrl_registry(
    grid: pd.DataFrame,
    raw_dir: Path,
) -> pd.Series | None:
    """Flag grid rows where the XBRL amendment registry shows >=2 filings.

    Looks for ``xbrl_amendment_registry.parquet`` in ``raw_dir`` and in
    the sibling ``processed/`` directory.  Returns ``None`` if no registry
    exists.
    """
    candidates = [
        raw_dir / "xbrl_amendment_registry.parquet",
        raw_dir.parent / "processed" / "xbrl_amendment_registry.parquet",
    ]
    registry_path = next((p for p in candidates if p.exists()), None)
    if registry_path is None:
        logger.warning(
            "earnings_restate: XBRL amendment registry not found at %s",
            [str(p) for p in candidates],
        )
        return None

    registry = pd.read_parquet(registry_path)
    label = _empty_label(grid, fill=0)
    if registry.empty:
        return label

    registry = registry.copy()
    registry["cik"] = registry["cik"].astype(str).str.zfill(10)
    registry["period_end"] = pd.to_datetime(
        registry["period_end"], errors="coerce"
    ).dt.date

    counts = (
        registry.groupby(["cik", "period_end"]).size().reset_index(name="n_filings")
    )
    amended = counts.loc[counts["n_filings"] >= 2, ["cik", "period_end"]]
    if amended.empty:
        return label

    grid_work = grid[["cik", "period_end"]].copy()
    grid_work["period_end"] = pd.to_datetime(
        grid_work["period_end"], errors="coerce"
    ).dt.date
    grid_work["_grid_idx"] = grid.index

    amended = amended.assign(_amended=1)
    merged = grid_work.merge(amended, on=["cik", "period_end"], how="left")
    hit_idx = merged.loc[merged["_amended"] == 1, "_grid_idx"].values
    label.loc[hit_idx] = 1
    return label


def _build_restate_reconciled(
    grid: pd.DataFrame,
    raw_dir: Path,
    config: LabelConfig,
    *,
    return_source: bool,
) -> pd.Series | tuple[pd.Series, pd.Series | None]:
    """OR the widened-EDGAR and XBRL-registry signals."""
    edgar = _build_restate_from_edgar_index(
        grid,
        raw_dir,
        horizon_days=config.restatement_horizon_days,
        form_types=config.restatement_form_types,
    )
    xbrl = _build_restate_from_xbrl_registry(grid, raw_dir)

    source = pd.Series([None] * len(grid), index=grid.index, dtype="object")

    if edgar is None and xbrl is None:
        logger.warning(
            "earnings_restate (reconciled): neither EDGAR index nor XBRL "
            "amendment registry is available; returning all NaN.",
        )
        label = _empty_label(grid)
        return (label, source) if return_source else label

    zeros = pd.Series([0] * len(grid), index=grid.index, dtype="Int8")
    e_fired = (edgar if edgar is not None else zeros) == 1
    x_fired = (xbrl if xbrl is not None else zeros) == 1

    label = _empty_label(grid, fill=0)
    label[e_fired | x_fired] = 1

    source.loc[e_fired & x_fired] = "both"
    source.loc[e_fired & ~x_fired] = "edgar"
    source.loc[~e_fired & x_fired] = "xbrl"
    source.loc[~e_fired & ~x_fired] = "none"

    return (label, source) if return_source else label


def _build_audit_qualification(
    grid: pd.DataFrame,
    config: LabelConfig,
) -> pd.Series:
    """Build audit_qualification labels from external CSV.

    No free structured data source exists for going-concern opinions.
    Returns all NaN when no external file is provided.

    To supply this data, place a CSV or Parquet file at
    ``{external_label_dir}/audit_qualification.{csv,parquet}``
    with columns: ``cik``, ``period_end``, ``audit_qualification`` (0/1).
    """
    ext = _load_external_label("audit_qualification", config.external_label_dir)
    if ext is not None and "audit_qualification" in ext.columns:
        return _merge_external_label(grid, ext, "audit_qualification")

    logger.warning(
        "audit_qualification: no external data found.  All labels will be NaN.  "
        "Provide data from Audit Analytics or equivalent at "
        "%s/audit_qualification.csv",
        config.external_label_dir or "<external_label_dir>",
    )
    return _empty_label(grid)


def _build_sec_enforcement(
    grid: pd.DataFrame,
    config: LabelConfig,
) -> pd.Series:
    """Build sec_enforcement labels from the AAER database.

    Expects a CSV at ``{external_label_dir}/sec_enforcement.csv`` with
    columns: ``cik``, ``aaer_date`` (or ``start_date`` and ``end_date``
    for the violation period).

    Returns all NaN when no file is provided.  The Dechow et al. AAER
    database is freely available for academic use.
    """
    ext = _load_external_label("sec_enforcement", config.external_label_dir)
    if ext is None:
        logger.warning(
            "sec_enforcement: no external data found.  All labels will be NaN.  "
            "The Dechow et al. AAER database is freely available for academic "
            "use — place it at %s/sec_enforcement.csv",
            config.external_label_dir or "<external_label_dir>",
        )
        return _empty_label(grid)

    # Normalise date columns
    for col in ("aaer_date", "start_date", "end_date"):
        if col in ext.columns:
            ext[col] = pd.to_datetime(ext[col], errors="coerce")

    # If the file already has a binary sec_enforcement column, merge directly
    if "sec_enforcement" in ext.columns:
        return _merge_external_label(grid, ext, "sec_enforcement")

    if "cik" not in ext.columns:
        logger.warning(
            "sec_enforcement: CSV missing 'cik' column; cannot match.  "
            "Returning all NaN."
        )
        return _empty_label(grid)

    # Match on (cik, event window) — uses start_date/end_date overlap or
    # aaer_date within horizon.
    if "start_date" in ext.columns and "end_date" in ext.columns:
        return _match_sec_enforcement_overlap(grid, ext, config)
    elif "aaer_date" in ext.columns:
        return _match_events_to_grid(grid, ext, "aaer_date", config.horizon_days)

    return _empty_label(grid, fill=0)


def _match_sec_enforcement_overlap(
    grid: pd.DataFrame,
    ext: pd.DataFrame,
    config: LabelConfig,
) -> pd.Series:
    """Match SEC enforcement using violation-period overlap logic.

    An event matches if its ``[start_date, end_date]`` overlaps with the
    observation window ``[period_end, period_end + horizon]``.
    """
    label = _empty_label(grid, fill=0)

    grid_work = grid[["cik", "period_end"]].copy()
    grid_work["period_end"] = pd.to_datetime(grid_work["period_end"])
    grid_work["_grid_idx"] = grid.index

    events = ext[["cik", "start_date", "end_date"]].copy()

    merged = grid_work.merge(events, on="cik", how="inner")
    if merged.empty:
        return label

    horizon = pd.Timedelta(days=config.horizon_days)
    overlap = (
        (merged["start_date"] <= merged["period_end"] + horizon)
        & (merged["end_date"] >= merged["period_end"])
    )
    hit_indices = merged.loc[overlap, "_grid_idx"].unique()
    label.loc[hit_indices] = 1

    return label


def _build_bankruptcy(
    grid: pd.DataFrame,
    raw_dir: Path,
    config: LabelConfig,
) -> pd.Series:
    """Build bankruptcy labels from Compustat delist codes or external CSV.

    Primary source (``bankruptcy_source="compustat"``): uses ``cstat_dlrsn``
    and ``cstat_dldte`` columns from the universe enriched via
    :func:`~fin_jepa.data.compustat.merge_compustat`.

    Fallback: ``{external_label_dir}/bankruptcy.csv`` with columns
    ``cik``, ``filing_date``, ``chapter`` (7 or 11).
    """
    if config.bankruptcy_source == "external_csv":
        ext = _load_external_label("bankruptcy", config.external_label_dir)
        if ext is not None and "bankruptcy" in ext.columns:
            return _merge_external_label(grid, ext, "bankruptcy")
        if ext is not None and "filing_date" in ext.columns:
            return _match_events_to_grid(grid, ext, "filing_date", config.horizon_days)
        logger.warning(
            "bankruptcy: external CSV requested but not found or missing "
            "required columns in %s",
            config.external_label_dir,
        )
        return _empty_label(grid)

    # Compustat delist path: look for universe with cstat_ columns
    universe_path = raw_dir / "company_universe.parquet"
    if not universe_path.exists():
        # Try processed dir
        universe_path = raw_dir.parent / "processed" / "company_universe.parquet"

    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
        if "cstat_dlrsn" in universe.columns and "cstat_dldte" in universe.columns:
            return _match_compustat_bankruptcy(grid, universe, config)
        logger.info(
            "bankruptcy: universe at %s lacks Compustat delist columns.  "
            "Trying external CSV fallback.",
            universe_path,
        )

    # Fallback to external CSV
    ext = _load_external_label("bankruptcy", config.external_label_dir)
    if ext is not None and "bankruptcy" in ext.columns:
        return _merge_external_label(grid, ext, "bankruptcy")
    if ext is not None and "filing_date" in ext.columns:
        return _match_events_to_grid(grid, ext, "filing_date", config.horizon_days)

    logger.warning(
        "bankruptcy: no data source available.  All labels will be NaN.  "
        "Enrich the universe with Compustat (merge_compustat) or provide "
        "data at %s/bankruptcy.csv",
        config.external_label_dir or "<external_label_dir>",
    )
    return _empty_label(grid)


def _match_compustat_bankruptcy(
    grid: pd.DataFrame,
    universe: pd.DataFrame,
    config: LabelConfig,
) -> pd.Series:
    """Match Compustat delist reason '03' (bankruptcy) to the grid."""
    bankrupt = universe[
        universe["cstat_dlrsn"].astype(str).str.strip() == "03"
    ].copy()
    if bankrupt.empty:
        return _empty_label(grid, fill=0)

    bankrupt["cik"] = bankrupt["cik"].astype(str).str.zfill(10)
    bankrupt["cstat_dldte"] = pd.to_datetime(
        bankrupt["cstat_dldte"], errors="coerce"
    )

    return _match_events_to_grid(
        grid, bankrupt, "cstat_dldte", config.horizon_days
    )


# ---------------------------------------------------------------------------
# External label merge helper
# ---------------------------------------------------------------------------

def _merge_external_label(
    grid: pd.DataFrame,
    ext: pd.DataFrame,
    label_col: str,
) -> pd.Series:
    """Left-join an external label DataFrame onto the observation grid.

    The external DataFrame must have ``cik``, ``period_end``, and
    *label_col* columns.  Unmatched rows get NaN (data unavailable).
    """
    ext = ext.copy()
    ext["cik"] = ext["cik"].astype(str).str.zfill(10)
    ext["period_end"] = pd.to_datetime(ext["period_end"], errors="coerce").dt.date

    grid_merge = grid[["cik", "period_end"]].copy()
    grid_merge["period_end"] = pd.to_datetime(
        grid_merge["period_end"], errors="coerce"
    ).dt.date

    merged = grid_merge.merge(
        ext[["cik", "period_end", label_col]].drop_duplicates(
            subset=["cik", "period_end"], keep="first"
        ),
        on=["cik", "period_end"],
        how="left",
    )
    values = pd.to_numeric(merged[label_col], errors="coerce")
    return pd.Series(pd.array(values.values, dtype="Int8"), index=grid.index)


# ---------------------------------------------------------------------------
# Validation / coverage reporting
# ---------------------------------------------------------------------------

def validate_label_database(df: pd.DataFrame) -> dict:
    """Compute coverage statistics for the label database.

    Args:
        df: Label database DataFrame with ``ALL_OUTCOMES`` columns.

    Returns:
        Dict with per-label stats and overall summary.
    """
    stats: dict = {"per_label": {}, "n_observations": len(df)}

    for outcome in ALL_OUTCOMES:
        if outcome not in df.columns:
            stats["per_label"][outcome] = {
                "n_positive": 0,
                "n_negative": 0,
                "n_missing": len(df),
                "coverage_pct": 0.0,
                "positive_rate": None,
            }
            continue

        col = df[outcome]
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        n_miss = int(col.isna().sum())
        n_known = n_pos + n_neg
        coverage = 100.0 * n_known / max(len(df), 1)
        pos_rate = n_pos / max(n_known, 1) if n_known > 0 else None

        stats["per_label"][outcome] = {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_missing": n_miss,
            "coverage_pct": round(coverage, 2),
            "positive_rate": round(pos_rate, 4) if pos_rate is not None else None,
        }

        logger.info(
            "  %-25s  pos=%5d  neg=%5d  NA=%5d  coverage=%.1f%%  rate=%s",
            outcome,
            n_pos,
            n_neg,
            n_miss,
            coverage,
            f"{pos_rate:.4f}" if pos_rate is not None else "N/A",
        )

    # Overall stats
    covered_labels = [
        name
        for name, s in stats["per_label"].items()
        if s["coverage_pct"] > 50.0
    ]
    stats["n_labels_with_majority_coverage"] = len(covered_labels)
    stats["labels_with_majority_coverage"] = covered_labels

    all_present = df[ALL_OUTCOMES].notna().all(axis=1).sum()
    stats["n_rows_all_labels_present"] = int(all_present)

    if len(covered_labels) < 2:
        logger.warning(
            "Only %d / %d labels have >50%% coverage.  The go/no-go gate "
            "requires at least 3 evaluable outcomes.",
            len(covered_labels),
            len(ALL_OUTCOMES),
        )

    return stats


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_label_database(
    raw_dir: Path,
    output_path: Path | None = None,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Construct the full distress label table.

    Loads the observation grid from ``market_aligned.parquet``, computes
    or loads each of the five distress labels, merges them, writes the
    result to *output_path* with provenance metadata, and returns the
    DataFrame.

    Args:
        raw_dir: Root raw-data directory (e.g. ``data/raw``).  Must
            contain ``market/market_aligned.parquet`` and optionally
            ``edgar_index/*.parquet`` for the 10-K/A restatement proxy.
        output_path: Where to write the label database parquet.  Defaults
            to ``{raw_dir}/../processed/label_database.parquet``.
        config: Label configuration.  Uses defaults if not provided.

    Returns:
        DataFrame with columns ``[cik, period_end, stock_decline,
        earnings_restate, audit_qualification, sec_enforcement,
        bankruptcy]``.  Label columns are nullable ``Int8`` (0/1/NaN).
    """
    raw_dir = Path(raw_dir)
    if config is None:
        config = LabelConfig()

    if output_path is None:
        output_path = raw_dir.parent / "processed" / "label_database.parquet"

    logger.info("Building distress label database …")

    # 1. Load observation grid from market-aligned data
    market_path = raw_dir / "market" / "market_aligned.parquet"
    if not market_path.exists():
        raise FileNotFoundError(
            f"market_aligned.parquet not found at {market_path}.  "
            "Run build_market_dataset() first (ATS-163)."
        )

    market_df = pd.read_parquet(market_path)
    grid = market_df[["cik", "period_end"]].copy()

    # Normalize CIK once for all downstream builders
    grid["cik"] = grid["cik"].astype(str).str.strip().str.zfill(10)
    logger.info("  Observation grid: %d rows", len(grid))

    # 2. Compute / load each label
    logger.info("  Computing stock_decline (threshold=%.2f) …", config.decline_threshold)
    grid["stock_decline"] = _compute_stock_decline(market_df, config)

    logger.info("  Building earnings_restate (source=%s) …", config.restatement_source)
    restate_label, restate_source = _build_earnings_restate(
        grid, raw_dir, config, return_source=True
    )
    grid["earnings_restate"] = restate_label
    if restate_source is not None:
        grid["earnings_restate_source"] = restate_source

    logger.info(
        "  Building audit_qualification (source=%s) …",
        config.audit_qualification_source,
    )
    grid["audit_qualification"] = _build_audit_qualification(grid, config)

    logger.info(
        "  Building sec_enforcement (source=%s) …",
        config.sec_enforcement_source,
    )
    grid["sec_enforcement"] = _build_sec_enforcement(grid, config)

    logger.info("  Building bankruptcy (source=%s) …", config.bankruptcy_source)
    grid["bankruptcy"] = _build_bankruptcy(grid, raw_dir, config)

    # Ensure label columns are nullable Int8
    for col in ALL_OUTCOMES:
        grid[col] = grid[col].astype("Int8")

    # 3. Validate and log coverage
    logger.info("Label coverage:")
    report = validate_label_database(grid)

    # 4. Write with provenance metadata
    build_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    provenance = {
        "build_date": build_date,
        "n_observations": len(grid),
        "n_companies": int(grid["cik"].nunique()),
        "decline_threshold": config.decline_threshold,
        "treat_delisted_as_decline": config.treat_delisted_as_decline,
        "restatement_source": config.restatement_source,
        "restatement_horizon_days": config.restatement_horizon_days,
        "restatement_form_types": list(config.restatement_form_types),
        "bankruptcy_source": config.bankruptcy_source,
        "horizon_days": config.horizon_days,
        "label_coverage": report["per_label"],
        "data_sources": {
            "stock_decline": "market_aligned.parquet (ATS-163)",
            "earnings_restate": config.restatement_source,
            "audit_qualification": config.audit_qualification_source,
            "sec_enforcement": config.sec_enforcement_source,
            "bankruptcy": config.bankruptcy_source,
        },
    }

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(grid)
        meta = table.schema.metadata or {}
        meta[b"fin_jepa_provenance"] = json.dumps(provenance).encode()
        table = table.replace_schema_metadata(meta)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)
    except ImportError:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(output_path, index=False)
        prov_path = output_path.with_suffix(".provenance.json")
        with open(prov_path, "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2)

    logger.info(
        "Label database written to %s  (%d observations, %d companies)",
        output_path, len(grid), grid["cik"].nunique(),
    )
    return grid


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_label_database(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the label database Parquet file with provenance metadata.

    Args:
        path: Path to ``label_database.parquet``.

    Returns:
        Tuple of ``(labels_df, provenance_dict)``.
    """
    path = Path(path)
    provenance: dict = {}

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        raw_meta = pf.schema_arrow.metadata or {}
        if b"fin_jepa_provenance" in raw_meta:
            provenance = json.loads(raw_meta[b"fin_jepa_provenance"].decode())
        df = pf.read().to_pandas()
    except Exception:
        df = pd.read_parquet(path)
        prov_path = path.with_suffix(".provenance.json")
        if prov_path.exists():
            with open(prov_path, encoding="utf-8") as fh:
                provenance = json.load(fh)

    return df, provenance
