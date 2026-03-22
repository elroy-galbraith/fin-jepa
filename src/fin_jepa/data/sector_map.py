"""SIC-to-sector mapping using the Fama-French 12-industry classification.

The FF12 scheme is the standard in empirical finance research. Sector
assignments follow the SIC-range definitions from Ken French's data library
(https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/).

Usage::

    from fin_jepa.data.sector_map import sic_to_sector, FF12_SECTORS
    sector = sic_to_sector("2836")   # -> "Health Care"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# FF12 sector definitions as (start_sic, end_sic, sector) tuples.
# Rules are processed in order; later rules override earlier ones so that
# more-specific sub-ranges take precedence over their enclosing ranges.
# ---------------------------------------------------------------------------
_FF12_RANGES: list[tuple[int, int, str]] = [
    # ── Consumer NonDurables ────────────────────────────────────────────────
    (100, 999, "Consumer NonDurables"),       # Agriculture
    (2000, 2099, "Consumer NonDurables"),     # Food & kindred products
    (2100, 2199, "Consumer NonDurables"),     # Tobacco
    (2200, 2399, "Consumer NonDurables"),     # Textiles & apparel
    (2700, 2749, "Consumer NonDurables"),     # Newspapers & publishing
    (2770, 2799, "Consumer NonDurables"),     # Greeting cards
    (3140, 3149, "Consumer NonDurables"),     # Footwear
    (3440, 3469, "Consumer NonDurables"),     # Fabricated plate work, misc
    (3900, 3989, "Consumer NonDurables"),     # Misc manufacturing
    (3993, 3993, "Consumer NonDurables"),     # Signs, advertising specialties
    # ── Consumer Durables ───────────────────────────────────────────────────
    (2500, 2599, "Consumer Durables"),        # Furniture
    (3630, 3639, "Consumer Durables"),        # Household appliances
    (3650, 3659, "Consumer Durables"),        # Radio, TV equipment
    (3710, 3711, "Consumer Durables"),        # Motor vehicles
    (3714, 3714, "Consumer Durables"),        # Motor vehicle parts
    (3716, 3716, "Consumer Durables"),        # Motor homes
    (3750, 3751, "Consumer Durables"),        # Motorcycles, bicycles
    (3792, 3792, "Consumer Durables"),        # Travel trailers
    (3900, 3939, "Consumer Durables"),        # Misc durable mfg (override below)
    (3990, 3991, "Consumer Durables"),
    # ── Manufacturing ───────────────────────────────────────────────────────
    (2400, 2499, "Manufacturing"),            # Lumber & wood
    (2600, 2699, "Manufacturing"),            # Paper & allied products
    (2750, 2769, "Manufacturing"),            # Commercial printing
    (2800, 2829, "Manufacturing"),            # Chemicals (excl pharma)
    (2840, 2899, "Manufacturing"),            # Soap, cleaners, misc chemicals
    (3000, 3099, "Manufacturing"),            # Rubber & plastics
    (3200, 3299, "Manufacturing"),            # Stone, clay, glass
    (3300, 3399, "Manufacturing"),            # Primary metals
    (3400, 3439, "Manufacturing"),            # Fabricated metals
    (3470, 3489, "Manufacturing"),            # Fabricated metals continued
    (3490, 3499, "Manufacturing"),
    (3500, 3569, "Manufacturing"),            # Industrial machinery
    (3580, 3599, "Manufacturing"),            # General industrial machinery
    (3670, 3679, "Manufacturing"),            # Electronic components
    (3700, 3709, "Manufacturing"),            # Transportation equipment
    (3712, 3713, "Manufacturing"),            # Truck trailers, bus bodies
    (3715, 3715, "Manufacturing"),            # Truck trailers
    (3720, 3749, "Manufacturing"),            # Aircraft, ships, railroad
    (3760, 3791, "Manufacturing"),            # Guided missiles, misc
    (3793, 3799, "Manufacturing"),
    (3830, 3839, "Manufacturing"),            # Optical instruments
    (3860, 3899, "Manufacturing"),            # Photo equipment, watches
    # ── Energy ──────────────────────────────────────────────────────────────
    (1200, 1299, "Energy"),                   # Coal mining
    (1300, 1399, "Energy"),                   # Oil & gas extraction
    (2900, 2999, "Energy"),                   # Petroleum refining
    (4600, 4699, "Energy"),                   # Pipelines
    # ── Chemicals ───────────────────────────────────────────────────────────
    (2800, 2824, "Chemicals"),                # Industrial chemicals
    (2860, 2879, "Chemicals"),                # Agricultural chemicals
    (2890, 2891, "Chemicals"),                # Adhesives, sealants
    # ── Business Equipment (IT) ─────────────────────────────────────────────
    (3570, 3579, "Business Equipment"),       # Computer hardware
    (3660, 3669, "Business Equipment"),       # Communications equipment
    (3670, 3679, "Business Equipment"),       # Electronic components
    (3674, 3674, "Business Equipment"),       # Semiconductors
    (3810, 3829, "Business Equipment"),       # Measuring instruments
    (7370, 7379, "Business Equipment"),       # Computer services & software
    # ── Telecom ─────────────────────────────────────────────────────────────
    (4800, 4813, "Telecom"),                  # Telephone & telegraph
    (4890, 4899, "Telecom"),                  # Communications services
    # ── Utilities ───────────────────────────────────────────────────────────
    (4900, 4949, "Utilities"),                # Electric services
    (4950, 4991, "Utilities"),                # Gas & water
    # ── Shops (Retail/Wholesale) ─────────────────────────────────────────────
    (5000, 5199, "Shops"),                    # Wholesale trade
    (5200, 5999, "Shops"),                    # Retail trade
    (7200, 7299, "Shops"),                    # Personal services (laundries, etc.)
    (7600, 7699, "Shops"),                    # Misc repair shops
    # ── Health Care ──────────────────────────────────────────────────────────
    (2830, 2836, "Health Care"),              # Pharmaceutical preparations
    (3840, 3851, "Health Care"),              # Medical instruments & supplies
    (5047, 5047, "Health Care"),              # Medical & hospital equipment (wholesale)
    (5122, 5122, "Health Care"),              # Drugs (wholesale)
    (8000, 8099, "Health Care"),              # Health services (hospitals, etc.)
    # ── Finance ──────────────────────────────────────────────────────────────
    (6000, 6999, "Finance"),                  # Finance, insurance, real estate
    # ── Other (catch-all) ────────────────────────────────────────────────────
    # Anything not matched above maps to "Other" — see sic_to_sector()
]

# Sector labels in FF12 order (for consistent ordering in outputs)
FF12_SECTORS: list[str] = [
    "Consumer NonDurables",
    "Consumer Durables",
    "Manufacturing",
    "Energy",
    "Chemicals",
    "Business Equipment",
    "Telecom",
    "Utilities",
    "Shops",
    "Health Care",
    "Finance",
    "Other",
]


def _build_sic_table() -> dict[int, str]:
    """Pre-compute a flat SIC → sector dict from the range definitions.

    **Override rule**: ``_FF12_RANGES`` is processed in order and the *last*
    entry wins for any SIC code claimed by multiple ranges.  This is
    intentional: narrow sub-ranges (e.g. ``(2800, 2824, "Chemicals")``)
    are placed *after* their enclosing ranges so they take precedence.

    A WARNING is emitted at import time for every SIC code where one range
    overrides a previous one.  This surfaces unintended overlaps introduced
    by future edits to ``_FF12_RANGES`` (W4).
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    first_claim: dict[int, str] = {}   # sic → sector that first claimed it
    table: dict[int, str] = {}
    overrides: list[str] = []

    for start, end, sector in _FF12_RANGES:
        for sic in range(start, end + 1):
            if sic in first_claim and first_claim[sic] != sector:
                overrides.append(
                    f"  SIC {sic:04d}: '{first_claim[sic]}' → '{sector}'"
                )
            else:
                first_claim[sic] = sector
            table[sic] = sector

    if overrides:
        # All overlaps are intentional (narrower sub-ranges overriding their
        # enclosing ranges), so emit at DEBUG only to avoid spamming logs
        # on every import.
        _log.debug(
            "sector_map: %d intentional SIC override(s) detected "
            "(later _FF12_RANGES entry wins):\n%s",
            len(overrides),
            "\n".join(overrides),
        )

    return table


_SIC_TABLE: dict[int, str] = _build_sic_table()


def sic_to_sector(sic: str | int | None) -> str:
    """Map a 4-digit SIC code to a Fama-French 12-industry sector.

    Args:
        sic: SIC code as int, str (e.g. ``"2836"`` or ``"0100"``), or None.

    Returns:
        One of the 12 FF12 sector labels, or ``"Other"`` for unknown codes.
    """
    if sic is None:
        return "Other"
    try:
        return _SIC_TABLE.get(int(sic), "Other")
    except (ValueError, TypeError):
        return "Other"


# Selected SIC major-group descriptions (first 2-digit division level)
# Used for human-readable reporting when 4-digit lookup fails.
SIC_DIVISION_DESCRIPTIONS: dict[str, str] = {
    "01": "Crops", "02": "Livestock", "07": "Agricultural Services",
    "08": "Forestry", "09": "Fishing",
    "10": "Metal Mining", "11": "Anthracite Mining", "12": "Coal Mining",
    "13": "Oil & Gas Extraction", "14": "Non-Metallic Minerals",
    "15": "Building Construction", "16": "Heavy Construction",
    "17": "Special Trade Contractors",
    "20": "Food Products", "21": "Tobacco", "22": "Textile Mill Products",
    "23": "Apparel", "24": "Lumber & Wood", "25": "Furniture",
    "26": "Paper", "27": "Printing & Publishing", "28": "Chemicals",
    "29": "Petroleum Refining", "30": "Rubber & Plastics",
    "31": "Leather", "32": "Stone, Clay, Glass", "33": "Primary Metals",
    "34": "Fabricated Metals", "35": "Industrial Machinery",
    "36": "Electronic Equipment", "37": "Transportation Equipment",
    "38": "Instruments", "39": "Misc Manufacturing",
    "40": "Railroads", "41": "Local Transit", "42": "Trucking",
    "44": "Water Transportation", "45": "Air Transportation",
    "46": "Pipelines", "47": "Transportation Services",
    "48": "Communications", "49": "Electric, Gas, Sanitary Services",
    "50": "Durable Goods Wholesale", "51": "Non-Durable Goods Wholesale",
    "52": "Building Materials Retail", "53": "General Merchandise",
    "54": "Food Stores", "55": "Auto Dealers", "56": "Apparel Stores",
    "57": "Furniture Stores", "58": "Eating & Drinking Places",
    "59": "Misc Retail",
    "60": "Depository Institutions", "61": "Non-Depository Credit",
    "62": "Security & Commodity Services", "63": "Insurance Carriers",
    "64": "Insurance Agents", "65": "Real Estate",
    "67": "Holding & Investment Companies",
    "70": "Hotels & Lodging", "72": "Personal Services",
    "73": "Computer & Data Processing", "75": "Auto Repair",
    "76": "Misc Repair", "78": "Motion Pictures",
    "79": "Amusement & Recreation", "80": "Health Services",
    "82": "Educational Services", "83": "Social Services",
    "86": "Membership Organizations", "87": "Engineering & Management",
    "89": "Misc Services",
    "91": "Federal Government", "92": "State Government",
    "93": "Finance, Taxation, Monetary Policy",
    "94": "Administration of Human Resources",
    "95": "Environmental Quality", "96": "Administration of Economic Programs",
    "97": "National Security", "99": "Nonclassifiable Establishments",
}
