"""The Dim_KPI catalog: the 25 KPIs (K01-K25) the workbook reports, and how to format them.

Dim_KPI is fixed reference content. The tool embeds it, it does not calculate it. The
catalog below is Mahir's master file, transcribed. Point `config.KPI_DIM_KPI_FILE` at an
exported .xlsx or .csv to override it wholesale; that path is kept for a future real
export and is unchanged.

`KPI_Name` holds the KPI's display name, not the name of the column it lands in, so the
number format and target are joined to `KPI_Data` / `Perf_Computation` through
`COLUMN_BY_KPI_ID` below rather than by name. An external file that names columns directly
still works - `column_attributes()` falls back to matching `KPI_Name` against the column
names when a KPI_ID is unknown.

Two catalog rows legitimately share a physical column:

- K06 (Limit Margin Error) and K24 (Mean Signed Margin Error) have the same formula and
  the same primitives, listed separately because they belong to two sections. Both carry
  Number_Format "0.0", so the shared column is unambiguous.
- K19 and K20 are K01/K02 and K12 read on the `Case = "BaseCase"` rows. They are not
  separate columns and are resolved by `base_case_readings()` instead.
"""

import logging
import os

import numpy as np
import pandas as pd

from rosc_acdc import config

logger = logging.getLogger(__name__)

SHEET_NAME = "Dim_KPI"

COLUMNS = [
    "KPI_ID", "KPI_Name", "Section", "Priority", "Unit", "Direction",
    "Definition", "Measure_Logic_DAX", "Target", "Number_Format", "Source_Table",
]

_FLOW = "Flow deviation"
_LOADING = "Loading & limits"
_VIOLATION = "Violation detection"
_SEVERITY = "Severity"
_RANKING = "Ranking & critical elements"
_PERF = "Performance"
_BASE = "Base case"
_CO = "Contingency analysis"
_BIAS = "Bias"

_LOWER = "Lower is better"
_HIGHER = "Higher is better"
_CLOSER = "Closer to 0 is better"

_KPI_DATA = "KPI_Data"
_PERF_COMPUTATION = "Perf_Computation"

# Targets are real numbers. The master sheet displays them with a comma decimal separator
# (0,1 / 0,85); they are stored here as floats so the pass/fail comparison can use them.
_CATALOG = [
    ("K01", "Mean Absolute Error (MAE)", _FLOW, 1, "A", _LOWER,
     "Typical absolute Amp deviation between DC and AC flow.",
     "SUM(Sum_AbsErr_A) / SUM(N_Obs)", 60, "0.0", _KPI_DATA),
    ("K02", "Root Mean Square Error (RMSE)", _FLOW, 1, "A", _LOWER,
     "Overall deviation, penalising large mismatches more heavily.",
     "SQRT( SUM(Sum_SqErr_A) / SUM(N_Obs) )", 90, "0.0", _KPI_DATA),
    ("K03", "95th Percentile Error (P95)", _FLOW, 1, "A", _LOWER,
     "Error level reached in the worst 5% of observations.",
     "Weighted average of P95_AbsErr_A (exact P95 needs the element-level table)",
     200, "0.0", _KPI_DATA),
    ("K04", "Maximum Error", _FLOW, 1, "A", _LOWER,
     "Single largest observed DC vs AC mismatch.",
     "MAX(Max_AbsErr_A)", 400, "0.0", _KPI_DATA),

    ("K05", "Loading Percentage Deviation (signed)", _LOADING, 1, "pp", _CLOSER,
     "Signed difference in element loading, DC minus AC.",
     "SUM(Sum_LoadingDev_pp) / SUM(N_Obs)", 0, "0.00", _KPI_DATA),
    ("K06", "Limit Margin Error (signed)", _LOADING, 1, "A", _CLOSER,
     "Signed difference in remaining margin to the thermal limit, DC minus AC.",
     "SUM(Sum_MarginDev_A) / SUM(N_Obs)", 0, "0.0", _KPI_DATA),
    ("K07", "Near-Limit Error", _LOADING, 1, "pp", _CLOSER,
     "Loading deviation restricted to elements with AC loading >= 90%.",
     "SUM(Sum_NearLimit_LoadingDev_pp) / SUM(N_NearLimit_Obs)", 0, "0.00", _KPI_DATA),

    ("K08", "False Negatives (count)", _VIOLATION, 1, "count", _LOWER,
     "AC shows an overload, DC does not.",
     "SUM(N_FN)", 0, "#,##0", _KPI_DATA),
    ("K09", "False Positives (count)", _VIOLATION, 1, "count", _LOWER,
     "DC shows an overload, AC does not.",
     "SUM(N_FP)", 0, "#,##0", _KPI_DATA),
    ("K10", "Missed Overload Rate", _VIOLATION, 1, "%", _LOWER,
     "Share of AC violations that DC fails to detect.",
     "DIVIDE( SUM(N_FN), SUM(N_AC_Viol) )", 0.1, "0.0%", _KPI_DATA),
    ("K11", "False Alarm Rate", _VIOLATION, 1, "%", _LOWER,
     "Share of AC-secure observations DC flags as violations.",
     "DIVIDE( SUM(N_FP), SUM(N_Obs) - SUM(N_AC_Viol) )", 0.05, "0.0%", _KPI_DATA),

    ("K12", "Missed Overload Volume", _SEVERITY, 1, "A", _LOWER,
     "Total Amp severity of AC overloads DC fails to detect.",
     "SUM(Sum_MissedOverload_A)", 0, "#,##0", _KPI_DATA),
    ("K13", "False Overload Volume", _SEVERITY, 1, "A", _LOWER,
     "Total Amp severity of DC overloads that do not exist in AC.",
     "SUM(Sum_FalseOverload_A)", 0, "#,##0", _KPI_DATA),
    ("K14", "False Negatives avg load", _SEVERITY, 1, "%", _LOWER,
     "Average AC loading of the false-negative observations.",
     "DIVIDE( SUM(Sum_FN_Loading_AC_pct), SUM(N_FN) )", 0, "0.0", _KPI_DATA),
    ("K15", "False Positives avg load", _SEVERITY, 1, "%", _LOWER,
     "Average DC loading of the false-positive observations.",
     "DIVIDE( SUM(Sum_FP_Loading_DC_pct), SUM(N_FP) )", 0, "0.0", _KPI_DATA),

    ("K16", "Top-N Critical Element Overlap", _RANKING, 1, "%", _HIGHER,
     "Overlap between the AC and DC top-N most loaded elements (N = 5).",
     "DIVIDE( SUM(TopN_Overlap_Count), SUM(TopN_N) )", 0.9, "0.0%", _KPI_DATA),
    ("K17", "Worst-Case Match Rate", _RANKING, 1, "%", _HIGHER,
     "How often AC and DC identify the same single worst element.",
     "DIVIDE( SUM(WorstCase_Match_Flag), SUM(WorstCase_Total) )", 0.85, "0.0%", _KPI_DATA),

    ("K18", "Computation time", _PERF, 1, "s", _LOWER,
     "Wall-clock time per process and timestamp.",
     "SUM(Perf_Computation[Computation_Time_s])", 0, "#,##0.0", _PERF_COMPUTATION),

    ("K19", "N-0 Flow MAE / RMSE", _BASE, 2, "A", _LOWER,
     "MAE and RMSE on the base-case subset only.",
     'Same measures as K01/K02, filtered to Case = "BaseCase"', 40, "0.0", _KPI_DATA),
    ("K20", "N-0 Missed Overload Volume", _BASE, 2, "A", _LOWER,
     "Base-case AC overload severity missed by DC.",
     'SUM(Sum_MissedOverload_A) filtered to Case = "BaseCase"', 0, "#,##0", _KPI_DATA),

    ("K21", "Missed Critical CO Rate", _CO, 2, "%", _LOWER,
     "Contingencies critical in AC but not flagged in DC.",
     "DIVIDE( SUM(N_CO_MissedCritical), SUM(N_CO_AC_Viol) )", 0.1, "0.0%", _KPI_DATA),
    ("K22", "False Critical CO Rate", _CO, 2, "%", _LOWER,
     "Contingencies critical in DC but not in AC.",
     "DIVIDE( SUM(N_CO_FalseCritical), SUM(N_CO) - SUM(N_CO_AC_Viol) )",
     0.05, "0.0%", _KPI_DATA),
    ("K23", "Worst CO Severity Deviation", _CO, 2, "A", _CLOSER,
     "Difference in maximum overload severity per contingency, DC minus AC.",
     "DIVIDE( SUM(Sum_WorstCO_SevDev_A), SUM(N_CO_SevDev_Obs) )", 0, "0.0", _KPI_DATA),

    ("K24", "Mean Signed (Margin) Error", _BIAS, 2, "A", _CLOSER,
     "Positive = DC systematically overstates available margin (optimistic).",
     "SUM(Sum_MarginDev_A) / SUM(N_Obs)", 0, "0.0", _KPI_DATA),
    ("K25", "Near-Limit Underestimation Rate", _BIAS, 2, "%", _LOWER,
     "How often DC underestimates loading among near-limit elements.",
     "DIVIDE( SUM(N_NearLimit_Underest), SUM(N_NearLimit_Obs) )", 0.5, "0.0%", _KPI_DATA),
]

# Which physical column each catalog row governs. KPI_Name is a display name, so this is
# the join, keyed on the stable KPI_ID.
#
# K19 and K20 are absent deliberately: they are K01/K02 and K12 read on the BaseCase rows,
# not columns of their own, and the columns they read already take their format from
# K01/K02/K12. See base_case_readings().
COLUMN_BY_KPI_ID = {
    "K01": "MAE_A",
    "K02": "RMSE_A",
    "K03": "P95_Err_A",
    "K04": "Max_Err_A",
    "K05": "Mean_LoadingDev_pp",
    "K06": "Mean_Signed_Margin_Err_A",
    "K07": "NearLimit_Mean_LoadingDev_pp",
    "K08": "N_FN",
    "K09": "N_FP",
    "K10": "Missed_Overload_Rate",
    "K11": "False_Alarm_Rate",
    "K12": "Missed_Overload_Volume_A",
    "K13": "False_Overload_Volume_A",
    "K14": "Avg_FN_Loading_AC_pct",
    "K15": "Avg_FP_Loading_DC_pct",
    "K16": "TopN_Overlap_Rate",
    "K17": "WorstCase_Match_Rate",
    "K18": "Computation_Time_s",
    "K21": "Missed_Critical_CO_Rate",
    "K22": "False_Critical_CO_Rate",
    "K23": "Mean_WorstCO_SevDev_A",
    "K24": "Mean_Signed_Margin_Err_A",
    "K25": "NearLimit_Underest_Rate",
}

# Formats for the columns that are not one of the 25 named KPIs - the raw primitives and
# the key columns. These have no Dim_KPI row and therefore no authoritative format.
# UNCONFIRMED: counts as #,##0 and sums as #,##0.0 is this build's suggestion, per
# 04_business_rules_and_kpi_catalog.md's own note that it is a suggestion awaiting Mahir.
PRIMITIVE_COUNT_FORMAT = "#,##0"
PRIMITIVE_SUM_FORMAT = "#,##0.0"


def embedded_catalog() -> pd.DataFrame:
    """The catalog above as a dataframe, in K-number order."""
    return pd.DataFrame(_CATALOG, columns=COLUMNS)


def load_catalog() -> pd.DataFrame:
    """Mahir's exported Dim_KPI file when config points at one, otherwise the embedded copy.

    The external file is passed through unchanged - that is the whole point of it - so a
    column this build does not recognise simply travels into the workbook untouched.
    """
    path = (config.KPI_DIM_KPI_FILE or "").strip()
    if not path:
        return embedded_catalog()

    if not os.path.exists(path):
        logger.warning(
            "Dim_KPI: KPI_DIM_KPI_FILE %s does not exist, embedding the built-in catalog", path,
        )
        return embedded_catalog()

    catalog = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path, sheet_name=0)
    logger.info("Dim_KPI: embedded %d rows from %s", len(catalog), path)
    return catalog


def column_attributes(catalog: pd.DataFrame, columns=None) -> dict:
    """column name -> {"format": str, "direction": str, "target": float} from the catalog.

    Joined on KPI_ID through COLUMN_BY_KPI_ID. A catalog row whose KPI_ID is unknown falls
    back to treating its KPI_Name as a column name, which is what an externally exported
    file keyed on columns would rely on.

    Two rows governing the same column with different formats is a transcription error
    rather than something to resolve silently, so it is logged.
    """
    attributes = {}
    if "KPI_Name" not in catalog.columns:
        logger.warning("Dim_KPI: no KPI_Name column, no KPI formats or targets applied")
        return attributes

    known = set(columns) if columns is not None else None

    for row in catalog.itertuples():
        kpi_id = str(getattr(row, "KPI_ID", "") or "")
        column = COLUMN_BY_KPI_ID.get(kpi_id)
        if column is None:
            candidate = str(row.KPI_Name)
            if known is not None and candidate in known:
                column = candidate
        if column is None or (known is not None and column not in known):
            continue

        number_format = getattr(row, "Number_Format", None)
        target = getattr(row, "Target", None)
        direction = getattr(row, "Direction", None)

        existing = attributes.get(column)
        if existing and number_format and existing["format"] != str(number_format):
            logger.warning(
                "Dim_KPI: %s is governed by two rows with different Number_Format (%s vs %s); "
                "keeping the first", column, existing["format"], number_format,
            )
            continue

        attributes[column] = {
            "format": str(number_format) if number_format is not None
                      and not (isinstance(number_format, float) and np.isnan(number_format))
                      else None,
            "direction": str(direction) if direction is not None
                         and not (isinstance(direction, float) and np.isnan(direction))
                         else None,
            "target": _as_number(target),
        }
    return attributes


def _as_number(value):
    """A Target as a real number, tolerating a comma decimal separator from an export."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def fails_target(value, direction, target) -> bool:
    """Whether one KPI value misses its target, read through the catalog's Direction.

    "Closer to 0 is better" compares magnitudes, so with the catalog's target of 0 any
    non-zero value misses. That is the catalog as transcribed, not a choice made here.
    """
    if value is None or target is None or not direction:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if np.isnan(number):
        return False

    direction = direction.lower()
    if "closer" in direction:
        return abs(number) > abs(target)
    if "higher" in direction:
        return number < target
    if "lower" in direction:
        return number > target
    return False


def base_case_readings(kpi_data: pd.DataFrame, base_case_label: str = "BaseCase") -> dict:
    """K19 and K20: K01/K02 and K12 re-derived over the `Case = "BaseCase"` rows.

    Neither is a column of its own. Both are pooled from the primitives those rows already
    carry, rather than averaged from the pre-computed MAE_A / RMSE_A columns, which is what
    the workbook's Rule 2 requires of any aggregation beyond a single row.
    """
    empty = {"K19_MAE_A": np.nan, "K19_RMSE_A": np.nan, "K20_Missed_Overload_Volume_A": np.nan}
    if kpi_data.empty or "Case" not in kpi_data.columns:
        return empty

    base = kpi_data[kpi_data["Case"] == base_case_label]
    if base.empty:
        logger.warning(
            "Dim_KPI: no rows with Case == %r, so K19/K20 have nothing to read", base_case_label,
        )
        return empty

    observations = base["N_Obs"].sum()
    if not observations:
        return empty

    return {
        "K19_MAE_A": base["Sum_AbsErr_A"].sum() / observations,
        "K19_RMSE_A": (base["Sum_SqErr_A"].sum() / observations) ** 0.5,
        "K20_Missed_Overload_Volume_A": base["Sum_MissedOverload_A"].sum(),
    }
