"""Implements: Flow deviation, Loading & limit-based, Violation detection,
Severity-based, Ranking & critical element, and Performance KPIs.
"""

import logging

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from rosc_acdc.paths import output_path

logger = logging.getLogger(__name__)

VIOLATION_THRESHOLD_PCT = 100
NEAR_LIMIT_THRESHOLDS_PCT = (80, 90, 95)
TOP_N_DEFAULT = 10
BASE_CASE_GROUP_LABEL = "N-0 (base case)"

# How many contingencies of each per-CO KPI the log summary shows. This governs the size of
# the log only: every contingency is computed, and all of them reach the workbook below.
PER_CO_LOG_SUMMARY_N = 5

KPI_WORKBOOK_FILENAME = "kpi_results.xlsx"
KPI_WORKBOOK_SHEET = "KPIs"
OVERALL_LABEL = "Overall"


def prepare_comparison(df, ac_value_col, dc_value_col, limit_col,
                        ac_loading_col=None, dc_loading_col=None):
    """Normalize an AC/DC comparison dataframe to the common KPI column names.

    Flows, limits and margins are currents in A; loadings and their deviations are in %.

    Observations without both an AC and a DC value (and a limit) are dropped here so
    every KPI below runs on the same population: a missing DC flow is missing data, not
    "DC sees no violation" - counting it as one inflates the false-negative KPIs.

    Note: both study datasets are already filtered upstream by loading
    (config.BASE_CASE_ACTIVE_THRESHOLD_PCT / config.SA_ACTIVE_THRESHOLD_PCT), so the KPI
    population is the monitored elements above that loading, not all monitored elements.
    """
    out = pd.DataFrame(index=df.index)
    out["ac_value"] = df[ac_value_col]
    out["dc_value"] = df[dc_value_col]
    out["limit"] = df[limit_col]
    out["ac_loading_pct"] = df[ac_loading_col] if ac_loading_col else out["ac_value"].abs() / out["limit"] * 100
    out["dc_loading_pct"] = df[dc_loading_col] if dc_loading_col else out["dc_value"].abs() / out["limit"] * 100

    paired = out[["ac_value", "dc_value", "limit", "ac_loading_pct", "dc_loading_pct"]].notna().all(axis=1)
    if not paired.all():
        logger.warning(
            "Excluding %d of %d observations with no paired AC/DC value or no limit",
            int((~paired).sum()), len(out),
        )
        out = out[paired]

    out["abs_error"] = (out["dc_value"] - out["ac_value"]).abs()
    out["signed_loading_deviation"] = out["dc_loading_pct"] - out["ac_loading_pct"]
    out["margin_ac"] = out["limit"] - out["ac_value"].abs()
    out["margin_dc"] = out["limit"] - out["dc_value"].abs()
    out["signed_margin_error"] = out["margin_dc"] - out["margin_ac"]
    out["ac_violation"] = out["ac_loading_pct"] > VIOLATION_THRESHOLD_PCT
    out["dc_violation"] = out["dc_loading_pct"] > VIOLATION_THRESHOLD_PCT
    return out


def _contingency_groups(comparison: pd.DataFrame, group_col: pd.Series) -> pd.Series:
    """The contingency label of every observation, for the per-CO KPI breakdowns.

    Rows with no contingency (the N-state) are kept as their own group rather than
    dropped, so the per-CO views cover the same observations as the overall KPIs.
    """
    return group_col.reindex(comparison.index).fillna(BASE_CASE_GROUP_LABEL)


def _element_label(index, id_col: pd.Series, name_col: pd.Series = None,
                    side_col: pd.Series = None) -> str:
    """Identify one observation's element for reporting: id, plus name and side if given."""
    label = str(id_col.get(index, index))
    detail = [str(column.get(index)) for column in (name_col, side_col)
              if column is not None and pd.notna(column.get(index))]
    return f"{label} ({', '.join(detail)})" if detail else label


def _group_label(index, group_col: pd.Series = None) -> str:
    """The contingency label of one observation; the N-state carries the base-case label."""
    if group_col is None:
        return BASE_CASE_GROUP_LABEL
    value = group_col.get(index)
    return BASE_CASE_GROUP_LABEL if pd.isna(value) else str(value)


def flow_deviation_kpis(comparison: pd.DataFrame, id_col: pd.Series = None,
                         group_col: pd.Series = None, name_col: pd.Series = None,
                         side_col: pd.Series = None) -> dict:
    """MAE, RMSE, P95 and max absolute error between AC and DC flows, in A.

    When `id_col` is given, the element and the contingency that attained the maximum
    error are reported alongside the value, as the specification requires: the worst
    single mismatch is only actionable if it can be traced back to where it happened.
    """
    errors = comparison["abs_error"].dropna()
    if errors.empty:
        return {"MAE (A)": np.nan, "RMSE (A)": np.nan, "P95 Error (A)": np.nan, "Max Error (A)": np.nan}
    kpis = {
        "MAE (A)": errors.mean(),
        "RMSE (A)": np.sqrt((errors ** 2).mean()),
        "P95 Error (A)": errors.quantile(0.95),
        "Max Error (A)": errors.max(),
    }
    if id_col is not None:
        worst = errors.idxmax()
        kpis["Max Error Element"] = _element_label(worst, id_col, name_col, side_col)
        kpis["Max Error Contingency"] = _group_label(worst, group_col)
    return kpis


def mae_per_contingency(comparison: pd.DataFrame, group_col: pd.Series) -> pd.Series:
    """MAE_per_CO: the MAE (A) of one contingency group, worst first."""
    groups = _contingency_groups(comparison, group_col)
    return comparison["abs_error"].groupby(groups).mean().sort_values(ascending=False)


def rmse_per_contingency(comparison: pd.DataFrame, group_col: pd.Series) -> pd.Series:
    """RMSE_per_CO: the RMSE (A) of one contingency group, worst first."""
    groups = _contingency_groups(comparison, group_col)
    squared_error = (comparison["dc_value"] - comparison["ac_value"]) ** 2
    return squared_error.groupby(groups).mean().pow(0.5).sort_values(ascending=False)


def loading_limit_kpis(comparison: pd.DataFrame) -> dict:
    """Signed loading-% deviation and margin error (A), overall and near the thermal limit."""
    kpis = {
        "Mean Loading % Deviation (DC-AC)": comparison["signed_loading_deviation"].mean(),
        "Mean Margin Error (DC-AC) (A)": comparison["signed_margin_error"].mean(),
    }
    for threshold in NEAR_LIMIT_THRESHOLDS_PCT:
        subset = comparison[comparison["ac_loading_pct"] >= threshold]
        kpis[f"Near-Limit ({threshold}%+) Mean Loading % Deviation"] = (
            subset["signed_loading_deviation"].mean() if not subset.empty else np.nan
        )
    return kpis


def loading_deviation_per_contingency(comparison: pd.DataFrame, group_col: pd.Series) -> pd.Series:
    """Mean signed loading-% deviation (DC-AC) within each contingency group.

    Sorted by magnitude, worst first: the sign carries the over/underestimation, so the
    largest deviation in either direction is the one worth looking at.
    """
    groups = _contingency_groups(comparison, group_col)
    deviation = comparison["signed_loading_deviation"].groupby(groups).mean()
    return deviation.sort_values(key=abs, ascending=False)


def margin_error_per_contingency(comparison: pd.DataFrame, group_col: pd.Series) -> pd.Series:
    """Mean signed margin error (DC-AC) in A within each contingency group, worst first."""
    groups = _contingency_groups(comparison, group_col)
    margin_error = comparison["signed_margin_error"].groupby(groups).mean()
    return margin_error.sort_values(key=abs, ascending=False)


def violation_kpis(comparison: pd.DataFrame) -> dict:
    """False negatives/positives and their rates, using loading% > 100 as the violation flag.

    Both flags are well defined here because prepare_comparison has already excluded
    observations with a missing AC or DC value.
    """
    ac_violation = comparison["ac_violation"]
    dc_violation = comparison["dc_violation"]

    false_negatives = ac_violation & ~dc_violation
    false_positives = ~ac_violation & dc_violation

    ac_violation_count = int(ac_violation.sum())
    ac_secure_count = int((~ac_violation).sum())

    return {
        "False Negatives": int(false_negatives.sum()),
        "False Positives": int(false_positives.sum()),
        "Missed Overload Rate": (
            false_negatives.sum() / ac_violation_count if ac_violation_count else np.nan
        ),
        "False Alarm Rate": (
            false_positives.sum() / ac_secure_count if ac_secure_count else np.nan
        ),
        "_false_negatives_mask": false_negatives,
        "_false_positives_mask": false_positives,
    }


def severity_kpis(comparison: pd.DataFrame, violation: dict) -> dict:
    """Volume (sum) and average severity of missed/false overloads.

    Severity is reported both as the overload magnitude above the limit (A) and as the
    loading of the affected cases (%).
    """
    false_negatives = violation["_false_negatives_mask"]
    false_positives = violation["_false_positives_mask"]

    missed = comparison.loc[false_negatives]
    false = comparison.loc[false_positives]

    missed_overload = (missed["ac_value"].abs() - missed["limit"]).clip(lower=0)
    false_overload = (false["dc_value"].abs() - false["limit"]).clip(lower=0)

    return {
        "Missed Overload Volume (A)": missed_overload.sum(),
        "False Overload Volume (A)": false_overload.sum(),
        "False Negatives Avg Overload (A)": missed_overload.mean() if not missed.empty else np.nan,
        "False Positives Avg Overload (A)": false_overload.mean() if not false.empty else np.nan,
        "False Negatives Avg Loading %": missed["ac_loading_pct"].mean() if not missed.empty else np.nan,
        "False Positives Avg Loading %": false["dc_loading_pct"].mean() if not false.empty else np.nan,
    }


def severity_volumes_per_contingency(comparison: pd.DataFrame, violation: dict,
                                      group_col: pd.Series) -> dict:
    """Missed and false overload volume (A) within each contingency group, worst first.

    Only the contingencies that actually contribute appear: a group with no false negative
    contributes nothing to the missed volume, so listing it as 0 A would pad the report
    with every secure contingency in the dataset.
    """
    groups = _contingency_groups(comparison, group_col)

    missed = comparison.loc[violation["_false_negatives_mask"]]
    false = comparison.loc[violation["_false_positives_mask"]]

    missed_overload = (missed["ac_value"].abs() - missed["limit"]).clip(lower=0)
    false_overload = (false["dc_value"].abs() - false["limit"]).clip(lower=0)

    return {
        "Missed Overload Volume (A)": (
            missed_overload.groupby(groups.loc[missed.index]).sum().sort_values(ascending=False)
        ),
        "False Overload Volume (A)": (
            false_overload.groupby(groups.loc[false.index]).sum().sort_values(ascending=False)
        ),
    }


def ranking_kpis(comparison: pd.DataFrame, id_col: pd.Series, group_col: pd.Series = None,
                  top_n: int = TOP_N_DEFAULT) -> dict:
    """Top-N critical element overlap and worst-case match rate, between AC and DC rankings.

    When `group_col` is given (e.g. contingency_id), ranking is done per group
    and the reported figures are averaged across groups; otherwise ranking is
    global (base-case usage). Rows with no contingency (the N-state) form their
    own group rather than being dropped.

    The per-CO overlap values are returned under `_top_n_overlap_per_co` for reporting:
    the specification asks for this KPI per CO and then averaged, and the average alone
    hides which contingencies DC ranks differently. Groups with a single monitored element
    cannot be ranked and are excluded; they are counted and named so the per-CO list is
    visibly incomplete rather than silently so.
    """
    df = comparison.copy()
    df["_id"] = id_col.reindex(df.index)

    def _group_metrics(group_df):
        if len(group_df) < 2:
            return None
        ac_top = set(group_df.nlargest(min(top_n, len(group_df)), "ac_loading_pct")["_id"])
        dc_top = set(group_df.nlargest(min(top_n, len(group_df)), "dc_loading_pct")["_id"])
        overlap = len(ac_top & dc_top) / top_n  # per spec: divided by N, not by the group size
        ac_worst = group_df.loc[group_df["ac_loading_pct"].idxmax(), "_id"]
        dc_worst = group_df.loc[group_df["dc_loading_pct"].idxmax(), "_id"]
        return overlap, ac_worst == dc_worst

    if group_col is not None:
        df["_group"] = _contingency_groups(df, group_col)
        overlaps, matches, excluded = {}, [], []
        for group, group_df in df.groupby("_group"):
            metrics = _group_metrics(group_df)
            if metrics is None:
                excluded.append(group)
                continue
            overlaps[group] = metrics[0]
            matches.append(metrics[1])
        # Lowest overlap first: those are the contingencies DC ranks least like AC.
        overlap_per_co = pd.Series(overlaps, dtype=float).sort_values()
        return {
            "Top-N Critical Element Overlap": np.nanmean(overlap_per_co) if len(overlap_per_co) else np.nan,
            "Worst-Case Match Rate": np.mean(matches) if matches else np.nan,
            "Contingency Groups Ranked": len(overlap_per_co),
            "Contingency Groups Excluded (<2 monitored elements)": len(excluded),
            "_top_n_overlap_per_co": overlap_per_co,
            "_excluded_groups": excluded,
        }

    metrics = _group_metrics(df)
    if metrics is None:
        return {"Top-N Critical Element Overlap": np.nan, "Worst-Case Match Rate": np.nan}
    overlap, match = metrics
    return {"Top-N Critical Element Overlap": overlap, "Worst-Case Match Rate": float(match)}


def _speed_up(ac_time, dc_time):
    """AC wall-clock time / DC wall-clock time: > 1 means DC is that many times faster."""
    if not ac_time or not dc_time:
        return np.nan
    return ac_time / dc_time


def performance_kpis(ac_lf_time, dc_lf_time, ac_sa_time=None, dc_sa_time=None) -> dict:
    """Wall-clock time (s) of the AC and DC computation stages, and the DC speed-up.

    Bounded to the load flow and contingency analysis stages; sensitivity and RAO
    timings are deliberately out of scope. Contingency figures are only reported when
    the security analysis ran.
    """
    kpis = {
        "AC loadflow (s)": ac_lf_time,
        "DC loadflow (s)": dc_lf_time,
        "Loadflow DC speed-up (AC/DC)": _speed_up(ac_lf_time, dc_lf_time),
    }
    if ac_sa_time is not None and dc_sa_time is not None:
        kpis["AC contingency analysis (s)"] = ac_sa_time
        kpis["DC contingency analysis (s)"] = dc_sa_time
        kpis["Contingency analysis DC speed-up (AC/DC)"] = _speed_up(ac_sa_time, dc_sa_time)
        kpis["Total AC (s)"] = ac_lf_time + ac_sa_time
        kpis["Total DC (s)"] = dc_lf_time + dc_sa_time
        kpis["Total DC speed-up (AC/DC)"] = _speed_up(ac_lf_time + ac_sa_time, dc_lf_time + dc_sa_time)
    return {name: value for name, value in kpis.items() if value is not None}


def log_kpi_table(title: str, kpis: dict) -> None:
    """Render a KPI dict as a PrettyTable and log it."""
    table = PrettyTable()
    table.field_names = ["KPI", "Value"]
    table.align["KPI"] = "l"
    table.align["Value"] = "r"
    for name, value in kpis.items():
        if name.startswith("_"):
            continue
        if isinstance(value, float):
            table.add_row([name, f"{value:.3f}"])
        else:
            table.add_row([name, value])
    logger.info("%s\n%s", title, table)


_UNIT_SUFFIXES = {" (A)": "A", " (s)": "s"}


def _kpi_unit(name: str) -> str:
    """The unit of a KPI, taken from the label it already carries in the log."""
    for suffix, unit in _UNIT_SUFFIXES.items():
        if name.endswith(suffix):
            return unit
    return "%" if "%" in name else ""


def overall_rows(dataset: str, kpis: dict) -> list:
    """Workbook rows for one KPI table; private entries are skipped as they are in the log."""
    return [
        {"Dataset": dataset, "KPI": name, "Contingency": OVERALL_LABEL,
         "Value": value, "Unit": _kpi_unit(name)}
        for name, value in kpis.items() if not name.startswith("_")
    ]


def per_co_rows(dataset: str, kpi_name: str, values: pd.Series) -> list:
    """Workbook rows for one per-CO KPI: every contingency it was computed for."""
    unit = _kpi_unit(kpi_name)
    return [
        {"Dataset": dataset, "KPI": kpi_name, "Contingency": str(group),
         "Value": value, "Unit": unit}
        for group, value in values.items()
    ]


def _report_per_co(dataset: str, kpi_name: str, values: pd.Series, descriptor: str = "worst") -> list:
    """Log the first few contingencies of one per-CO KPI, and return all of them as rows.

    Each per-CO KPI function returns its values already ordered with the contingency of most
    interest first - largest error or volume, largest deviation either side of zero, lowest
    overlap - so the log summary is the head of that order and `descriptor` only names the
    direction for the reader.
    """
    summary = values.head(PER_CO_LOG_SUMMARY_N)
    log_kpi_table(
        f"{dataset} - {kpi_name} per Contingency, {descriptor} {len(summary)} of "
        f"{len(values)} (full detail in {KPI_WORKBOOK_FILENAME})",
        summary.to_dict(),
    )
    return per_co_rows(dataset, kpi_name, values)


def write_kpi_workbook(rows: list, filename: str = KPI_WORKBOOK_FILENAME) -> str:
    """Write every collected KPI result to one sheet under output/, and return the path.

    One combined sheet rather than one per KPI: the overall and per-CO rows of a KPI stay
    together and the Contingency column separates them, so the whole record can be filtered
    or pivoted in one place. This is the complete record; the log is a summary of it.
    """
    if not rows:
        logger.warning("No KPI results collected, %s not written", filename)
        return None

    path = output_path(filename)
    results = pd.DataFrame(rows, columns=["Dataset", "KPI", "Contingency", "Value", "Unit"])
    results.to_excel(path, sheet_name=KPI_WORKBOOK_SHEET, index=False)
    logger.info("Wrote %d KPI result rows to %s", len(results), path)
    return path


def log_all_priority1_kpis(label: str, comparison: pd.DataFrame, id_col: pd.Series = None,
                            group_col: pd.Series = None, name_col: pd.Series = None,
                            side_col: pd.Series = None) -> list:
    """Compute and log every Priority 1 KPI group for one comparison dataset.

    Returns the workbook rows for this dataset, which `write_kpi_workbook()` writes out in
    full. The overall KPIs are logged as they are computed; of the per-CO KPIs the log keeps
    only the first PER_CO_LOG_SUMMARY_N contingencies, so that adding a per-CO breakdown to
    a KPI does not cost the log one line per contingency.
    """
    rows = []

    flow_deviation = flow_deviation_kpis(comparison, id_col, group_col, name_col, side_col)
    log_kpi_table(f"{label} - Flow Deviation KPIs", flow_deviation)
    rows += overall_rows(label, flow_deviation)

    if group_col is not None:
        rows += _report_per_co(label, "MAE (A)", mae_per_contingency(comparison, group_col))
        rows += _report_per_co(label, "RMSE (A)", rmse_per_contingency(comparison, group_col))

    loading_limit = loading_limit_kpis(comparison)
    log_kpi_table(f"{label} - Loading & Limit-Based KPIs", loading_limit)
    rows += overall_rows(label, loading_limit)

    if group_col is not None:
        # Both are signed, so the head of the order is the largest deviation either way.
        rows += _report_per_co(
            label, "Mean Loading % Deviation (DC-AC)",
            loading_deviation_per_contingency(comparison, group_col), descriptor="largest",
        )
        rows += _report_per_co(
            label, "Mean Margin Error (DC-AC) (A)",
            margin_error_per_contingency(comparison, group_col), descriptor="largest",
        )

    violation = violation_kpis(comparison)
    log_kpi_table(f"{label} - Violation Detection KPIs", violation)
    rows += overall_rows(label, violation)

    severity = severity_kpis(comparison, violation)
    log_kpi_table(f"{label} - Severity-Based KPIs", severity)
    rows += overall_rows(label, severity)

    if group_col is not None:
        volumes = severity_volumes_per_contingency(comparison, violation, group_col)
        for name, volume_per_co in volumes.items():
            rows += _report_per_co(label, name, volume_per_co)

    if id_col is not None:
        ranking = ranking_kpis(comparison, id_col, group_col)
        log_kpi_table(f"{label} - Ranking & Critical Element KPIs", ranking)
        rows += overall_rows(label, ranking)
        rows += _report_top_n_overlap_per_co(label, ranking)

    return rows


def _report_top_n_overlap_per_co(label: str, ranking: dict) -> list:
    """Log the lowest per-CO Top-N overlaps and the groups the ranking could not cover.

    The excluded groups stay in the log rather than only in the workbook: how many
    contingencies this KPI could not be computed for is needed to read the KPI at all.
    """
    overlap_per_co = ranking.get("_top_n_overlap_per_co")
    if overlap_per_co is None or overlap_per_co.empty:
        return []

    # Lowest overlap first: those are the contingencies DC ranks least like AC.
    rows = _report_per_co(label, "Top-N Critical Element Overlap", overlap_per_co, descriptor="lowest")

    excluded = ranking.get("_excluded_groups") or []
    if excluded:
        logger.info(
            "%s - %d contingency group(s) excluded from the Top-N overlap: fewer than 2 "
            "monitored elements, so AC and DC rankings cannot be compared: %s",
            label, len(excluded), ", ".join(str(group) for group in excluded),
        )
        rows += [
            {"Dataset": label, "KPI": "Top-N Overlap Excluded Group", "Contingency": str(group),
             "Value": "fewer than 2 monitored elements", "Unit": ""}
            for group in excluded
        ]
    return rows
