"""Implements: Flow deviation, Loading & limit-based, Violation detection,
Severity-based, Ranking & critical element, and Performance KPIs.
"""

import logging

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from rosc_acdc import config
from rosc_acdc.paths import output_path

logger = logging.getLogger(__name__)

BASE_CASE_GROUP_LABEL = "N-0 (base case)"

# The three near-limit thresholds this study reported before the KPI workbook existed.
# Core_AC_DC_KPI.xlsx carries exactly one near-limit column, at
# config.KPI_NEAR_LIMIT_THRESHOLD_PCT; kpi_results.xlsx and the log keep all three, since
# loading_limit_kpis() takes whatever thresholds it is handed and nothing was gained by
# dropping two of them from an output that already had room.
LEGACY_NEAR_LIMIT_THRESHOLDS_PCT = (80, 90, 95)

# How many contingencies of each per-CO KPI the log summary shows. This governs the size of
# the log only: every contingency is computed, and all of them reach the workbook below.
PER_CO_LOG_SUMMARY_N = 5

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

    out["signed_error"] = out["dc_value"] - out["ac_value"]
    out["abs_error"] = out["signed_error"].abs()
    out["signed_loading_deviation"] = out["dc_loading_pct"] - out["ac_loading_pct"]
    out["margin_ac"] = out["limit"] - out["ac_value"].abs()
    out["margin_dc"] = out["limit"] - out["dc_value"].abs()
    out["signed_margin_error"] = out["margin_dc"] - out["margin_ac"]
    # `>=`, not `>`: an element exactly at the threshold is in violation. This matches the
    # near-limit selection below and the workbook spec's own wording ("in violation when
    # loading >= this value"). It is a real behaviour change from the earlier `>`.
    out["ac_violation"] = out["ac_loading_pct"] >= config.KPI_VIOLATION_THRESHOLD_PCT
    out["dc_violation"] = out["dc_loading_pct"] >= config.KPI_VIOLATION_THRESHOLD_PCT
    # Overload magnitude above the limit, per side of the comparison. Computed once here so
    # the severity KPIs and their primitives read the same numbers.
    out["ac_overload"] = (out["ac_value"].abs() - out["limit"]).clip(lower=0)
    out["dc_overload"] = (out["dc_value"].abs() - out["limit"]).clip(lower=0)
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


def max_error_per_contingency(comparison: pd.DataFrame, group_col: pd.Series) -> pd.Series:
    """MaxError_per_CO: the largest absolute error (A) in one contingency group, worst first.

    Additive to the overall maximum and to the element/contingency that attained it: the
    single worst mismatch says nothing about how the rest of the contingencies behave.
    """
    groups = _contingency_groups(comparison, group_col)
    return comparison["abs_error"].groupby(groups).max().sort_values(ascending=False)


def near_limit_subset(comparison: pd.DataFrame, threshold_pct: float = None) -> pd.DataFrame:
    """The observations whose AC loading reaches the near-limit threshold.

    One place decides this population, so the near-limit KPI and its primitives
    (N_NearLimit_Obs, Sum_NearLimit_LoadingDev_pp, N_NearLimit_Underest) cannot drift
    apart. `>=` is deliberate and matches the violation test.
    """
    if threshold_pct is None:
        threshold_pct = config.KPI_NEAR_LIMIT_THRESHOLD_PCT
    return comparison[comparison["ac_loading_pct"] >= threshold_pct]


def loading_limit_kpis(comparison: pd.DataFrame, thresholds_pct=None) -> dict:
    """Signed loading-% deviation and margin error (A), overall and near the thermal limit.

    `thresholds_pct` is any iterable of near-limit thresholds: the KPI workbook passes the
    single configured one, the legacy log and workbook pass all three they always reported.
    """
    if thresholds_pct is None:
        thresholds_pct = (config.KPI_NEAR_LIMIT_THRESHOLD_PCT,)
    kpis = {
        "Mean Loading % Deviation (DC-AC)": comparison["signed_loading_deviation"].mean(),
        "Mean Margin Error (DC-AC) (A)": comparison["signed_margin_error"].mean(),
    }
    for threshold in thresholds_pct:
        subset = near_limit_subset(comparison, threshold)
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


def violation_counts_per_contingency(comparison: pd.DataFrame, violation: dict,
                                      group_col: pd.Series) -> dict:
    """False negative and false positive counts within each contingency group, worst first.

    Every group is listed, the zeros included: unlike the severity volumes below, these
    counts are the populations the per-CO rates are built on, so a group DC got right is
    a result to read and not an absence.
    """
    groups = _contingency_groups(comparison, group_col)
    return {
        "False Negatives": (
            violation["_false_negatives_mask"].groupby(groups).sum().sort_values(ascending=False)
        ),
        "False Positives": (
            violation["_false_positives_mask"].groupby(groups).sum().sort_values(ascending=False)
        ),
    }


def violation_rates_per_contingency(comparison: pd.DataFrame, violation: dict,
                                     group_col: pd.Series) -> dict:
    """Missed overload and false alarm rate within each contingency group, worst first.

    Each rate is computed inside its own group, against that group's own denominator: the
    AC violations it holds for the missed overload rate, the AC-secure observations for the
    false alarm rate. A group whose denominator is empty has no rate - not a rate of zero -
    so it is excluded and counted, as the Top-N overlap does for groups it cannot rank. The
    excluded groups are returned under `_excluded_groups` for reporting.
    """
    groups = _contingency_groups(comparison, group_col)
    ac_violation = comparison["ac_violation"]

    def _rate_per_group(hits, population):
        denominator = population.groupby(groups).sum()
        defined = denominator > 0
        rate = hits.groupby(groups).sum()[defined] / denominator[defined]
        return rate.sort_values(ascending=False), list(denominator.index[~defined])

    missed_rate, missed_excluded = _rate_per_group(violation["_false_negatives_mask"], ac_violation)
    false_rate, false_excluded = _rate_per_group(violation["_false_positives_mask"], ~ac_violation)

    return {
        "Missed Overload Rate": missed_rate,
        "False Alarm Rate": false_rate,
        "_excluded_groups": {
            "Missed Overload Rate - Contingency Groups Excluded (no AC violation)": missed_excluded,
            "False Alarm Rate - Contingency Groups Excluded (no AC-secure observation)": false_excluded,
        },
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

    missed_overload = missed["ac_overload"]
    false_overload = false["dc_overload"]

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

    missed_overload = missed["ac_overload"]
    false_overload = false["dc_overload"]

    return {
        "Missed Overload Volume (A)": (
            missed_overload.groupby(groups.loc[missed.index]).sum().sort_values(ascending=False)
        ),
        "False Overload Volume (A)": (
            false_overload.groupby(groups.loc[false.index]).sum().sort_values(ascending=False)
        ),
    }


def severity_averages_per_contingency(comparison: pd.DataFrame, violation: dict,
                                       group_col: pd.Series) -> dict:
    """Average severity of the missed/false overloads within each contingency group, worst first.

    Both readings of the KPI are kept per CO exactly as they are overall: the loading of the
    affected cases (%) and the overload magnitude above the limit (A). As with the volumes
    above, only the contingencies that carry a false negative (resp. false positive) appear -
    a group with none has no cases to average.
    """
    groups = _contingency_groups(comparison, group_col)

    missed = comparison.loc[violation["_false_negatives_mask"]]
    false = comparison.loc[violation["_false_positives_mask"]]

    missed_groups = groups.loc[missed.index]
    false_groups = groups.loc[false.index]

    missed_overload = missed["ac_overload"]
    false_overload = false["dc_overload"]

    return {
        "False Negatives Avg Overload (A)": (
            missed_overload.groupby(missed_groups).mean().sort_values(ascending=False)
        ),
        "False Positives Avg Overload (A)": (
            false_overload.groupby(false_groups).mean().sort_values(ascending=False)
        ),
        "False Negatives Avg Loading %": (
            missed["ac_loading_pct"].groupby(missed_groups).mean().sort_values(ascending=False)
        ),
        "False Positives Avg Loading %": (
            false["dc_loading_pct"].groupby(false_groups).mean().sort_values(ascending=False)
        ),
    }


def ranking_kpis(comparison: pd.DataFrame, id_col: pd.Series, group_col: pd.Series = None,
                  top_n: int = None) -> dict:
    """Top-N critical element overlap and worst-case match rate, between AC and DC rankings.

    When `group_col` is given (e.g. contingency_id), ranking is done per group and the
    reported figures are pooled across groups; otherwise ranking is global (base-case
    usage). Rows with no contingency (the N-state) form their own group rather than
    being dropped.

    The overall overlap is the pooled ratio `SUM(overlap count) / SUM(N)`, not the mean of
    the per-group ratios. Those two differ whenever the groups are unequally sized, and
    Dim_KPI's K16 formula is the pooled one - averaging pre-computed ratios is exactly what
    the workbook's Rule 2 forbids. The summed numerator and denominator are returned as
    `Top-N Overlap Count` and `TopN_N`, which are also the workbook's own primitive columns.

    The per-CO overlap values are returned under `_top_n_overlap_per_co` for reporting:
    the average alone hides which contingencies DC ranks differently. Groups with a single
    monitored element cannot be ranked and are excluded; they are counted and named so the
    per-CO list is visibly incomplete rather than silently so.
    """
    if top_n is None:
        top_n = config.KPI_TOP_N

    df = comparison.copy()
    df["_id"] = id_col.reindex(df.index)

    def _group_metrics(group_df):
        if len(group_df) < 2:
            return None
        ac_top = set(group_df.nlargest(min(top_n, len(group_df)), "ac_loading_pct")["_id"])
        dc_top = set(group_df.nlargest(min(top_n, len(group_df)), "dc_loading_pct")["_id"])
        overlap_count = len(ac_top & dc_top)  # per spec: divided by N, not by the group size
        ac_worst = group_df.loc[group_df["ac_loading_pct"].idxmax(), "_id"]
        dc_worst = group_df.loc[group_df["dc_loading_pct"].idxmax(), "_id"]
        return overlap_count, ac_worst == dc_worst

    if group_col is not None:
        df["_group"] = _contingency_groups(df, group_col)
        overlaps, counts, matches, excluded = {}, 0, [], []
        for group, group_df in df.groupby("_group"):
            metrics = _group_metrics(group_df)
            if metrics is None:
                excluded.append(group)
                continue
            counts += metrics[0]
            overlaps[group] = metrics[0] / top_n
            matches.append(metrics[1])
        # Lowest overlap first: those are the contingencies DC ranks least like AC.
        overlap_per_co = pd.Series(overlaps, dtype=float).sort_values()
        denominator = len(overlap_per_co) * top_n
        return {
            "Top-N Critical Element Overlap": counts / denominator if denominator else np.nan,
            "Worst-Case Match Rate": np.mean(matches) if matches else np.nan,
            "Contingency Groups Ranked": len(overlap_per_co),
            "Contingency Groups Excluded (<2 monitored elements)": len(excluded),
            "_top_n_overlap_count": counts,
            "_top_n_denominator": denominator,
            "_worst_case_matches": int(np.sum(matches)) if matches else 0,
            "_worst_case_total": len(matches),
            "_top_n_overlap_per_co": overlap_per_co,
            "_excluded_groups": excluded,
        }

    metrics = _group_metrics(df)
    if metrics is None:
        return {
            "Top-N Critical Element Overlap": np.nan, "Worst-Case Match Rate": np.nan,
            "_top_n_overlap_count": 0, "_top_n_denominator": 0,
            "_worst_case_matches": 0, "_worst_case_total": 0,
        }
    overlap_count, match = metrics
    return {
        "Top-N Critical Element Overlap": overlap_count / top_n,
        "Worst-Case Match Rate": float(match),
        "_top_n_overlap_count": overlap_count,
        "_top_n_denominator": top_n,
        "_worst_case_matches": int(match),
        "_worst_case_total": 1,
    }


def element_ranking_kpis(comparison: pd.DataFrame, id_col: pd.Series,
                          top_n: int = None) -> dict:
    """Top-N overlap and worst-case match for one KPI_Data row, ranked over its elements.

    One ranking per row, not one per contingency: `TopN_N` is the configured N on every
    row regardless of Case, so the numerator has to be a single group's overlap count for
    `SUM(TopN_Overlap_Count) / SUM(TopN_N)` to stay in range when rows are rolled up.

    Elements are ranked on their worst loading within the row's population. An element
    appearing under several contingencies would otherwise occupy several of the N slots
    and collapse the comparison to a handful of distinct elements.

    `WorstCase_Match_Rate` is therefore 0 or 1 here - whether AC and DC agree on this
    row's single most loaded element. That is consistent with the two primitives
    `02_kpi_data_schema.md` records as missing for K17, `WorstCase_Match_Flag` and
    `WorstCase_Total`: a flag and a count are what a roll-up of it would need.
    """
    if top_n is None:
        top_n = config.KPI_TOP_N

    unrankable = {
        "TopN_Overlap_Count": np.nan, "TopN_N": top_n,
        "TopN_Overlap_Rate": np.nan, "WorstCase_Match_Rate": np.nan,
    }
    if comparison.empty:
        return unrankable

    elements = pd.DataFrame({
        "ac_loading_pct": comparison["ac_loading_pct"],
        "dc_loading_pct": comparison["dc_loading_pct"],
        "_id": id_col.reindex(comparison.index),
    }).groupby("_id")[["ac_loading_pct", "dc_loading_pct"]].max()

    if len(elements) < 2:
        return unrankable

    ac_top = set(elements.nlargest(min(top_n, len(elements)), "ac_loading_pct").index)
    dc_top = set(elements.nlargest(min(top_n, len(elements)), "dc_loading_pct").index)
    overlap_count = len(ac_top & dc_top)

    return {
        "TopN_Overlap_Count": overlap_count,
        "TopN_N": top_n,
        # Divided by N, as the specification defines it - not by the number of elements
        # ranked, so a population smaller than N cannot reach 1.0.
        "TopN_Overlap_Rate": overlap_count / top_n,
        "WorstCase_Match_Rate": float(
            elements["ac_loading_pct"].idxmax() == elements["dc_loading_pct"].idxmax()
        ),
    }


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
        f"{len(values)} (full detail in {config.LEGACY_KPI_WORKBOOK_FILENAME})",
        summary.to_dict(),
    )
    return per_co_rows(dataset, kpi_name, values)


def write_kpi_workbook(rows: list, filename: str = None) -> str:
    """Write every collected KPI result to one sheet under output/, and return the path.

    One combined sheet rather than one per KPI: the overall and per-CO rows of a KPI stay
    together and the Contingency column separates them, so the whole record can be filtered
    or pivoted in one place. This is the complete record; the log is a summary of it.

    This is the per-contingency detail workbook, and it is not superseded by
    Core_AC_DC_KPI.xlsx: that one reports N-0 + all-COs-combined only, so the per-CO
    series below exists nowhere else. Both are written by every run.
    """
    if filename is None:
        filename = config.LEGACY_KPI_WORKBOOK_FILENAME
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
        rows += _report_per_co(label, "Max Error (A)", max_error_per_contingency(comparison, group_col))

    # All three legacy thresholds, not the single configured one: this workbook and its log
    # have always carried 80/90/95 and nothing is gained by narrowing them here.
    loading_limit = loading_limit_kpis(comparison, LEGACY_NEAR_LIMIT_THRESHOLDS_PCT)
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

    if group_col is not None:
        counts = violation_counts_per_contingency(comparison, violation, group_col)
        for name, count_per_co in counts.items():
            rows += _report_per_co(label, name, count_per_co)
        rates = violation_rates_per_contingency(comparison, violation, group_col)
        for name, rate_per_co in rates.items():
            if name.startswith("_"):
                continue
            rows += _report_per_co(label, name, rate_per_co)
        rows += _report_rate_exclusions(label, rates)

    severity = severity_kpis(comparison, violation)
    log_kpi_table(f"{label} - Severity-Based KPIs", severity)
    rows += overall_rows(label, severity)

    if group_col is not None:
        volumes = severity_volumes_per_contingency(comparison, violation, group_col)
        for name, volume_per_co in volumes.items():
            rows += _report_per_co(label, name, volume_per_co)
        averages = severity_averages_per_contingency(comparison, violation, group_col)
        for name, average_per_co in averages.items():
            rows += _report_per_co(label, name, average_per_co)

    if id_col is not None:
        ranking = ranking_kpis(comparison, id_col, group_col)
        log_kpi_table(f"{label} - Ranking & Critical Element KPIs", ranking)
        rows += overall_rows(label, ranking)
        rows += _report_top_n_overlap_per_co(label, ranking)

    return rows


# ---------------------------------------------------------------------------
# Raw primitives for Core_AC_DC_KPI.xlsx
#
# Every ratio KPI in KPI_Data is paired with the counts and sums behind it, because a
# consumer aggregating beyond that table's grain has to re-derive the ratio from summed
# primitives rather than average pre-computed ratios (the workbook's Rule 2). Nothing
# below is a new measurement: these are the intermediate values the KPI functions above
# already compute and then divide away.
# ---------------------------------------------------------------------------


def contingency_count(group_col: pd.Series) -> int:
    """Distinct contingencies behind a population; the intact state is not one of them."""
    if group_col is None:
        return 0
    return int(group_col.dropna().nunique())


def raw_primitives(comparison: pd.DataFrame, violation: dict, id_col: pd.Series = None,
                    group_col: pd.Series = None) -> dict:
    """The counts and sums KPI_Data carries alongside its computed KPIs."""
    false_negatives = violation["_false_negatives_mask"]
    false_positives = violation["_false_positives_mask"]
    ac_violation = comparison["ac_violation"]
    dc_violation = comparison["dc_violation"]

    return {
        "N_Obs": int(len(comparison)),
        "N_CO": contingency_count(group_col),
        "N_Elements": int(id_col.reindex(comparison.index).nunique()) if id_col is not None else 0,
        "Sum_AbsErr_A": comparison["abs_error"].sum(),
        "Sum_SqErr_A": (comparison["abs_error"] ** 2).sum(),
        "Sum_SignedErr_A": comparison["signed_error"].sum(),
        "Max_AbsErr_A": comparison["abs_error"].max() if len(comparison) else np.nan,
        "Sum_LoadingDev_pp": comparison["signed_loading_deviation"].sum(),
        "Sum_MarginDev_A": comparison["signed_margin_error"].sum(),
        "N_AC_Viol": int(ac_violation.sum()),
        "N_DC_Viol": int(dc_violation.sum()),
        # N_TN is not the existing ac_secure_count: that one is TN + FP.
        "N_TP": int((ac_violation & dc_violation).sum()),
        "N_TN": int((~ac_violation & ~dc_violation).sum()),
        "Sum_MissedOverload_A": comparison.loc[false_negatives, "ac_overload"].sum(),
        "Sum_FalseOverload_A": comparison.loc[false_positives, "dc_overload"].sum(),
        "Sum_FN_Loading_AC_pct": comparison.loc[false_negatives, "ac_loading_pct"].sum(),
        "Sum_FP_Loading_DC_pct": comparison.loc[false_positives, "dc_loading_pct"].sum(),
    }


def near_limit_kpis(comparison: pd.DataFrame, threshold_pct: float = None) -> dict:
    """The near-limit KPI at one threshold, with the primitives that let it be re-derived.

    An "underestimation" is an observation where DC reports a lower loading than AC: the
    error that matters near the limit, because it is the one that hides an overload.
    """
    subset = near_limit_subset(comparison, threshold_pct)
    deviation = subset["signed_loading_deviation"]
    return {
        "Sum_NearLimit_LoadingDev_pp": deviation.sum(),
        "N_NearLimit_Obs": int(len(subset)),
        "NearLimit_Mean_LoadingDev_pp": deviation.mean() if len(subset) else np.nan,
        "N_NearLimit_Underest": int((deviation < 0).sum()),
        "NearLimit_Underest_Rate": (
            float((deviation < 0).sum()) / len(subset) if len(subset) else np.nan
        ),
    }


def co_level_kpis(comparison: pd.DataFrame, group_col: pd.Series = None) -> dict:
    """The contingency-level (K21-K23) KPIs: criticality agreement and worst-CO severity.

    A contingency is "critical" when at least one of its observations is in violation. The
    rates below therefore count contingencies, not elements, which is what separates these
    from the element-level Missed Overload / False Alarm rates.

    Only real contingencies take part: intact-state observations carry no contingency id
    and are dropped here, so the All case's base-case half cannot be counted as a CO.

    Formulas follow Dim_KPI's K21, K22 and K23 exactly:

        K21 = N_CO_MissedCritical / N_CO_AC_Viol
        K22 = N_CO_FalseCritical / (N_CO - N_CO_AC_Viol)
        K23 = Sum_WorstCO_SevDev_A / N_CO_SevDev_Obs

    The CO-level primitives those formulas name are returned under private keys. They are
    deliberately **not** written to KPI_Data: `02_kpi_data_schema.md` fixes that schema and
    records these columns as absent, so these four KPIs ship at their native grain only and
    no roll-up beyond it is possible - or is to be attempted.
    """
    empty = {
        "Missed_Critical_CO_Rate": np.nan,
        "False_Critical_CO_Rate": np.nan,
        "Mean_WorstCO_SevDev_A": np.nan,
        "_n_co": 0, "_n_co_ac_viol": 0, "_n_co_missed_critical": 0,
        "_n_co_false_critical": 0, "_sum_worstco_sevdev_a": np.nan, "_n_co_sevdev_obs": 0,
    }
    if group_col is None:
        return empty

    groups = group_col.reindex(comparison.index)
    per_co = comparison[groups.notna()]
    if per_co.empty:
        return empty
    labels = groups[groups.notna()]

    ac_critical = per_co["ac_violation"].groupby(labels).any()
    dc_critical = per_co["dc_violation"].groupby(labels).any()

    n_co = int(len(ac_critical))
    n_co_ac_viol = int(ac_critical.sum())
    n_co_ac_secure = n_co - n_co_ac_viol
    n_missed = int((ac_critical & ~dc_critical).sum())
    n_false = int((~ac_critical & dc_critical).sum())

    # Worst overload severity in a contingency, AC vs DC. "Overload severity" is the
    # magnitude above the limit and never below zero - the same ac_overload / dc_overload
    # the volume KPIs are built from - so a contingency neither AC nor DC overloads
    # contributes a deviation of zero rather than a margin.
    worst_ac = per_co["ac_overload"].groupby(labels).max()
    worst_dc = per_co["dc_overload"].groupby(labels).max()
    severity_deviation = worst_dc - worst_ac
    sum_severity_deviation = severity_deviation.sum()
    n_severity_observations = int(len(severity_deviation))

    return {
        "Missed_Critical_CO_Rate": n_missed / n_co_ac_viol if n_co_ac_viol else np.nan,
        # Denominator is the AC-secure contingencies, mirroring K11's false alarm rate.
        "False_Critical_CO_Rate": n_false / n_co_ac_secure if n_co_ac_secure else np.nan,
        "Mean_WorstCO_SevDev_A": (
            sum_severity_deviation / n_severity_observations if n_severity_observations
            else np.nan
        ),
        "_n_co": n_co,
        "_n_co_ac_viol": n_co_ac_viol,
        "_n_co_missed_critical": n_missed,
        "_n_co_false_critical": n_false,
        "_sum_worstco_sevdev_a": sum_severity_deviation,
        "_n_co_sevdev_obs": n_severity_observations,
    }


def _report_rate_exclusions(label: str, rates: dict) -> list:
    """Log how many contingency groups each per-CO rate has no denominator for.

    The counts stay in the log for the same reason the Top-N overlap's does: a rate covering
    only part of the contingencies cannot be read without knowing how many it could not be
    computed for. The groups are not named as the Top-N exclusions are - there is one per
    secure contingency, which is most of them.
    """
    counts = {name: len(groups) for name, groups in (rates.get("_excluded_groups") or {}).items()}
    if not counts:
        return []
    log_kpi_table(f"{label} - Per-CO Rate Coverage", counts)
    return overall_rows(label, counts)


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
