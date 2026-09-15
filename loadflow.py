"""AC/DC load flow execution and base-case (N-0) AC vs DC comparison dataframes."""

import logging
import time

import numpy as np
import pandas as pd
import pypowsybl as pp

from rosc_acdc import config

logger = logging.getLogger(__name__)

# IIDM side label -> the column suffix pypowsybl uses for that side (i1/p1, i2/p2, i3/p3).
SIDE_COLUMN_SUFFIX = {"ONE": 1, "TWO": 2, "THREE": 3}


def run_ac(network):
    parameters = pp.loadflow.Parameters()
    parameters.dc = False
    t0 = time.perf_counter()
    ac_result = pp.loadflow.run_ac(network, parameters)
    ac_lf_time = time.perf_counter() - t0
    logger.info("AC loadflow: %.3f s", ac_lf_time)
    return ac_result, ac_lf_time


def run_dc(network):
    parameters = pp.loadflow.Parameters()
    parameters.dc = True
    t0 = time.perf_counter()
    dc_result = pp.loadflow.run_dc(network, parameters)
    dc_lf_time = time.perf_counter() - t0
    logger.info("DC loadflow: %.3f s", dc_lf_time)
    return dc_result, dc_lf_time


def _by_side(current_per_side):
    """Turn {side: current per element} into one row per (element_id, side)."""
    return (
        pd.DataFrame(current_per_side)
        .rename_axis(index="element_id", columns="side")
        .stack()
    )


def branch_current(elements_df):
    """AC current (A) per side, as left on the network by the AC load flow.

    One row per (element, side) - the shape security_analysis.branch_results() uses - so
    each side can be compared against its own thermal limit.
    """
    return _by_side({
        side: elements_df[f"i{suffix}"].abs()
        for side, suffix in SIDE_COLUMN_SUFFIX.items() if f"i{suffix}" in elements_df
    })


def branch_current_dc(elements_df, voltage_levels):
    """DC current (A) per side, rebuilt from the DC active flow: |P| / (sqrt(3) * Vnom).

    The DC load flow leaves i1/q1 unset on the network, so the current is derived from
    the nominal voltage of each side - the same convention pypowsybl uses for DC
    security-analysis branch results, which keeps the base case comparable with the SA
    dataset. Each side is taken separately because each carries its own voltage level.
    """
    currents = {}
    for side, suffix in SIDE_COLUMN_SUFFIX.items():
        if f"p{suffix}" not in elements_df:
            continue
        nominal_v = elements_df[f"voltage_level{suffix}_id"].map(voltage_levels["nominal_v"])
        currents[side] = elements_df[f"p{suffix}"].abs().mul(1000).div(np.sqrt(3) * nominal_v)
    return _by_side(currents)


def _compare_ac_dc(ac_currents, dc_currents, patl_per_side, name_lookup, active_threshold_pct,
                    kind_label):
    """Build the AC/DC comparison dataframe for one element type (lines, 2W-TR, 3W-TR).

    Each element is reported on its binding side: the side whose AC loading against that
    side's own PATL is highest. The side is chosen on the AC loading because AC is the
    reference the study measures DC against, and the DC current is then read from that same
    side, so the reported deviation stays a DC-approximation error rather than a difference
    between two terminals. An element rated on one side only has that side as its binding
    side. AC LF / DC LF / PATL are all currents in A; the % columns are loadings in %.
    """
    per_side = pd.DataFrame(index=ac_currents.index)
    per_side["AC LF"] = ac_currents
    per_side["DC LF"] = dc_currents
    per_side["PATL"] = patl_per_side
    per_side["AC LF %"] = per_side["AC LF"].div(per_side["PATL"]).mul(100)

    # A side with no rating, or no current, has no loading to rank it by.
    ranked = per_side.dropna(subset=["AC LF %"])
    binding_side = ranked.groupby(level="element_id")["AC LF %"].idxmax()

    elements = ac_currents.index.get_level_values("element_id").unique()
    comparison = ranked.loc[list(binding_side)].reset_index("side").rename(
        columns={"side": "Binding Side"},
    )
    comparison = comparison.reindex(elements[elements.isin(comparison.index)])
    comparison = comparison[["AC LF", "DC LF", "PATL", "Binding Side", "AC LF %"]]
    comparison["DC LF %"] = comparison["DC LF"].div(comparison["PATL"]).mul(100)
    comparison["(DC-AC)/AC %"] = (
        (comparison["DC LF"].sub(comparison["AC LF"])).div(comparison["AC LF"])
    ).mul(100).fillna(0)

    for element_id in elements.difference(comparison.index):
        name = name_lookup.loc[element_id]["name"] if element_id in name_lookup.index else ""
        logger.warning("No PATL found for %s: %s (%s)", kind_label, element_id, name)

    if not comparison.empty:
        logger.info("Binding side for %s: %s", kind_label,
                    comparison["Binding Side"].value_counts().to_dict())

    active_comparison = comparison[comparison["AC LF %"] > active_threshold_pct]
    return active_comparison, comparison.index.tolist()


def build_base_case_comparison(limits, hv_lines, hv_transformers, hv_transformers3,
                                ac_ln_i, ac_tr_i, ac_tr3_i,
                                dc_ln_i, dc_tr_i, dc_tr3_i):
    """Build the three base-case (N-0) AC vs DC comparison dataframes (lines, 2W-TR, 3W-TR).

    The ac_*/dc_* series carry one current (A) per (element, side); each element is then
    reported on its binding side, against that side's own CURRENT PATL. CIM current limits
    are terminal-specific, so a branch can be rated differently on each side - a transformer
    most of all, where the two ratings express the same MVA at different voltages - and
    side ONE is not reliably the side that binds.
    """
    patl = limits[(limits["acceptable_duration"] == -1) & (limits["type"] == "CURRENT")]
    # One PATL per (element, side); the most restrictive rating wins if a side carries several.
    patl_per_side = patl.groupby(["element_id", "side"])["value"].min()

    lines_cmp, final_lines_id = _compare_ac_dc(
        ac_ln_i, dc_ln_i, patl_per_side, hv_lines, config.BASE_CASE_ACTIVE_THRESHOLD_PCT, "line",
    )
    transformers_cmp, final_tr_id = _compare_ac_dc(
        ac_tr_i, dc_tr_i, patl_per_side, hv_transformers, config.BASE_CASE_ACTIVE_THRESHOLD_PCT,
        "2-winding transformer",
    )
    transformers3_cmp, final_tr3_id = _compare_ac_dc(
        ac_tr3_i, dc_tr3_i, patl_per_side, hv_transformers3, config.BASE_CASE_ACTIVE_THRESHOLD_PCT,
        "3-winding transformer",
    )

    assessed = final_lines_id + final_tr_id + final_tr3_id
    single_sided = _single_sided_elements(patl_per_side, assessed)

    return (
        lines_cmp, transformers_cmp, transformers3_cmp,
        final_lines_id, final_tr_id, final_tr3_id,
        single_sided,
    )


def _single_sided_elements(patl_per_side, assessed):
    """The assessed elements carrying a CURRENT PATL on one side only.

    A data-quality note about the ratings in the input: such an element has no second side to
    compare, so its binding side is decided by the data rather than by the load flow. Counted
    over the elements the base case assessed, before the loading threshold is applied.
    """
    rated_sides = patl_per_side.groupby(level="element_id").size().reindex(assessed)
    return rated_sides[rated_sides == 1].index.tolist()
