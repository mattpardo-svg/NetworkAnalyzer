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


def branch_current(elements_df):
    """AC current (A) per side, as left on the network by the AC load flow.

    One column per side the element carries (ONE/TWO, plus THREE for 3-winding
    transformers), so the base case can compare each side against its own thermal limit
    instead of assuming side ONE is the binding one.
    """
    return pd.DataFrame({
        side: elements_df[f"i{suffix}"].abs()
        for side, suffix in SIDE_COLUMN_SUFFIX.items() if f"i{suffix}" in elements_df
    })


def branch_current_dc(elements_df, voltage_levels):
    """DC current (A) per side, rebuilt from the DC active flow: |P| / (sqrt(3) * Vnom).

    The DC load flow leaves i1/q1 unset on the network, so the current is derived from
    the nominal voltage of each side - the same convention pypowsybl uses for DC
    security-analysis branch results, which keeps the base case comparable with the SA
    dataset.
    """
    currents = {}
    for side, suffix in SIDE_COLUMN_SUFFIX.items():
        if f"p{suffix}" not in elements_df:
            continue
        nominal_v = elements_df[f"voltage_level{suffix}_id"].map(voltage_levels["nominal_v"])
        currents[side] = elements_df[f"p{suffix}"].abs().mul(1000).div(np.sqrt(3) * nominal_v)
    return pd.DataFrame(currents)


def _side_loadings(ac_currents, patl_per_side):
    """AC loading (%) on every side for which the element has both a current and a PATL."""
    patl_per_side = patl_per_side.reindex(ac_currents.index)
    sides = [side for side in SIDE_COLUMN_SUFFIX
             if side in ac_currents.columns and side in patl_per_side.columns]
    return pd.DataFrame(
        {side: ac_currents[side].div(patl_per_side[side]).mul(100) for side in sides},
        index=ac_currents.index,
    )


def _pick_side(per_side, binding_side):
    """Take one value per element, from the side named for that element."""
    pairs = pd.MultiIndex.from_arrays([binding_side.index, binding_side.to_numpy()])
    return per_side.stack().reindex(pairs).set_axis(binding_side.index)


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
    loadings = _side_loadings(ac_currents, patl_per_side)
    rated = loadings.dropna(how="all")  # an element with no PATL on any side cannot be assessed
    binding_side = rated.idxmax(axis=1) if not rated.empty else pd.Series(dtype=object)

    for element_id in ac_currents.index.difference(binding_side.index):
        name = name_lookup.loc[element_id]["name"] if element_id in name_lookup.index else ""
        logger.warning("No PATL found for %s: %s (%s)", kind_label, element_id, name)

    comparison = pd.DataFrame({
        "AC LF": _pick_side(ac_currents, binding_side),
        "DC LF": _pick_side(dc_currents, binding_side),
        "PATL": _pick_side(patl_per_side.reindex(ac_currents.index), binding_side),
        "Binding Side": binding_side,
    })
    comparison["AC LF %"] = comparison["AC LF"].div(comparison["PATL"]).mul(100)
    comparison["DC LF %"] = comparison["DC LF"].div(comparison["PATL"]).mul(100)
    comparison["(DC-AC)/AC %"] = (
        (comparison["DC LF"].sub(comparison["AC LF"])).div(comparison["AC LF"])
    ).mul(100).fillna(0)

    _log_binding_sides(kind_label, comparison, loadings)

    active_comparison = comparison[comparison["AC LF %"] > active_threshold_pct]
    return active_comparison, binding_side.index.tolist()


def _log_binding_sides(kind_label, comparison, loadings):
    """Report which side ended up binding, and what side ONE alone would have got wrong."""
    if comparison.empty:
        return
    side_one = loadings["ONE"].reindex(comparison.index) if "ONE" in loadings.columns else None
    understated = (comparison["AC LF %"] - side_one).dropna() if side_one is not None else pd.Series(dtype=float)
    logger.info(
        "Binding side for %s: %s | vs side ONE alone: %d of %d element(s) understated by up "
        "to %.2f pp, %d element(s) would have had no rating at all",
        kind_label, comparison["Binding Side"].value_counts().to_dict(),
        int((understated > 1e-9).sum()), len(comparison),
        understated.max() if not understated.empty else 0.0,
        int(side_one.isna().sum()) if side_one is not None else len(comparison),
    )


def build_base_case_comparison(limits, hv_lines, hv_transformers, hv_transformers3,
                                ac_ln_i, ac_tr_i, ac_tr3_i,
                                dc_ln_i, dc_tr_i, dc_tr3_i):
    """Build the three base-case (N-0) AC vs DC comparison dataframes (lines, 2W-TR, 3W-TR).

    The ac_*/dc_* frames carry one current (A) per side; each element is then reported on
    its binding side, against that side's own CURRENT PATL. CIM current limits are
    terminal-specific, so a branch can be rated differently on each side - a transformer
    most of all, where the two ratings express the same MVA at different voltages - and
    side ONE is not reliably the side that binds.
    """
    patl = limits[(limits["acceptable_duration"] == -1) & (limits["type"] == "CURRENT")]
    # One column per side; the most restrictive rating wins if a side carries several.
    patl_per_side = patl.pivot_table(
        index="element_id", columns="side", values="value", aggfunc="min",
    )

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

    return (
        lines_cmp, transformers_cmp, transformers3_cmp,
        final_lines_id, final_tr_id, final_tr3_id,
    )
