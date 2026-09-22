"""AC vs DC load flow / contingency-analysis comparison study - entry point.
"""

import logging

import json
import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import pypowsybl as pp
import seaborn as sns

from rosc_acdc import (
    config,
    contingencies,
    kpi_workbook,
    kpis,
    loadflow,
    logging_setup,
    network_io,
    plotting,
    security_analysis,
    sensitivity,
)

logger = logging.getLogger(__name__)


def main():
    log_path = logging_setup.configure_logging()
    logger.info("Logging to %s", log_path)
    
    network = network_io.load_network()

    # --- AC load flow ---
    ac_result, ac_lf_time = loadflow.run_ac(network)

    buses = network.get_buses().fillna(0)
    lines = network.get_lines().fillna(0)
    transformers = network.get_2_windings_transformers().fillna(0)
    transformers3 = network.get_3_windings_transformers().fillna(0)

    hv_lines, rcc_lines, hv_transformers, rcc_transformers, hv_transformers3, rcc_transformers3 = (
        network_io.get_network_items(network)
    )

    # Base-case flows are currents in A per side, to match the CURRENT PATL and the SA dataset.
    ac_ln_i = loadflow.branch_current(hv_lines)
    ac_tr_i = loadflow.branch_current(hv_transformers)
    ac_tr3_i = loadflow.branch_current(hv_transformers3)

    # The RCC elements never enter the base-case comparison, but they do enter the security
    # analysis, and the KPI workbook has to attribute every element it reports to one
    # country and voltage level. Captured here, before the DC load flow overwrites the
    # network's AC current results.
    ac_all_i = pd.concat([
        ac_ln_i, ac_tr_i, ac_tr3_i,
        loadflow.branch_current(rcc_lines),
        loadflow.branch_current(rcc_transformers),
        loadflow.branch_current(rcc_transformers3),
    ])

    # --- DC load flow ---
    dc_result, dc_lf_time = loadflow.run_dc(network)

    buses = network.get_buses().fillna(0)
    hv_lines, rcc_lines, hv_transformers, rcc_transformers, hv_transformers3, rcc_transformers3 = (
        network_io.get_network_items(network)
    )
    generators = network.get_generators().fillna(0)
    ptcs = network.get_phase_tap_changers().fillna(0)

    voltage_levels = network.get_voltage_levels()
    dc_ln_i = loadflow.branch_current_dc(hv_lines, voltage_levels)
    dc_tr_i = loadflow.branch_current_dc(hv_transformers, voltage_levels)
    dc_tr3_i = loadflow.branch_current_dc(hv_transformers3, voltage_levels)

    limits = network.get_loading_limits().reset_index()

    # Each element is attributed to a single side for the KPI workbook's Country and
    # VoltageLevel grouping keys - the side the base case already reports it on. The SA
    # dataset keeps both sides as separate observations, which is correct and unchanged;
    # this only decides which country and voltage level those observations are filed under,
    # so a tie line does not land in two countries depending on the row.
    binding_side = loadflow.binding_sides(ac_all_i, loadflow.permanent_current_limits(limits))

    (
        lines_cmp, transformers_cmp, transformers3_cmp,
        final_lines_id, final_tr_id, final_tr3_id,
        single_sided_elements,
    ) = loadflow.build_base_case_comparison(
        limits, hv_lines, hv_transformers, hv_transformers3,
        ac_ln_i, ac_tr_i, ac_tr3_i,
        dc_ln_i, dc_tr_i, dc_tr3_i,
    )

    base_case_df = pd.concat([lines_cmp, transformers_cmp, transformers3_cmp])
    base_case_comparison = kpis.prepare_comparison(
        base_case_df, ac_value_col="AC LF", dc_value_col="DC LF", limit_col="PATL",
        ac_loading_col="AC LF %", dc_loading_col="DC LF %",
    )
    # Every KPI result is collected here and written to one workbook at the end of the run.
    kpi_rows = kpis.log_all_priority1_kpis(
        "Base Case (N-0)", base_case_comparison, id_col=pd.Series(base_case_df.index, index=base_case_df.index),
        side_col=base_case_df["Binding Side"],
    )
    # Workbook only: a note on the input ratings, not a KPI, so it is not logged.
    kpi_rows += kpis.overall_rows("Base Case (N-0)", {
        "Elements Rated on One Side Only": len(single_sided_elements),
    })

    # --- Sensitivity analysis (optional) ---
    if config.Sens:
        sensitivity.run_sensitivity_analyses(network, final_lines_id, generators, ptcs)

    # --- Contingency scenarios ---
    data = contingencies.load_contingency_data()
    valid_ids = contingencies.build_valid_ids(network)

    sa = pp.security.create_analysis()
    branch_ids = (
        hv_lines.index.tolist() + hv_transformers.index.tolist() +
        rcc_lines.index.tolist() + rcc_transformers.index.tolist()
    )
    tr3_ids = hv_transformers3.index.tolist() + rcc_transformers3.index.tolist()

    element_info = pd.concat([
        hv_lines, hv_transformers, hv_transformers3, rcc_lines, rcc_transformers, rcc_transformers3,
    ])
    element_info.index.name = "subject_id"

    sa.add_monitored_elements(branch_ids=branch_ids, three_windings_transformer_ids=tr3_ids)
    # The contingency count is Perf_Computation's N_Contingencies: how many contingencies
    # the security analysis actually ran, not how many the scenario file holds.
    _missing, contingency_count = contingencies.add_contingencies_and_actions(sa, data, valid_ids)

    shortlist_con_mge = None
    sides_ac_all = None
    sa_comparison = None
    patl_all = None
    con_analysis = ac_con_analysis = None

    if config.SA:
        result_dc, result_ac, con_analysis, ac_con_analysis = security_analysis.run_security_analysis(sa, network)
        sides_ac = security_analysis.build_side_comparison(result_dc, result_ac, lines, transformers)
        shortlist_con_mge, sides_ac_all, patl_all = security_analysis.build_shortlist_and_full_comparison(
            limits, sides_ac, network, element_info,
        )

        sa_comparison = kpis.prepare_comparison(
            sides_ac_all, ac_value_col="i", dc_value_col="i_DC", limit_col="patl",
            ac_loading_col="loading_pct",
        )
        kpi_rows += kpis.log_all_priority1_kpis(
            "Contingency (SA)", sa_comparison,
            id_col=sides_ac_all["subject_id"], group_col=sides_ac_all["contingency_id"],
            name_col=sides_ac_all["subject_name"], side_col=sides_ac_all["side"],
        )

        performance = kpis.performance_kpis(
            ac_lf_time, dc_lf_time, ac_sa_time=ac_con_analysis, dc_sa_time=con_analysis,
        )

        plotting.plot_sa_comparison(sides_ac_all)
    else:
        performance = kpis.performance_kpis(ac_lf_time, dc_lf_time)

    kpis.log_kpi_table("Performance", performance)
    kpi_rows += kpis.overall_rows("Performance", performance)
    kpis.write_kpi_workbook(kpi_rows)

    # --- KPI workbook (Core_AC_DC_KPI.xlsx) ---
    # A second, differently-grained output written alongside kpi_results.xlsx above, not a
    # replacement for it: this one reports N-0 + all-COs-combined per country and voltage
    # level, that one keeps the full per-contingency series.
    stage_times = [
        ("AC Load Flow", "Load Flow", "AC", ac_lf_time),
        ("DC Load Flow", "Load Flow", "DC", dc_lf_time),
        ("AC Security Analysis", "Security Analysis", "AC", ac_con_analysis),
        ("DC Security Analysis", "Security Analysis", "DC", con_analysis),
    ]
    kpi_workbook.build_and_write(
        network,
        network_io.element_locations(network, element_info, binding_side),
        stage_times,
        n_elements_evaluated=len(branch_ids) + len(tr3_ids),
        n_contingencies=contingency_count,
        base_case_comparison=base_case_comparison,
        base_case_ids=pd.Series(base_case_df.index, index=base_case_df.index),
        sa_comparison=sa_comparison,
        sa_element_ids=None if sides_ac_all is None else sides_ac_all["subject_id"],
        sa_contingency_ids=None if sides_ac_all is None else sides_ac_all["contingency_id"],
    )

    if not config.RAO_RUN:
        return

    # --- RAO ---
    from rosc_acdc.rao import crac_builder, runner

    topo_actions = crac_builder.load_topo_actions(network)
    redispatch_actions = crac_builder.load_redispatch_actions(network)
    pst_actions = crac_builder.load_pst_actions(network)

    if config.SA:
        rcc_pool = pd.concat([rcc_lines, rcc_transformers], axis=0)
        ncc_pool = pd.concat([hv_lines, hv_transformers], axis=0)
        rcc_line_ids = rcc_pool.loc[rcc_pool.index.isin(shortlist_con_mge["subject_id"])]
        ncc_line_ids = ncc_pool.loc[ncc_pool.index.isin(shortlist_con_mge["subject_id"])]
    else:
        rcc_line_ids = rcc_lines
        ncc_line_ids = hv_lines

    if config.RANDOM_SEL:
        if config.rand_seed:
            rcc_line_ids = rcc_line_ids.sample(n=config.rNoRCC, random_state=config.rand_seed)
            ncc_line_ids = ncc_line_ids.sample(n=config.rNoNCC, random_state=config.rand_seed)
        else:
            rcc_line_ids = rcc_line_ids.sample(n=config.rNoRCC)
            ncc_line_ids = ncc_line_ids.sample(n=config.rNoNCC)

    if config.SA:
        shortlist_con_mge = shortlist_con_mge.copy()
        shortlist_con_mge["CC"] = pd.NA
        shortlist_con_mge.loc[shortlist_con_mge["subject_id"].isin(ncc_line_ids.index), "CC"] = "NCC"
        shortlist_con_mge.loc[shortlist_con_mge["subject_id"].isin(rcc_line_ids.index), "CC"] = "RCC"

    crac = crac_builder.build_crac(
        network, data, valid_ids, shortlist_con_mge, rcc_line_ids, ncc_line_ids,
        patl_all, topo_actions, redispatch_actions, pst_actions,
    )

    rao_crac = runner.load_crac(network, crac)
    result, rao_time = runner.run_rao(network, rao_crac, len(crac["flowCnecs"]), ac_con_analysis)

    if config.save_result:
        runner.save_and_process_result(result)


if __name__ == "__main__":
    main()
