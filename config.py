"""Run configuration (constants and paths) for the AC/DC load flow comparison study.

This file is a template: every value below is a generic default of the right type.
Real values for your environment go in input/config_local.py (git-ignored, overrides
the defaults below via the import at the bottom of this file). See
input/config_local.py.example for the list of keys and how to fill them in.
"""

# ---------------------------------------------------------------------------
# General run flags
# ---------------------------------------------------------------------------
Sens = 0                # Perform Sensitivity Analysis
SA = 0                   # Perform Security Analysis
cgmes_zip = ""
xiidm_file = 0            # 0 - loads the cgmes_zip file, saves xiidm_file with same name and location, 1-loads xiidm file with same name and location
CON_OPT = 0               # 0-Use only NCC con from json_file, 1- Use both NCC and RCC con, 2- Use only RCC con from json_file_rcc

# CONTINGENCY FILE (To Be Modified to CO XML)
json_file = ""
json_file_rcc = ""

# ---------------------------------
# --------RAO PART-----------------
RAO_RUN = 0    # Perform OpenRAO
CRAC_BUFF = 0  # 1-CRAC from buffer, dont write it as json (so no read/write time)
CON_FCNEC = 1  # Consider contingencies in flowCNEC definition, 0 means only base case flows are considered

# Randomly select MGE and CON to test NCC, RCC, NCC+RCC performance
RANDOM_SEL = 0   # SELECT RANDOMLY - 1
rNoCON = 800     # NO of random con
rNoRCC = 1500    # NO of random mon for RCC
rNoNCC = 1500    # NO of random opt for NCC
rand_seed = 0    # 0 for no seed
NCC_MON = False  # Should NCC Circuits be monitored?
NCC_OPT = True   # Should NCC Circuits be optimized?
RCC_MON = True   # Should RCC Circuits be monitored?
RCC_OPT = False  # Should RCC Circuits be optimized?
mod_crac = 0     # modified crac to change CBCO applied only to 0 injectionRA item, test only, keep this 0
mod_crac_cnec_id = ""  # cnecId used by the mod_crac debug patch, only used if mod_crac is truthy
save_result = 1  # save rao result as json file
SA_Thres = 80    # % threshold for using CBCO from AC - SA
input_crac = ""
# input_crac is only used if CRAC_BUFF=2
inputs_RA = ""  # This will have all the Remedial Action information, only used in CRAC_BUFF!=2
logging_PSB = 0  # Enable debug logging infor for pypowsybl

ignore_mge_list = []

# CGMES import parameters (used only when xiidm_file == 0)
cgmes_import_parameters = {
    "iidm.import.cgmes.create-busbar-section-for-every-connectivity-node": "true",
    "iidm.import.cgmes.cgm-with-subnetworks": "false",
    "iidm.import.cgmes.source-for-iidm-id": "mRID",
}

# HV / RCC voltage-level classification thresholds (kV), used by network_io.get_network_items
# 225 kV (FR) and 400 kV are transmission levels that appear in the KPI workbook's own
# VoltageLevel dimension; without them here those elements never enter the study at all.
# RCC_VOLTAGE_LEVELS_KV deliberately does NOT get them: RCC is the lower-voltage regional
# set, and the HV/RCC split below relies on the two lists staying disjoint at the top end.
HV_VOLTAGE_LEVELS_KV = [220.0, 225.0, 380.0, 400.0, 150.0]
HV_TRAFO_VOLTAGE_LEVELS_KV = [220.0, 225.0, 380.0, 400.0, 150.0, 110.0]
RCC_VOLTAGE_LEVELS_KV = [150.0, 110.0, 70.0]

# Loading threshold below which a branch is excluded from the base-case comparison
BASE_CASE_ACTIVE_THRESHOLD_PCT = 50

# Loading threshold below which a branch is excluded from the SA comparison
SA_ACTIVE_THRESHOLD_PCT = 50

# ---------------------------------------------------------------------------
# KPI workbook (output/Core_AC_DC_KPI.xlsx) - calculation parameters
#
# These are the 7 parameters the workbook's own `Parameters` tab reports, plus a
# fallback country code. Every one is read at run time and written back out after the
# run: `Parameters` is a report of what this run used, never an input.
#
# Naming note: three unrelated loading thresholds already live in this file -
# BASE_CASE_ACTIVE_THRESHOLD_PCT and SA_ACTIVE_THRESHOLD_PCT (population filters, 50)
# and SA_Thres (the RAO CNEC shortlist cutoff, 80). None of them is the near-limit KPI
# threshold. The KPI_ prefix below keeps the two families apart.
# ---------------------------------------------------------------------------

# An element is in violation when its loading reaches this value (>=, not >).
KPI_VIOLATION_THRESHOLD_PCT = 100

# The near-limit KPIs look only at elements whose AC loading reaches this value (>=).
# Distinct from the violation threshold above: this one selects a population, that one
# decides pass/fail. NOT to be confused with SA_Thres.
KPI_NEAR_LIMIT_THRESHOLD_PCT = 90

# N for the Top-N critical element overlap KPI (K16) and the TopN_N column.
KPI_TOP_N = 5

# Reported in `Parameters` only. This build runs a single snapshot: one business day,
# one timestamp. They size the BusinessDay / Timestamp dimensions in a multi-run build
# that does not exist yet, so they are report-only values here.
KPI_BUSINESS_DAYS_SIMULATED = 1
KPI_TIMESTAMPS_PER_BUSINESS_DAY = 1

# Reported in `Parameters` only. Perf_Computation's N_Contingencies reports the number
# of contingencies the security analysis actually ran, not this value.
KPI_CONTINGENCIES_PER_COUNTRY_VOLTAGE_LEVEL = 1

# Display / labelling only; every flow figure in the workbook is a current in A.
KPI_FLOW_UNIT = "A"

# Country code used when an element's substation carries no country. CGMES derives the
# country from the GeographicalRegion, which lives in the boundary (EQBD) file - a model
# imported without one resolves every country to null, as the Belgovia sample does.
KPI_FALLBACK_COUNTRY = "UNKNOWN"

# Optional path to Mahir's Dim_KPI catalog (.xlsx or .csv). When set, its content is
# embedded into the workbook's Dim_KPI tab unchanged and drives every number format.
# When empty, rosc_acdc/dim_kpi.py's reconstructed catalog is used instead.
KPI_DIM_KPI_FILE = ""

# ---------------------------------------------------------------------------
# KPI workbook - Excel output settings
# ---------------------------------------------------------------------------

# Output folder for every generated artifact, relative to where the tool runs from.
KPI_OUTPUT_DIR = "output"

# Fixed filename, overwritten on every run: no versioning, no timestamp, no tagging.
KPI_WORKBOOK_FILENAME = "Core_AC_DC_KPI.xlsx"

# The pre-existing per-contingency detail workbook. Written alongside the one above,
# not replaced by it: the two carry different granularities.
LEGACY_KPI_WORKBOOK_FILENAME = "kpi_results.xlsx"

# ---------------------------------------------------------------------------
# Local overrides (git-ignored, real values for this environment)
# ---------------------------------------------------------------------------
try:
    from input.config_local import *  # noqa: F401,F403
except ImportError:
    pass
