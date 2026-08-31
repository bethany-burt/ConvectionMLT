# =============================================================================
# Configuration file: dry H2/H2–He radiative–convective equilibrium (RCE)
#
# Edit the assignments below, then run:
#   cd convection_mlt
#   PYTHONPATH=src python examples/rce/run_rce.py --config examples/rce/cfg_demo.py
#
# Flux sign convention (read once):
#   f_int  — net upward internal flux imposed at the bottom boundary [W m^-2]
#   f_irr  — downward irradiation at the top boundary [W m^-2]
#   Profile fluxes F_rad, F_conv, F_total are positive upward.
#
# Physical convergence gate (1e-3) is fixed by the solver and is NOT set here.
# =============================================================================

# ====== Atmosphere & thermodynamics ======
n_layers = 100          # vertical layers; any integer >= 4 (96/192/384 = nested spatial gate)
p_bottom = 10.0          # bottom pressure [bar]
p_top = 1.0e-5           # top pressure [bar]
gravity = 15.0           # surface gravity [m s^-2]
x_he = 0.3               # helium mole fraction in [0, 1]; 0 = pure H2

# ====== Boundary fluxes ======
f_int = 300.0            # internal heat flux at bottom [W m^-2], upward positive
f_irr = 1500.0              # stellar irradiation at top [W m^-2], downward positive

# ====== Convection & opacity ======
alpha = 1.0              # mixing-length parameter (dimensionless)
seed = 'radiative_convective'
# seed options:
#   'radiative_convective' — grey RC profile (recommended default)
#   'radiative_equilibrium' — pure radiative equilibrium seed
opacity_model = 'analytic_grey_powerlaw'   # only supported model in v1

# ====== Solver (computational) ======
procedure = 'production'
# procedure options:
#   'production'     — discrete-RZ accelerator + live polish + recovery (default)
#   'adaptive_only'  — skip discrete-RZ; adaptive integrator only

max_steps_live_polish = 200
max_steps_continuation = 500
max_recovery_cycles = 2
dt_accuracy_s = 50000.0           # pseudo-time accuracy ceiling [s]
dt_hold_init_s = 18415.0          # initial held step during polish [s]
continuation_dt_accuracy_s = 2500.0 # continuation phase step ceiling [s]
prescribed_dt_s = None            # set a float (e.g. 2500.0) for physical Δt; None = pseudo-time
max_steps_adaptive_only = 20000   # used only when procedure = 'adaptive_only'

# ====== Output paths ======
output_dir = 'examples/rce/runs'
out_name = 'firr1500_alpha1_n100'      # run subfolder under output_dir (optional label)
overwrite = False                 # True: replace a non-empty run directory

write_profiles = True             # profiles_centres.csv, profiles_interfaces.csv, profiles.npz
write_convergence = True          # convergence.csv
write_result_json = True          # result.json (full serialized state)
write_status = True               # status.json (verdict and gate summary)
write_figure = True               # figure_summary.png (see panel toggles below)

# ====== Plotting ======
figure_dpi = 150
plot_temperature = True           # T–P panel (initial vs final)
plot_fluxes = True                # F_rad, F_conv, F_total vs P
plot_gradients = True             # nabla, nabla_ad, Delta nabla vs P
plot_convergence = True           # flux flatness and tendency vs step
