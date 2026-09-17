# MCMC-based "Unhooking" of SPARC RARs

Purpose: Uses Bayesian MCMC (via NumPyro/NUTS) to fit mass-to-light (M/L) ratios for galaxies
showing "upward hooks" in their Radial Acceleration Relation (RAR), attempting to "unhook"
them by fitting for M/L profiles.

Key Components:
- unhook_RAR.py — Main script running MCMC on 8 "upward hooked" galaxies (DDO170, NGC4100,
NGC0024, NGC5585, NGC0247, UGC02259, NGC3877, UGC04325). Fits M/L ratios with configurable
priors (uniform/gaussian) and M/L profiles (constant/linear/free). Uses a monotonicity 
likelihood on g_bar to enforce physically plausible acceleration curves.
- model.py — The NumPyro probabilistic model: samples Gas M/L, Disk M/L (with profile),
computes Vbar, g_bar, and applies a discrete monotonicity log-likelihood.
- utils.py — Helper functions: vel2acc (conversion), transformed_normal (log-normal priors),
discrete_monotonic_ll (Thurstone-Mosteller pairwise likelihood), get_SBdisk (loads/normalizes
surface brightness from .dens files), compute_A_B (integrals for linear M/L profiles).

Output: Corner plots comparing prior vs posterior for each galaxy, plus 2×4 grid comparison
plots of g_bar and RAR before/after fitting. Results saved to
/mnt/users/koe/unhook_RAR/plots/{profile}_ML/{priors}_priors/.
