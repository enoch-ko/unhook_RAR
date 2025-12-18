#!/usr/bin/env python
# Attempt to remove hooks from the SPARC RAR.
import numpy as np
import pandas as pd
from jax import numpy as jnp

from numpyro import distributions as dist
from numpyro import deterministic, sample, factor


import sys
import os
from utils import vel2acc, transformed_normal, discrete_monotonic_ll

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_analysis.params import pdisk


def model(data, A, B, profile:str, priors:str, pdisk:float=pdisk):
    """
    MCMC model for fitting linear M/L profile to SPARC RAR data.

    Parameters
    ----------
    data : pd.DataFrame
        SPARC data for a single galaxy.
    A, B : scalars
        Integrals over surface brightness profile.
    priors : str
        Type of priors to use ("uniform" or "gaussian").
    pdisk : float
        Central disk mass-to-light ratio (M/L_disk) to use for Gaussian prior.

    Returns
    -------
    None
    """

    r = deterministic("r", jnp.array(data["Rad"]))
    b = sample("b", dist.Uniform(-5.0, 5.0))    # slope of linear M/L profile (always uniform prior)

    # Sample log mass-to-light ratios.
    if priors == "uniform":
        log_pgas = sample("Gas M/L", dist.Uniform(jnp.log10(1.)-0.4, jnp.log10(1.)+0.4))
        if profile == "free":
            log_pdisk = sample("Rad M/L", dist.Uniform(jnp.log10(pdisk)-1.0, jnp.log10(pdisk)+1.0, shape=(len(r),)))
        else:
            log_pdisk = sample("Disk M/L", dist.Uniform(jnp.log10(pdisk)-1.0, jnp.log10(pdisk)+1.0))
    elif priors == "gaussian":
        log_pgas = sample("Gas M/L", transformed_normal(jnp.log10(1.), 0.04))
        if profile == "free":
            log_pdisk = sample("Rad M/L", dist.Normal(jnp.log10(pdisk), 0.1, shape=(len(r),)))
        else:
            log_pdisk = sample("Disk M/L", transformed_normal(jnp.log10(pdisk), 0.1))
    else:
        raise ValueError("Invalid prior choice. The only valid choices are 'uniform' or 'gaussian'.")

    Upsilon_gas, smp_pdisk = 10**log_pgas, 10**log_pdisk

    # Impose M/L profile.
    if profile == "constant" or profile == "free":
        Upsilon_disk = smp_pdisk
    elif profile == "linear":
        a = deterministic("a", smp_pdisk - b * (B / A) )
        Upsilon_disk = a + b * r
    else:
        raise ValueError("Invalid profile choice. The only valid choices are 'constant', 'linear', or 'free'.")
    
    Vbar_squared = (jnp.array(data["Vgas"]**2) * Upsilon_gas + jnp.array(data["Vdisk"]**2) * Upsilon_disk)
    deterministic("Vbar", jnp.sqrt(Vbar_squared))

    # Sample parameters and calculate the predicted g_obs.
    g_bar = deterministic("g_bar", vel2acc(Vbar_squared, r))

    # Likelihood: monotonicity constraint on gbar.
    log_gbar = jnp.log10(g_bar)
    ll_gbar = discrete_monotonic_ll(log_gbar)

    ll = deterministic("log_likelihood", ll_gbar)

    factor("ll", ll)
