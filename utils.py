#!/usr/bin/env python
import jax
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions.transforms import AffineTransform

import pandas as pd
import numpy as _np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_analysis.get_SPARC import load_sparc_table


def vel2acc(vel_sq, rad):
    """
    Calculate the gravitational acceleration (g_obs or g_bar).
    In the data, vel is in km/s, and rad is in kpc.
    Returns g_x in m/s².
    """
    return (vel_sq * 1e6) / (rad * 3.086e19)


def transformed_normal(loc, scale):
    """
    Log-normal prior for SPARC mass-to-light ratios.
    """
    return dist.TransformedDistribution( dist.Normal(0, 1), AffineTransform(loc=loc, scale=scale) )


def discrete_monotonic_ll(arr, scale=1.0):
    """
    Normalized likelihood based on pairwise comparisons of neighboring points to enforce monotonicity.
    arr: an array of deterministic values [a_1, a_2, ..., a_n]

    Returns: log-likelihood that penalizes pairwise non-monotonicity for all valid pairs.
    """

    # Only compare neighboring pairs
    differences = arr[:-1] - arr[1:]  # arr[i] - arr[i+1] for i in range(n-1)
    
    probs = jax.nn.sigmoid( differences / scale )
    total_log_prob = jnp.sum(jnp.log(probs))

    return total_log_prob


def get_SBdisk(galaxy:str):
    """
    Extract surface brightness profile of disk component
    from SPARC .dens file for vcdisk calculation.
    
    Parameters
    ----------
    galaxy : str
        Name of the galaxy.
    inc : float
        Inclination in degrees.
    
    Returns
    -------
    rad (kpc), SBdisk (Lsun/kpc^2) : 1D jnp arrays
    """

    file = f"/mnt/users/koe/SPARC_RAR/BulgeDiskDec_LTG/{galaxy}.dens"
    columns = [ "Rad", "SBdisk", "SBbul" ]
    data = pd.read_csv(file, names=columns, skiprows=1, sep="\t")

    rad = jnp.array(data["Rad"])    # in kpc

    # Normalise SBdisk such that total disk luminosity matches L[3.6] in SPARC data (see email from Federico)
    SBdisk_raw = jnp.array(data["SBdisk"]) * 1e6
    Ltot, _ = compute_A_B(rad, SBdisk_raw)

    table = load_sparc_table()
    i_table = _np.where(table["Galaxy"] == galaxy)[0][0]
    L = table["L"][i_table] * 1e9   # in Lsun

    SBdisk = SBdisk_raw * L / Ltot

    return rad, SBdisk


def compute_A_B(R, I_R):
    """
    Compute A = 2π ∫ I(R) R dR,
            B = 2π ∫ I(R) R^2 dR
    for arbitrary nonuniform SPARC radii R.
    Used in linear M/L profile.
    
    Parameters
    ----------
    R   : radii (1D array) in kpc
    I_R : surface brightness (Lsun/kpc^2) at each radius
    
    Returns
    -------
    A, B : scalars
    """

    # geometric factor
    fac = 2 * jnp.pi

    # trapezoidal integrals
    A = fac * jnp.trapezoid(I_R * R, R)
    B = fac * jnp.trapezoid(I_R * R**2, R)

    return A, B

