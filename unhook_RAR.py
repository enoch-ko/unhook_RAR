#!/usr/bin/env python
# Attempt to remove hooks from the SPARC RAR.
import jax
from jax import numpy as jnp
import jax.random as random
import numpy as np

import numpyro
from numpyro.infer import Predictive, MCMC, NUTS, init_to_median
import corner
import matplotlib.pyplot as plt

import argparse
from utils import vel2acc, get_SBdisk, compute_A_B
from model import model

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_analysis.get_SPARC import load_sparc_table, get_SPARC_data


def set_args():
    assert numpyro.__version__.startswith("0.15.0")
    numpyro.enable_x64()
    parser = argparse.ArgumentParser(description="MCMC for unhooking RARs")
    parser.add_argument("--testing", default=False, type=bool, help="Testing; runs first two galaxies and shows MCMC progress bar.")
    parser.add_argument("--hooks", default="up", type=str, help="Whether to fit 'up' or 'down' hooks.")
    parser.add_argument("--profile", type=str, help="Profile for M/L_disk ('constant', 'linear', or 'free').")
    parser.add_argument("--priors", type=str, help="Type of priors to use ('uniform' or 'gaussian').")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = set_args()
    testing = args.testing

    full_param_fit = False  # Whether to fit all parameters or just M/L ratios.
    priors = args.priors    # Whether to use Gaussian priors or uniform (wide) priors.
    profile = args.profile  # M/L profile to use: 'constant', 'linear', or 'free'.

    if profile not in ["constant", "linear", "free"]:
        raise ValueError(f"Invalid profile choice ('{profile}'); valid options are 'constant', 'linear', or 'free'.")
    elif priors not in ["uniform", "gaussian"]:
        raise ValueError(f"Invalid choice of priors ('{priors}'); valid options are 'uniform' or 'gaussian'.")
    else:
        fileloc = f"/mnt/users/koe/unhook_RAR/plots/{profile}_ML/{priors}_priors/"

    SPARC_data, _, _ = get_SPARC_data()
    galaxies = [
        "DDO170", "NGC4100", "NGC0024", "NGC5585",
        "NGC0247", "UGC02259", "NGC3877", "UGC04325"
        ]   # 8 'upward hooked' galaxies from https://arxiv.org/pdf/2307.09507
    if testing: galaxies = galaxies[:1]

    unhooked_RAR = {}
    table = load_sparc_table()

    for i, gal in enumerate(galaxies):
        print(f"\nProcessing galaxy {gal} ({i+1}/{len(galaxies)})...")
        i_table = np.where(table["Galaxy"] == gal)[0][0]
        r = jnp.array(SPARC_data[gal]["r"])
        data = SPARC_data[gal]["data"]

        rad, SBdisk_dens = get_SBdisk(gal)
        A, B = compute_A_B(rad, SBdisk_dens)
        SBdisk = jnp.interp(r, rad, SBdisk_dens)

        nuts_kernel = NUTS(model, init_strategy=init_to_median(num_samples=2000))
        mcmc = MCMC(nuts_kernel, num_warmup=20000, num_samples=40000, progress_bar=testing)
        mcmc.run(random.PRNGKey(0), data, A, B, profile=profile, priors=priors)
        mcmc.print_summary()
        samples = mcmc.get_samples()

        prior_predictive = Predictive(model, num_samples=40000)
        prior_samples = prior_predictive(random.PRNGKey(0), data, A, B, profile=profile, priors=priors)

        min_smp_ll, max_smp_ll = jnp.min( samples["log_likelihood"] ), jnp.max( samples["log_likelihood"] )
        min_ll = min_smp_ll - max( ( max_smp_ll - min_smp_ll ) / 2, abs(min_smp_ll) / 10 )
        max_ll = max_smp_ll + max( ( max_smp_ll - min_smp_ll ) / 2, abs(min_smp_ll) / 10 )

        corner_samples = {
            "Gas M/L (log)": samples["Gas M/L"],
            "Disk M/L (log)": samples["Disk M/L"],
            "a": samples["a"],
            "b": samples["b"],
            "LL (all)": samples["log_likelihood"],
        }
        corner_priors = {
            "Gas M/L (log)": prior_samples["Gas M/L"],
            "Disk M/L (log)": prior_samples["Disk M/L"],
            "a": prior_samples["a"],
            "b": prior_samples["b"],
            "LL (all)": prior_samples["log_likelihood"],
        }
        range = [ 1., 1., 1., (-5.0, 5.0), (min_ll, max_ll) ]

        figure = corner.corner(np.column_stack([np.array(corner_priors[key]) for key in corner_priors]),
                               color="tab:blue", bins=40, range=range)
        sample_array = np.column_stack([np.array(corner_samples[key]) for key in corner_samples])
        corner.corner(sample_array, labels=list(corner_samples.keys()), show_titles=True,
                      color="tab:red", bins=40, fig=figure, range=range)
        

        figure.savefig(f"{fileloc}{gal}.png", dpi=300)
        plt.close(figure)
        print("Corner plots saved.")

        # Create individual plots to compare RARs before and after fits.
        log_likelihood = samples["log_likelihood"]
        max_ll_idx = jnp.argmax(log_likelihood)
        g_bar = samples["g_bar"][max_ll_idx]

        g_obs = vel2acc( jnp.array(data["Vobs"])**2, r )

        unhooked_RAR[gal] = {
            'r': r,
            'g_bar': g_bar,
            'g_obs': g_obs,
        }

        jax.clear_caches()


    ### Extra comparison plots for all galaxies. ###

    # 2x4 grid of g_bar comparison plots.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, gal in enumerate(galaxies):
        ax = axes[idx]
        ax.plot(unhooked_RAR[gal]["r"], SPARC_data[gal]["g_bar"], marker='.', color='k', alpha=0.5, label='data')
        ax.plot(unhooked_RAR[gal]["r"], unhooked_RAR[gal]["g_bar"], marker='.', color='tab:red', label='max LL')
        ax.set_yscale('log')
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('g_bar [m/s²]')
        ax.set_title(f"{gal} g_bar comparison")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{fileloc}g_bar.png", dpi=300)
    plt.close()

    print("g_bar comparison plot saved.")


    # 2x4 grid of RAR comparison plots.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, gal in enumerate(galaxies):
        ax = axes[idx]
        ax.plot(SPARC_data[gal]["g_bar"], SPARC_data[gal]["g_obs"], marker='.', color='k', alpha=0.5, label='data')
        ax.plot(unhooked_RAR[gal]["g_bar"], unhooked_RAR[gal]["g_obs"], marker='.', color='tab:red', label='max LL')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('g_bar [m/s²]')
        ax.set_ylabel('g_obs [m/s²]')
        ax.set_title(f"{gal} RAR (log-log)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{fileloc}RAR.png", dpi=300)
    plt.close()

    print("RAR comparison plot saved.")
    print(f"\nUnhooking complete with *{profile}* profile and *{priors}* priors. Good luck! :)")
