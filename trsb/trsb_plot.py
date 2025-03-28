"""
Plot the two TRSB models side by side.
Need to run trsb1 & trsb2 for successful plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

# Set global font sizes
mpl.rcParams.update({
    'font.size': 12,              
    'axes.titlesize': 14,         
    'axes.labelsize': 14,        
    'xtick.labelsize': 14,       
    'ytick.labelsize': 14,       
    'legend.fontsize': 12,       
    'figure.titlesize': 14,      
})

# Load saved data
trsb1 = np.load("trsb1_data.npz")
trsb2 = np.load("trsb2_data.npz")

datasets = [trsb1, trsb2]
titles = ['TRSB-1', 'TRSB-2']

# Set colour scale
divnorm = mcolors.TwoSlopeNorm(vmin=-200, vcenter=0, vmax=200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), sharey=True)

for ax, data, title in zip([ax1, ax2], datasets, titles):
    energies = data['energies']
    berry_curvs = data['berry_curvs']
    k_index = data['k_index']
    chern = data['chern']
    hs_idx = data['high_symmetry_indices']

    for band in range(energies.shape[1]):
        ax.plot(k_index, energies[:, band], 'k-', lw=1)
        ax.scatter(
            k_index,
            energies[:, band],
            c=berry_curvs[:, band],
            cmap='RdBu_r',
            norm=divnorm,
            s=5,
            alpha=1
        )

    # Label only bands 5&6, starting from the bottom
    bands_to_label = [4, 5]  
    label_x_pos = int(0.9 * len(k_index))

    for band_idx in bands_to_label:
        y_val = energies[label_x_pos, band_idx]
        band_chern = chern[band_idx]
        
        ax.text(label_x_pos*9/10,
                y_val,
                f"C={band_chern}",
                fontsize=12,
                color='black',
                va='center',
                ha='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
        )

    # x-ticks and symmetry lines
    for d in hs_idx:
        ax.axvline(x=d, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xticks(hs_idx)
    ax.set_xticklabels(['Γ', r'$\mathrm{M_1}$', r'$\mathrm{K_1}$', r'$\mathrm{M_2}$', r'$\mathrm{K_2}$', 'Γ'])
    ax.set_yticks([-0.4, 0, 0.4])
    ax.set_title(title)

# y-axis label only on left plot
ax1.set_ylabel("Energy (eV)")

# Set same limits across both plots
ax1.set_xlim(0, trsb1['k_index'][-1])
ax2.set_xlim(0, trsb2['k_index'][-1])
ax1.set_ylim(-0.4, 0.5)

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=divnorm, cmap='RdBu_r'), cax=cbar_ax)
cbar.ax.set_title(r'$\Omega_n(k)$', fontsize=14, pad=10)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("TRSB_comparison.png", dpi=600)
plt.show()
