
def radial_plotting_function(ydata, xdata, data_labels, xlabel, ylabel, temp_split=False, colors=['blue', 'lime'], Tcolors=['r', 'orange', 'b', 'aqua'], 
                             xscale='log', yscale='log', r_vir=None, r_vir_label=True, xlim=None, ylim=None, figsize=(8,6), save_name=None, save_dir=None, 
                             legend=False, v_k = None, r_soft1=None, r_soft2=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    plt.rcParams.update({'font.size': 22})
    plt.rcParams['axes.linewidth'] = 1.25
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.major.width'] = 1.25
    fig, ax1 = plt.subplots(figsize=figsize)
    plt.tight_layout(pad=3)
    if temp_split:
        for i, data in enumerate(ydata):
            ax1.plot(xdata, data, linestyle='-', color=Tcolors[i], linewidth=3, label=data_labels[i])
    else:
        for i, data in enumerate(ydata):
            ax1.plot(xdata, data, linestyle='-', color=colors[i], linewidth=3, label=data_labels[i])
    ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
    if v_k is not None:
        ax1.plot(xdata, v_k, linestyle='-', color='dimgrey', linewidth=2, label='v$_{\mathrm{K}}$')
    if r_soft1 is not None:
        ax1.axvline(r_soft1, color='k', linestyle=':', linewidth=2, label='r$_{\mathrm{soft}}$')
    ax1.set_xlabel(xlabel)
    ax1.set_xscale(xscale)
    if r_vir_label:
        ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
    else:
        ax1.set_xticks([1e3, 1e4], labels=['$10^3$', '$10^4$'])
    if xlim is None:
        ax1.set_xlim(r_soft2, xdata[-1])
    else:
        ax1.set_xlim(xlim)
    ax1.minorticks_on()
    ax1.tick_params(which='minor', length=3, width=1)
    ax1.tick_params(which='both', direction='in')
    ax1.tick_params(left=True, right=True)
    ax1.set_ylabel(ylabel)
    ax1.set_yscale(yscale)
    if ylim is not None:
        ax1.set_ylim(ylim)
    if legend:
        ax1.legend(frameon = False, fontsize=12, loc='lower left')
    if save_name is not None:
        os.chdir(save_dir)
        plt.savefig(save_name)
    plt.close()



