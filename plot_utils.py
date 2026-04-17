import numpy as np
import matplotlib.pyplot as plt


def plot_vol_trajectories_grid(
    vol_cube: np.ndarray,
    *,
    t: np.ndarray | None = None,
    tenors: np.ndarray | None = None,
    log_moneyness: np.ndarray | None = None,
    stride_tenor: int = 1,
    stride_m: int = 1,
    max_panels: int = 120,
    sharex: bool = True,
    sharey: bool = True,
    figsize_per_ax: tuple[float, float] = (3.2, 2.2),
    suptitle: str | None = "Implied vol trajectories",
    show: bool = True,
):
    """
    Visualize vol trajectories (time series) on a tenor x log-moneyness grid.

    Parameters
    ----------
    vol_cube : array
        Shape (n_steps, J, M) or (n_steps+1, J, M).
    t : array, optional
        Shape (n_steps,) matching first dim of vol_cube. If None, uses np.arange(n_steps).
    tenors : array, optional
        Shape (J,) for labeling.
    log_moneyness : array, optional
        Shape (M,) for labeling.
    stride_tenor, stride_m : int
        Plot every k-th tenor / moneyness to keep the grid readable.
    max_panels : int
        If the grid becomes too large, it will be split into multiple figures (pages).
    figsize_per_ax : tuple
        Figure size scales with number of subplots: (w_per_ax, h_per_ax).
    """
    vol_cube = np.asarray(vol_cube, dtype=float)
    if vol_cube.ndim != 3:
        raise ValueError(f"vol_cube must be 3D (n_steps, J, M). Got shape={vol_cube.shape}")

    n_steps, J, M = vol_cube.shape

    if t is None:
        t = np.arange(n_steps, dtype=float)
    else:
        t = np.asarray(t, dtype=float)
        if t.shape[0] != n_steps:
            raise ValueError(f"t must have length {n_steps}, got {t.shape[0]}")

    if tenors is None:
        tenors = np.arange(J, dtype=float)
    else:
        tenors = np.asarray(tenors, dtype=float)
        if tenors.shape[0] != J:
            raise ValueError(f"tenors must have length {J}, got {tenors.shape[0]}")

    if log_moneyness is None:
        log_moneyness = np.arange(M, dtype=float)
    else:
        log_moneyness = np.asarray(log_moneyness, dtype=float)
        if log_moneyness.shape[0] != M:
            raise ValueError(f"log_moneyness must have length {M}, got {log_moneyness.shape[0]}")

    tenor_idx = np.arange(0, J, stride_tenor, dtype=int)
    m_idx = np.arange(0, M, stride_m, dtype=int)

    JJ, MM = len(tenor_idx), len(m_idx)
    total_panels = JJ * MM
    if total_panels == 0:
        raise ValueError("Nothing to plot (check stride_tenor/stride_m).")

    # how many panels per figure (page)
    panels_per_fig = min(max_panels, total_panels)
    figs = []

    # Flatten panel ordering: loop by tenor row then moneyness col
    panel_pairs = [(j, m) for j in tenor_idx for m in m_idx]

    start = 0
    page = 1
    while start < total_panels:
        end = min(start + panels_per_fig, total_panels)
        chunk = panel_pairs[start:end]

        # pick a near-square grid for the current page
        n_panels = len(chunk)
        ncols = int(np.ceil(np.sqrt(n_panels)))
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
            sharex=sharex,
            sharey=sharey,
        )
        axes = np.atleast_1d(axes).ravel()

        for ax_i, (j, m) in enumerate(chunk):
            ax = axes[ax_i]
            ax.plot(t, vol_cube[:, j, m])  # no explicit color
            ax.set_title(f"T={tenors[j]:.3g}, m={log_moneyness[m]:.3g}")
            ax.grid(True, alpha=0.25)

        # turn off extra axes
        for k in range(n_panels, len(axes)):
            axes[k].axis("off")

        if suptitle:
            fig.suptitle(f"{suptitle} (page {page})" if total_panels > panels_per_fig else suptitle)

        fig.supxlabel("time")
        fig.supylabel("vol")
        fig.tight_layout()

        figs.append(fig)
        start = end
        page += 1

    if show:
        plt.show()

    return figs


def plot_vol_smile_over_time(
    vol_cube: np.ndarray,
    *,
    time_index: int,
    log_moneyness: np.ndarray,
    tenors: np.ndarray,
    t: np.ndarray | None = None,
    title: str | None = None,
    show: bool = True,
):
    """
    Plot the implied-vol "smile" at a fixed time index:
      x-axis: log-moneyness
      y-axis: implied vol
    On ONE figure: many curves, each curve corresponds to a different tenor.

    Parameters
    ----------
    vol_cube : array
        Shape (n_steps, J, M) or (n_steps+1, J, M)
    time_index : int
        Which time slice to plot (0 ... n_steps-1)
    log_moneyness : array
        Shape (M,)
    tenors : array
        Shape (J,)
    t : array, optional
        If provided, used for labeling the time in title.
    """
    vol_cube = np.asarray(vol_cube, dtype=float)
    if vol_cube.ndim != 3:
        raise ValueError(f"vol_cube must be 3D (n_steps, J, M). Got {vol_cube.shape}")

    n_steps, J, M = vol_cube.shape

    if not (0 <= time_index < n_steps):
        raise ValueError(f"time_index must be in [0, {n_steps-1}], got {time_index}")

    log_moneyness = np.asarray(log_moneyness, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    if log_moneyness.shape[0] != M:
        raise ValueError(f"log_moneyness length must be {M}, got {log_moneyness.shape[0]}")
    if tenors.shape[0] != J:
        raise ValueError(f"tenors length must be {J}, got {tenors.shape[0]}")

    if t is not None:
        t = np.asarray(t, dtype=float)
        if t.shape[0] != n_steps:
            raise ValueError(f"t length must match vol_cube first dim ({n_steps}), got {t.shape[0]}")
        time_label = f"t={t[time_index]:.4g}"
    else:
        time_label = f"index={time_index}"

    plt.figure(figsize=(8.5, 5.2))
    for j in range(J):
        plt.plot(log_moneyness, vol_cube[time_index, j, :], label=f"T={tenors[j]:.3g}")

    plt.xlabel("log-moneyness  m = log(S/K)")
    plt.ylabel("implied vol")
    plt.title(title if title is not None else f"Vol smile at {time_label}")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    if show:
        plt.show()


def plot_hist_grid(mat, strike, history, simulations, save_name = None):
    """
    history: T x M x K
    simulations: nsim x T x M x K
    """
    M = len(mat)
    K = len(strike)

    fig, ax = plt.subplots(M,K,figsize = (3*M, 3*K))


    for m in range(M):
        for k in range(K):
            ts = np.diff(history[:,m,k])
            ts_sim = np.diff(simulations[:,:, m,k].flatten())
            ax[m][k].hist(ts, color = "blue", alpha = 0.5, density = True, bins =30)
            ax[m][k].hist(ts_sim, color = "orange", alpha = 0.5, density = True, bins = 30)
            ax[m][k].set_title(f"Maturity {mat[m]} days, K = {strike[k]}S")
            ax[m][k].set_yscale("log")
            ax[m][k].grid(True)
    
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, dpi=300)


def plot_cdf_grid(mat, strike, history, simulations, save_name = None, log = False):
    """
    history: T x M x K
    simulations: nsim x T x M x K
    """
    M = len(mat)
    K = len(strike)

    fig, ax = plt.subplots(M,K,figsize = (3*M, 3*K))


    for m in range(M):
        for k in range(K):
            ts = history[:,m,k].flatten()
            ts_sim = simulations[:,:, m,k].flatten()
            ax[m][k].ecdf(ts, color = "blue", alpha = 1)
            ax[m][k].ecdf(ts_sim, color = "orange", alpha = 1)
            ax[m][k].set_title(f"Maturity {mat[m]} days, K = {strike[k]}S")
            if log:
                ax[m][k].set_yscale("log")
            ax[m][k].grid(True)
    
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, dpi=300)
    
