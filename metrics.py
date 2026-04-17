import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EMD_Surface:
    '''
    only np.arrays
    history: T x M x K
    sims: nsims x T x M x K

    returns emd_matrix, emd_mean
    '''

    def __init__(self, history):
        T, M, K = history.shape
        self.history = history.reshape(T, -1)
        self.q_grid = np.linspace(0.01, 0.99, 100)

        self.history_diff = np.diff(np.log(self.history), axis=0)
        self.q_history = np.quantile(self.history_diff, q=self.q_grid, axis=0)

        self.constant = np.sum(
            np.abs(self.history_diff.min(axis=0)[None, :] - self.q_history),
            axis=0
        ) / len(self.q_grid)

        self.constant += np.sum(
            np.abs(self.history_diff.max(axis=0)[None, :] - self.q_history),
            axis=0
        ) / len(self.q_grid)

        self.constant = np.maximum(self.constant, 1e-12)

    def compute(self, sims):
        nsims, T, M, K = sims.shape
        sims = sims.reshape(nsims, T, -1)
        sims_diff = np.diff(np.log(sims), axis=1).reshape(nsims * (T - 1), -1)

        emd_matrix = np.sum(
            np.abs(np.quantile(sims_diff, q=self.q_grid, axis=0) - self.q_history),
            axis=0
        ) / len(self.q_grid) / self.constant

        emd_matrix_mean = emd_matrix.mean()
        emd_matrix = emd_matrix.reshape(M, K)

        return emd_matrix, emd_matrix_mean

    def compute_sliced(self, sims, nrepeat=500, random_seed = 0):
        np.random.seed(random_seed)
        nsims, T, M, K = sims.shape
        sims = sims.reshape(nsims, T, -1)
        sims_diff = np.diff(np.log(sims), axis=1).reshape(nsims * (T - 1), -1)
    
        eps = np.random.normal(size=(nrepeat, M * K))
        eps = eps / np.linalg.norm(eps, axis=1, keepdims=True)
    
        sims_proj = eps @ sims_diff.T
        history_proj = eps @ self.history_diff.T
    
        sims_q = np.quantile(sims_proj, q=self.q_grid, axis=1)
        hist_q = np.quantile(history_proj, q=self.q_grid, axis=1)
    
        emd_per_proj = np.mean(np.abs(sims_q - hist_q), axis=0)
        emd_mean = emd_per_proj.mean()
    
        return emd_mean

    def visualize(self, iv_sims, M_labels, K_labels, figsize=(8, 6), cmap='viridis'):
        emd_matrix, emd_mean = self.compute(iv_sims)
    
        if emd_matrix.shape != (len(M_labels), len(K_labels)):
            raise ValueError(
                f'Shape mismatch: emd_matrix has shape {emd_matrix.shape}, '
                f'but len(M_labels)={len(M_labels)}, len(K_labels)={len(K_labels)}'
            )
    
        plt.figure(figsize=figsize)
        sns.heatmap(
            emd_matrix,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            xticklabels=K_labels,
            yticklabels=M_labels,
            cbar_kws={'label': 'Normalized 1D EMD'}
        )
        plt.title('EMD heatmap')
        plt.xlabel('Strike (fraction of spot)')
        plt.ylabel('Maturity (days)')
        plt.tight_layout()
        plt.show()
    
        print(f'Mean EMD: {emd_mean:.6f}')

import numpy as np


class MMD:
    def __init__(self, history, delta, block_size=1024):
        '''
        history: T x M x K
        delta: bandwidth of Gaussian kernel
        '''
        T, M, K = history.shape
        self.history = history.reshape(T, -1).astype(np.float32, copy=False)
        self.delta = delta
        self.block_size = block_size

        self.history_diff = np.diff(np.log(self.history), axis=0).astype(np.float32, copy=False)

        self.K_xx = self._Kxy(self.history_diff, self.history_diff, self.delta)

    def _gaussian_kernel_sum(self, X, Y, delta, symmetric=False):
        """
        Возвращает сумму kernel values:
            sum_{i,j} exp(-||X_i - Y_j||^2 / delta)

        Если symmetric=True, предполагается X is Y,
        и используется симметрия для ускорения.
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        n = X.shape[0]
        m = Y.shape[0]
        total = 0.0
        bs = self.block_size

        for i in range(0, n, bs):
            Xi = X[i:i + bs]
            Xi_sq = np.sum(Xi ** 2, axis=1, keepdims=True)

            j_start = i if symmetric else 0

            for j in range(j_start, m, bs):
                Yj = Y[j:j + bs]
                Yj_sq = np.sum(Yj ** 2, axis=1, keepdims=True).T

                dist2 = Xi_sq + Yj_sq - 2.0 * (Xi @ Yj.T)
                dist2 = np.maximum(dist2, 0.0)
                K_block = np.exp(-dist2 / delta)

                block_sum = K_block.sum(dtype=np.float64)

                if symmetric:
                    if i == j:
                        total += block_sum
                    else:
                        total += 2.0 * block_sum
                else:
                    total += block_sum

        return total

    def _Kxy(self, x, y, delta):
        n = x.shape[0]
        m = y.shape[0]

        if n == m and x is y:
            total = self._gaussian_kernel_sum(x, y, delta, symmetric=True)
            diag = n  # k(x_i, x_i) = 1 for Gaussian kernel
            return (total - diag) / (n * (n - 1))
        else:
            total = self._gaussian_kernel_sum(x, y, delta, symmetric=False)
            return total / (n * m)

    def compute_mmd(self, sims, delta=None):
        if delta is None:
            delta = self.delta

        nsims, T, M, K = sims.shape
        sims = sims.reshape(nsims, T, -1).astype(np.float32, copy=False)
        sims_diff = np.diff(np.log(sims), axis=1).reshape(nsims * (T - 1), -1).astype(np.float32, copy=False)

        if delta == self.delta:
            K_xx = self.K_xx
        else:
            K_xx = self._Kxy(self.history_diff, self.history_diff, delta)

        K_yy = self._Kxy(sims_diff, sims_diff, delta)
        K_xy = self._Kxy(self.history_diff, sims_diff, delta)

        return K_xx + K_yy - 2 * K_xy

import numpy as np


def acf_fft(x, nlags, adjusted=False, eps=1e-12):
    """
    x: array of shape (..., T)
    returns: ACF of shape (..., nlags + 1)
    """
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[-1]

    if nlags >= T:
        raise ValueError(f"nlags must be < series length, got nlags={nlags}, T={T}")

    x = x - x.mean(axis=-1, keepdims=True)

    nfft = 1 << (2 * T - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft, axis=-1)
    acov = np.fft.irfft(fx * np.conj(fx), n=nfft, axis=-1)[..., :nlags + 1]

    if adjusted:
        denom = (T - np.arange(nlags + 1)).reshape((1,) * (x.ndim - 1) + (nlags + 1,))
    else:
        denom = T

    acov = acov / denom
    acf = acov / np.maximum(acov[..., [0]], eps)

    return acf


def acf_score(history, sims, nlags=20, adjusted=False, drop_lag0=True):
    """
    history: T_hist x M x K
    sims: nsims x T_sim x M x K

    returns dict with score matrices and mean scores for:
    - initial series
    - log-difference
    - abs(log-difference)
    - squared(log-difference)
    """
    history = np.asarray(history, dtype=np.float64)
    sims = np.asarray(sims, dtype=np.float64)

    if history.ndim != 3:
        raise ValueError(f"history must be 3D: T_hist x M x K, got shape {history.shape}")
    if sims.ndim != 4:
        raise ValueError(f"sims must be 4D: nsims x T_sim x M x K, got shape {sims.shape}")

    T_hist, M_hist, K_hist = history.shape
    nsims, T_sim, M_sim, K_sim = sims.shape

    if (M_hist, K_hist) != (M_sim, K_sim):
        raise ValueError(
            f"Surface shapes must match: history has {(M_hist, K_hist)}, sims have {(M_sim, K_sim)}"
        )

    M, K = M_hist, K_hist

    history = history.reshape(T_hist, -1)    # T_hist x (M*K)
    sims = sims.reshape(nsims, T_sim, -1)    # nsims x T_sim x (M*K)

    def _score_from_series(history_arr, sims_arr, name):
        """
        history_arr: T1 x D
        sims_arr: nsims x T2 x D
        """
        T1 = history_arr.shape[0]
        T2 = sims_arr.shape[1]

        if nlags >= T1:
            raise ValueError(
                f"nlags={nlags} is too large for history in block '{name}': length={T1}"
            )
        if nlags >= T2:
            raise ValueError(
                f"nlags={nlags} is too large for sims in block '{name}': length={T2}"
            )

        acf_history = acf_fft(history_arr.T, nlags=nlags, adjusted=adjusted)            # D x (L+1)
        acf_sims = acf_fft(sims_arr.transpose(0, 2, 1), nlags=nlags, adjusted=adjusted) # nsims x D x (L+1)
        acf_sims_mean = acf_sims.mean(axis=0)                                            # D x (L+1)

        if drop_lag0:
            diff = acf_history[:, 1:] - acf_sims_mean[:, 1:]
        else:
            diff = acf_history - acf_sims_mean

        score = (diff ** 2).sum(axis=1)         # D
        score_matrix = score.reshape(M, K)      # M x K
        score_mean = score.mean()

        return score_matrix, score_mean, acf_history, acf_sims_mean

    # 1. ACF for initial series
    initial_matrix, initial_mean, acf_initial_history, acf_initial_sims = _score_from_series(
        history, sims, name="initial"
    )

    # 2. ACF for log-difference
    history_diff = np.diff(np.log(history), axis=0)   # (T_hist-1) x D
    sims_diff = np.diff(np.log(sims), axis=1)         # nsims x (T_sim-1) x D

    log_diff_matrix, log_diff_mean, acf_diff_history, acf_diff_sims = _score_from_series(
        history_diff, sims_diff, name="log_diff"
    )

    # 3. ACF for abs values of log-difference
    history_abs_diff = np.abs(history_diff)
    sims_abs_diff = np.abs(sims_diff)

    abs_log_diff_matrix, abs_log_diff_mean, acf_abs_diff_history, acf_abs_diff_sims = _score_from_series(
        history_abs_diff, sims_abs_diff, name="abs_log_diff"
    )

    # 4. ACF for squared values of log-difference
    history_sq_diff = history_diff ** 2
    sims_sq_diff = sims_diff ** 2

    sq_log_diff_matrix, sq_log_diff_mean, acf_sq_diff_history, acf_sq_diff_sims = _score_from_series(
        history_sq_diff, sims_sq_diff, name="sq_log_diff"
    )

    return {
        "initial": {
            "matrix": initial_matrix,
            "mean": initial_mean,
            "acf_history": acf_initial_history,
            "acf_sims_mean": acf_initial_sims,
        },
        "log_diff": {
            "matrix": log_diff_matrix,
            "mean": log_diff_mean,
            "acf_history": acf_diff_history,
            "acf_sims_mean": acf_diff_sims,
        },
        "abs_log_diff": {
            "matrix": abs_log_diff_matrix,
            "mean": abs_log_diff_mean,
            "acf_history": acf_abs_diff_history,
            "acf_sims_mean": acf_abs_diff_sims,
        },
        "sq_log_diff": {
            "matrix": sq_log_diff_matrix,
            "mean": sq_log_diff_mean,
            "acf_history": acf_sq_diff_history,
            "acf_sims_mean": acf_sq_diff_sims,
        },
    }


def visualize_acf_panels(
    history,
    sims,
    m_idx,
    k_idx,
    nlags=20,
    nsamples=10,
    adjusted=False,
    drop_lag0=False,
    random_state=0,
    figsize=(12, 8)
):
    """
    history: T_hist x M x K
    sims: nsims x T_sim x M x K
    m_idx, k_idx: indices of the surface point to visualize

    Draws a 2x2 grid:
    - initial series
    - log-difference
    - abs(log-difference)
    - squared(log-difference)
    """

    history = np.asarray(history, dtype=np.float64)
    sims = np.asarray(sims, dtype=np.float64)

    if history.ndim != 3:
        raise ValueError(f"history must be 3D: T x M x K, got {history.shape}")
    if sims.ndim != 4:
        raise ValueError(f"sims must be 4D: nsims x T x M x K, got {sims.shape}")

    T_hist, M_hist, K_hist = history.shape
    nsims, T_sim, M_sim, K_sim = sims.shape

    if (M_hist, K_hist) != (M_sim, K_sim):
        raise ValueError(
            f"Surface shapes must match: history has {(M_hist, K_hist)}, sims have {(M_sim, K_sim)}"
        )

    if not (0 <= m_idx < M_hist and 0 <= k_idx < K_hist):
        raise IndexError(
            f"(m_idx, k_idx)=({m_idx}, {k_idx}) is out of bounds for shape {(M_hist, K_hist)}"
        )

    hist0 = history[:, m_idx, k_idx]      # (T_hist,)
    sims0 = sims[:, :, m_idx, k_idx]      # (nsims, T_sim)

    if nsamples > nsims:
        nsamples = nsims

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(nsims, size=nsamples, replace=False)

    def _prepare_blocks(hist_series, sim_series):
        if np.any(hist_series <= 0) or np.any(sim_series <= 0):
            raise ValueError("Log-difference blocks require strictly positive values.")

        hist_diff = np.diff(np.log(hist_series), axis=0)
        sim_diff = np.diff(np.log(sim_series), axis=1)

        return {
            "Initial series": (hist_series, sim_series),
            "Log-difference": (hist_diff, sim_diff),
            "Abs(log-difference)": (np.abs(hist_diff), np.abs(sim_diff)),
            "Squared(log-difference)": (hist_diff ** 2, sim_diff ** 2),
        }

    blocks = _prepare_blocks(hist0, sims0)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, (title, (hist_series, sim_series)) in zip(axes, blocks.items()):
        T1 = hist_series.shape[0]
        T2 = sim_series.shape[1]

        if nlags >= T1:
            raise ValueError(
                f"nlags={nlags} is too large for history in block '{title}': length={T1}"
            )
        if nlags >= T2:
            raise ValueError(
                f"nlags={nlags} is too large for sims in block '{title}': length={T2}"
            )

        hist_acf = acf_fft(hist_series[None, :], nlags=nlags, adjusted=adjusted)[0]   # (L+1,)
        sim_acf = acf_fft(sim_series, nlags=nlags, adjusted=adjusted)                  # (nsims, L+1)
        sim_acf_mean = sim_acf.mean(axis=0)                                            # (L+1,)

        if drop_lag0:
            lags = np.arange(1, nlags + 1)
            hist_plot = hist_acf[1:]
            sim_mean_plot = sim_acf_mean[1:]
            sim_plot = sim_acf[sample_idx, 1:]
        else:
            lags = np.arange(nlags + 1)
            hist_plot = hist_acf
            sim_mean_plot = sim_acf_mean
            sim_plot = sim_acf[sample_idx]

        for i, idx in enumerate(sample_idx):
            ax.plot(
                lags,
                sim_plot[i],
                linewidth=1.0,
                alpha=0.25,
                label="Simulations" if i == 0 else None
            )

        ax.plot(
            lags,
            sim_mean_plot,
            linewidth=2.5,
            linestyle="--",
            label="Mean over simulations"
        )
        ax.plot(
            lags,
            hist_plot,
            linewidth=2.5,
            label="History"
        )

        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"ACF comparison at surface point (M={m_idx}, K={k_idx})", y=0.98)
    #plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.tight_layout()
    plt.show()