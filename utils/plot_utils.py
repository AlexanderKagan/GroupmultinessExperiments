import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, jarque_bera, shapiro, anderson
import seaborn as sns
from pandas import DataFrame
from GroupMultiNeSS.utils import cos_sim, pairwise_metric_matrix
from kneefinder import KneeFinder


def plot_errors(err_dict, x_range, ylabel="Error", xlabel=None,
                fontsize=15,
                colors=("black", "green", "orange", "red"),
                markers=("o", "^", "s", "d"),
                markersize=11,
                ax=None,
                legend=True):
    # sns.set_theme(style="whitegrid")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    for marker, color, (model_name, errors) in zip(markers, colors, err_dict.items()):
        sns.lineplot(x=x_range, y=errors, label=model_name, marker=marker, markersize=markersize,
                     color=color, linewidth=2, ax=ax)
    ax.tick_params(axis='both', labelsize=fontsize - 3)
    if legend:
        ax.legend(fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticks(x_range)


def plot_sim_metric_heatmap(matrices, metric, labels=None, title=None, **metric_kwargs):
    n_mat = len(matrices)
    metric_matrix = pairwise_metric_matrix(matrices, metric, **metric_kwargs)
    sns.heatmap(metric_matrix, mask=np.triu(np.ones((n_mat, n_mat)), k=1))
    plt.title(title)
    if labels:
        plt.xticks(ticks=np.linspace(0.5, n_mat - 0.5, n_mat), labels=labels, rotation=45)
        plt.yticks(ticks=np.linspace(0.5, n_mat - 0.5, n_mat), labels=labels, rotation=0)
    plt.show()


def check_if_normal_distrib(vals, test="shapiro", plot_normal_approx=True,
                            n_bins=None, ax=None, fontsize=14):

    mean, std = norm.fit(vals)
    if test == "shapiro":
        pval = shapiro(vals).pvalue
    elif test == "anderson":
        res = anderson(vals)
        pval = "reject (95%)" if res.statistic > res.critical_values[2] else "not rejected (95%)"
    elif test == "jarque-berra":
        pval = jarque_bera(vals).pvalue
    else:
        raise NotImplementedError()
        
    pval = np.round(pval, 5)
    
    if plot_normal_approx:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.hist(vals, bins=n_bins, density=True)
        x_min, x_max = ax.get_xlim()
        x = np.linspace(x_min, x_max, 100)
        p = norm.pdf(x, mean, std)
        ax.plot(x, p, color="black", linewidth=2, label=f"p-value={pval}")
        ax.tick_params(axis='both', labelsize=fontsize - 2)
        ax.set_ylabel("Density", fontsize=fontsize)
        ax.set_xlabel("Values", fontsize=fontsize)
        ax.legend(fontsize=fontsize)
    return pval


def plot_eigvals_elbow_plot(matrix, ax=None, normalize=True, find_knee=True,
                            marker=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    eigvals = np.linalg.svd(matrix, compute_uv=False)
    if normalize:
        eigvals = np.cumsum(eigvals) / np.sum(eigvals)
        ax.plot([0] + list(eigvals), marker=marker)
        ax.hlines(0.7, 0, len(matrix), color="red")
        ax.set_yticks(np.linspace(0, 1, 11))
    else:
        ax.plot(eigvals, marker=marker)
    if find_knee:
        knee_range = list(range(len(eigvals)))
        kn = KneeFinder(knee_range, eigvals)
        kn_x, kn_y = kn.find_knee()
        ax.vlines(kn_x, 0, max(eigvals), color="black", linestyle="dashed", label=f"Knee x={kn_x}")
        ax.legend()
    ax.set_title(title)


def plot_pvals_heatmap(pvals_df, title=None, annot=True):
    assert isinstance(pvals_df, DataFrame)
    assert pvals_df.shape[0] == pvals_df.shape[1]
    mask = np.tril(np.ones_like(pvals_df, dtype=bool), k=-1)
    ax = sns.heatmap(pvals_df, mask=mask, vmin=0, vmax=1, annot=annot)
    c_bar = ax.collections[0].colorbar
    c_bar.set_ticks(np.linspace(0, 1, 11))
    ax.set_title(title)
    plt.show()


def plot_multiplex_edge_predictions(As_true, As_pred, ax=None, title=None, eps=None, figsize=(8, 5)):
    assert As_true.shape == As_pred.shape
    assert As_true.ndim == 3
    assert As_true.shape[1] == As_true.shape[2]
    m, n, _ = As_true.shape
    triu_indices = np.triu_indices(n)
    triu_true = As_true[:, triu_indices[0], triu_indices[1]]
    triu_pred = As_pred[:, triu_indices[0], triu_indices[1]]
    present_indices = ~np.isnan(triu_true)
    flat_pred, flat_true = triu_pred[present_indices].flatten(), triu_true[present_indices].flatten()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(flat_pred, flat_true, alpha=0.3)
    min_ax, max_ax = np.min(np.hstack(flat_pred, flat_true)), np.max(np.hstack(flat_pred, flat_true))
    ax.plot([min_ax, max_ax], [min_ax, max_ax], linestyle="dashed", c="black", label="$y=x$")
    if eps is not None:
        ax.plot([min_ax, max_ax], [min_ax - eps, max_ax - eps], linestyle="dashed", c="red", label=f"$y=x\pm${eps}")
        ax.plot([min_ax, max_ax], [min_ax + eps, max_ax + eps], linestyle="dashed", c="red")
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Truth")
    ax.legend()


def compare_pairwise_matrix_metric(matrix_collections, names, metric=cos_sim,
                                   figsize=(30, 7), fontsize=18, compare_with_noise=True,
                                   **metric_kwargs):
    assert len(names) == len(matrix_collections)
    M, n, d = matrix_collections[0].shape
    n_plots = len(matrix_collections) + 1 if compare_with_noise else len(matrix_collections)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axs = [axs]

    for idx, (name, mat_collection) in enumerate(zip(names, matrix_collections)):
        sns.heatmap(pairwise_metric_matrix(mat_collection, metric=metric, **metric_kwargs),
                    mask=np.triu(np.ones((M, M))), vmin=0, vmax=1, ax=axs[idx])
        axs[idx].set_title(f"{metric.__name__} for {name}", fontsize=fontsize)
    if compare_with_noise:
        es = np.random.randn(M, n, d)
        sns.heatmap(pairwise_metric_matrix(es, metric=metric, **metric_kwargs),
                    mask=np.triu(np.ones((M, M))), vmin=0, vmax=1, ax=axs[-1])
        axs[-1].set_title(f"{metric.__name__} for random Gaussian noise", fontsize=fontsize)
    for ax in axs:
        c_bar = ax.collections[0].colorbar
        c_bar.set_ticks(np.linspace(0, 1, 11))
    plt.show()


def plot_latent_positions(X: np.array, node_types: list[str] = None,
                          node_shapes: list[str] = None, dims: tuple = (0, 1),
                          fontsize=10, title=None,
                          node_type_2_color=None,
                          ax=None, plot_legend=True, markersize=11, alpha=0.6):
    assert X.shape[1] >= 2
    assert len(dims) in (2, 3)
    if ax is None:
        if len(dims) == 2:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(1, 1, 1, projection='3d')

    if node_types is None:
        node_types = np.zeros(len(X))
    else:
        assert len(X) == len(node_types)

    if node_shapes is None:
        node_shapes = ['o'] * len(X)
    else:
        assert len(X) == len(node_shapes)

    plotted_legend_types = set()
    for node, (node_type, node_shape) in enumerate(zip(node_types, node_shapes)):
        color = node_type_2_color[node_type]
        node_type = None if node_type in plotted_legend_types else node_type
        ax.scatter(*X[node, dims],
                   label=node_type, alpha=alpha, s=markersize,
                   marker=node_shape, c=color)
        plotted_legend_types.add(node_type)
    if plot_legend:
        ax.legend(fontsize=fontsize + 3)
    ax.set_xlabel(f"Latent dimension {dims[0] + 1}", fontsize=fontsize, labelpad=10)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylabel(f"Latent dimension {dims[1] + 1}", fontsize=fontsize, labelpad=10)
    if len(dims) == 3:
        ax.set_zlabel(f"Latent dimension {dims[2] + 1}", fontsize=fontsize, labelpad=10)
    ax.set_title(title, fontsize=fontsize + 8)


def plot_components_errors(param_2_comp_errors_over_folds, param_name="parameter", ax=None, fontsize=10,
                           comp_names=(r'$S$', r'$Q$', r'$R$', r'$\Theta$'), plot_std=False,
                           figsize=(10, 8), xrot=0., markersize=11, plot_legend=True):
    param_2_comp_errors_over_folds = dict(sorted(param_2_comp_errors_over_folds.items(), key=lambda kv: kv[0]))
    param_range = list(param_2_comp_errors_over_folds.keys())
    param_values_over_folds = np.stack(list(param_2_comp_errors_over_folds.values()))
    if param_values_over_folds.ndim == 2:
        param_values_over_folds = param_values_over_folds[:, None, :]
    elif param_values_over_folds.ndim != 3:
        raise NotImplementedError()
    mean_errs_over_params = param_values_over_folds.mean(1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if not plot_std:
        for name, mean_err in zip(comp_names, mean_errs_over_params.T):
            sns.lineplot(x=param_range, y=mean_err, label=name, marker='o', markersize=markersize, ax=ax)
    else:
        std_errs_over_params = param_values_over_folds.std(1)
        for name, mean_err, std_err in zip(comp_names, mean_errs_over_params.T, std_errs_over_params.T):
            ax.errorbar(param_range, mean_err, yerr=std_err, label=name, marker='o')
    ax.tick_params(axis='both', labelsize=fontsize - 3)
    ax.set_xlabel(param_name, fontsize=fontsize)
    ax.set_ylabel("Relative Frobenius Error", fontsize=fontsize)
    ax.set_xticks(param_range, labels=param_range, rotation=xrot)
    if plot_legend:
        ax.legend(fontsize=fontsize)
    else:
        ax.legend().remove()


def significance_heatmap(values, pvals, colnames, ax=None, title=None, figsize=(8, 6), vmin=None, vmax=None,
                         crit_pvals=(0.01, 0.05, 0.1)):
    """
    Plot only the upper triangle (including diagonal) of a values matrix,
    annotated with significance stars based on pvals.
    """

    values = np.asarray(values)
    pvals = np.asarray(pvals)
    n = values.shape[0]
    crit_pvals = sorted(crit_pvals)
    assert 1 <= len(crit_pvals) <= 3

    # significance star function
    def get_stars(p):

        if len(crit_pvals) == 3:
            if p <= crit_pvals[0]:
                return "***"
            elif p <= crit_pvals[1]:
                return "**"
            elif p <= crit_pvals[2]:
                return "*"
            else:
                return ""
        elif len(crit_pvals) == 2:
            if p <= crit_pvals[0]:
                return "**"
            elif p <= crit_pvals[1]:
                return "*"
            else:
                return ""
        else:
            if p <= crit_pvals[0]:
                return "*"
            else:
                return ""

    # Build annotation matrix
    annot = np.full_like(values, "", dtype=object)

    for i in range(n):
        for j in range(i, n):  # <-- only upper triangle + diagonal
            stars = get_stars(pvals[i, j])
            annot[i, j] = f"{values[i, j]:.3g} {stars}"

    # Mask for lower triangle
    mask = np.zeros_like(values, dtype=bool)
    mask[np.tril_indices(n, k=-1)] = True  # hide lower triangle

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Diverging cmap: negative→blue, positive→red, 0→white
    # cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        values,
        mask=mask,  # <<< hide lower tr\angle
        annot=annot,
        fmt="",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=None,
        center=0,
        square=True,
        xticklabels=colnames,
        yticklabels=colnames,
        linewidths=0.5,
        linecolor="white"
    )

    if title is not None:
        ax.set_title(title)
