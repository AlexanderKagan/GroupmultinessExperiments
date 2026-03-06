from statsmodels.stats.multitest import multipletests
from collections import Counter
import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from GroupMultiNeSS.utils import fill_diagonals


def make_layers_and_group_indices_from_attributes(subject_2_matrix: dict[str, np.array],
                                                  metadata: pd.DataFrame,
                                                  attrs_to_form_groups: tuple[str],
                                                  zerofill_diags: bool = True):

    assert set(attrs_to_form_groups).issubset(set(metadata.columns))
    assert "Subject" in metadata.columns
    assert set(subject_2_matrix.keys()).issubset(set(metadata["Subject"]))

    groups = list(product(*[list(metadata[attr].unique()) for attr in attrs_to_form_groups]))
    group_2_index = {group: idx for idx, group in enumerate(groups)}
    subject_2_group = {}
    for _, row in metadata.iterrows():
        group = tuple(row[attr] for attr in attrs_to_form_groups)
        subject_2_group[row["Subject"]] = group
    print(Counter(subject_2_group.values()))
    subjects = list(subject_2_group.keys())
    group_indices = np.array([group_2_index[subject_2_group[subject]] for subject in subjects])
    As = np.stack([subject_2_matrix[subject] for subject in subjects])
    if zerofill_diags:
        fill_diagonals(As, val=0.)
    return As, subjects, group_indices, group_2_index


def construct_cell_matrix(cor_matrix, regions, transform=None):
    assert len(regions) == len(cor_matrix)
    assert cor_matrix.shape[0] == cor_matrix.shape[1]
    assert all([val > 1 for val in Counter(regions).values()]), "each region should have at least 2 elements"
    if transform is None:
        transform = np.mean
    unique_regions = np.unique(regions)
    cell_matrix = pd.DataFrame(columns=unique_regions, index=unique_regions)
    for r1 in unique_regions:
        for r2 in unique_regions:
            r1_mask = regions == r1
            r2_mask = regions == r2
            sub_mat = cor_matrix[r1_mask][:, r2_mask]
            if r1 == r2:
                cors = sub_mat[np.triu_indices(sub_mat.shape[0], k=1)]
            else:
                cors = sub_mat.flatten()
            cell_matrix.at[r1, r2] = transform(cors)
    return cell_matrix


def regress_out_covariate_effects(As: np.ndarray, subjects: list,
                                  metadata: pd.DataFrame, covariates: list[str],
                                  include_intercept=True):
    assert set(subjects).issubset(set(metadata["Subject"]))
    assert set(covariates).issubset(set(metadata.columns))
    assert len(As) == len(subjects)
    M, n, _ = As.shape
    X = metadata.set_index("Subject").loc[subjects, covariates]
    X = pd.get_dummies(X, drop_first=True)
    
    cells = [(i, j) for i in range(n) for j in range(i + 1, n)]
    coefs_table = pd.DataFrame(columns=["Intercept"] + list(X.columns) if include_intercept else X.columns,
                               index=cells)
    cell_2_ys = {}
    As_resids = np.zeros_like(As, dtype=np.float32)

    X = StandardScaler().fit_transform(X)
    for cell, (i, j) in enumerate(cells):
        y = As[:, i, j]
        linreg = LinearRegression(fit_intercept=include_intercept)
        linreg.fit(X, y)
        y_pred = linreg.predict(X)
        resids = y - y_pred
        As_resids[:, i, j] = resids
        As_resids[:, j, i] = resids
        coefs_table.iloc[cell] = [linreg.intercept_, *linreg.coef_] if include_intercept else linreg.coef_
        cell_2_ys[(i, j)] = {"y_true": y, "y_pred": y_pred}
    return As_resids, coefs_table, cell_2_ys


def cell_matrix_permutation_test(cell_matrices, group_indices, n_permutations=1000):
    unique_regions = cell_matrices[0].index
    cell_matrices = np.array(cell_matrices)
    group_names = np.unique(group_indices)
    assert len(group_indices) == len(cell_matrices)
    assert cell_matrices.shape[1] == cell_matrices.shape[2]
    assert len(group_names) == 2

    m, n, _ = cell_matrices.shape  # m is the number of matrices (m x n x n)

    # Get upper-diagonal indices
    triu_indices = np.triu_indices(n, k=0)  # k=0 to include the diagonal

    # Extract the upper-diagonal entries from all matrices at once
    upper_entries = cell_matrices[:, triu_indices[0], triu_indices[1]]

    # Group labels (binary vector of 0s and 1s)
    group_0_entries = upper_entries[group_indices == group_names[0]]
    group_1_entries = upper_entries[group_indices == group_names[1]]

    # Compute the observed differences (mean of group 1 - mean of group 0)
    observed_diff = np.mean(group_1_entries, axis=0) - np.mean(group_0_entries, axis=0)

    # Initialize an array to hold the permuted differences
    permuted_diffs = []

    # Perform the permutation test
    for perm in range(n_permutations):
        # Shuffle the labels
        shuffled_indices = np.random.permutation(group_indices)

        # Group the permuted entries
        perm_group_0_entries = upper_entries[shuffled_indices == group_names[0]]
        perm_group_1_entries = upper_entries[shuffled_indices == group_names[1]]

        # Calculate the difference in means for the permuted data
        perm_diff = np.mean(perm_group_1_entries, axis=0) - np.mean(perm_group_0_entries, axis=0)
        permuted_diffs.append(perm_diff)

    # Compute p-values for each upper-diagonal entry
    p_values = np.round(np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff), axis=0), 3)

    # Adjust p-values for multiple comparisons (Benjamini-Hochberg)
    adjusted_p_values = np.round(multipletests(p_values, method='fdr_bh')[1], 3)

    p_values_df = pd.DataFrame(0., columns=unique_regions, index=unique_regions)
    adjusted_p_values_df = pd.DataFrame(0., columns=unique_regions, index=unique_regions)
    for idx, (r1, r2) in enumerate(zip(unique_regions[triu_indices[0]],
                                       unique_regions[triu_indices[1]])):
        p_values_df.at[r1, r2] = p_values[idx]
        adjusted_p_values_df.at[r1, r2] = adjusted_p_values[idx]
    return p_values_df, adjusted_p_values_df
