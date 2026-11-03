"""
HW3, Member 2 (Federico): data loading and construction of the linear system
(I - alpha W) x = (1 - alpha) s.

Functions only, no top-level execution. Run from the repository root.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

processed_folder = Path("data") / "processed"
edges_path = processed_folder / "edges_indexed.csv"
s_path = processed_folder / "m2_s_initial_opinions.npy"


def load_edges(path=edges_path):
    """
    Read the edge list saved by Member 1 (01_dataset_preprocessing.ipynb, cell 10).
    Edge (i, j) means "user i trusts user j", indices are already 0-based.

    :param path: csv file with columns source_index, target_index
    :return edges_df: pandas DataFrame with columns source_index, target_index
    """
    return pd.read_csv(path)


def build_W(edges_df):
    """
    Row-normalized trust matrix W, same logic as Member 1 (cells 8, 11, 12):
    W_ij = 1 / outdeg(i) for each edge i -> j. Sink nodes (outdeg 0) never
    appear as source, so their row of W stays zero (zero-row approach).

    :param edges_df: DataFrame returned by load_edges()
    :return W: scipy CSR matrix of shape (n_users, n_users)
    """
    rows = edges_df["source_index"].to_numpy()
    columns = edges_df["target_index"].to_numpy()
    n_users = int(max(rows.max(), columns.max())) + 1

    # out_degree[i] = number of times i appears as source (cell 8, vectorized)
    out_degree = np.bincount(rows, minlength=n_users)
    value = 1.0 / out_degree[rows]

    W = sp.coo_matrix((value, (rows, columns)),
                      shape=(n_users, n_users),
                      dtype=np.float64).tocsr()
    return W


def build_system(alpha=0.85, edges_df=None):
    """
    Friedkin-Johnsen system matrix A = I - alpha W (cell 16). The right-hand
    side is b = (1 - alpha) s, with s from initial_opinions().

    :param alpha: susceptibility to neighbours, (1 - alpha) is the dogmatism
    :param edges_df: optional DataFrame from load_edges(), read from disk if None
    :return A: scipy CSR matrix I - alpha W
    :return W: scipy CSR matrix W
    """
    if edges_df is None:
        edges_df = load_edges()
    W = build_W(edges_df)
    I = sp.eye(W.shape[0], format="csr", dtype=np.float64)
    A = (I - alpha * W).tocsr()
    return A, W


def bandwidth(A):
    """
    Bandwidth Bw(A) = max |i - j| over the nonzero entries a_ij (cell 18).

    :param A: scipy sparse matrix
    :return bw: integer bandwidth
    """
    i, j = A.nonzero()
    return int(np.max(np.abs(i - j)))


def rcm_permutation(A):
    """
    Reverse Cuthill-McKee ordering (cell 20). With symmetric_mode=False it
    works on the pattern of A + A^T, since A is not symmetric. Apply it as
    A[p, :][:, p] and b[p] (symmetric permutation P A P^T).

    :param A: scipy CSR matrix
    :return p: 1D numpy array, the permutation
    """
    return reverse_cuthill_mckee(A, symmetric_mode=False)


def initial_opinions(n_users, seed=42, path=s_path):
    """
    Intrinsic opinions s of the Friedkin-Johnsen model, s_i ~ Uniform[-1, 1]
    (-1 against, 0 neutral, +1 in favour). If the file already exists it is
    loaded instead of regenerated, so the whole group uses the same vector.

    :param n_users: number of users (75,879 for soc-Epinions1)
    :param seed: seed of the random number generator
    :param path: .npy file where s is saved
    :return s: 1D numpy array of length n_users
    """
    if path.exists():
        s = np.load(path)
        assert s.shape == (n_users,), "saved s has the wrong length"
        return s

    s = np.random.default_rng(seed).uniform(-1, 1, size=n_users)
    np.save(path, s)
    return s
