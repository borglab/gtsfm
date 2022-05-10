"""Implements the CLEAR algorithm for CorrespondenceGraph estimation.

Refs:
- https://github.com/mit-acl/clear

Author: Travis Driver
"""

from typing import Optional

import numpy as np
import dask

from dask.delayed import Delayed
from scipy.sparse import csgraph

from gtsfm.data_association.correspondence_graph import CorrespondenceGraph
from gtsfm.data_association.corr_graph_estimator_base import CorrespondenceGraphEstimatorBase


class ClearCorrespondenceEstimator(CorrespondenceGraphEstimatorBase):
    """"""

    def __init__(self) -> None:
        """ """
        pass

    def run(self) -> CorrespondenceGraph:
        """ """
        # TT, numSmp = self.corr_graph.get_aggregate_assoc_matrix()
        # numAgt = self.corr_graph.num_images
        # XX = self.CLEAR(TT, numSmp, numAgt, numObj=None)
        pass

    def CLEAR(self, TT: np.ndarray, numSmp: np.ndarray, numAgt: int, numObj: Optional[int] = None) -> np.ndarray:
        """Runs CLEAR algorithm.

        Args:
            TT: initial permutations (aka. correspondences or score matrices).
            numSmp: number of observations for each frame.
            numAgt: number of frames.
            numObj: number of objects in the universe. The default is None. If None, numObj will be estimated from the
                spectrum of Laplacian automatically.

        Returns:
            XX: consistent pairwise permutations
            X: map to universe (lifting permutations)
            numObj: estimated number of objects.
        """

        # Also work with regular array input
        # TT = np.array(TT)
        # numSmp = np.array(numSmp)

        idxSum = np.append([0], np.cumsum(numSmp))  # Cumulative index
        numSmpSum = np.sum(numSmp)  # Total number of observations

        if numSmpSum != TT.shape[0]:
            raise ValueError("Incorrect number of samples.")

        P = (TT + TT.T) / 2  # Make association matrix symmetric
        for i in range(numAgt):  # Remove any self associations (distinctness constraint)
            l, r = idxSum[i], idxSum[i + 1]
            P[l:r, l:r] = np.eye(numSmp[i])  # Block of P associated to agent i

        A = P - np.diag(np.diag(P))  # Adjacency matrix of induced graph
        L = P2L(P)  # Graph Laplacian matrix

        # Normalize L
        Lnrm = normalize_lap(L, "DI", "sym")

        # Compute SVD using union of connected components' SVDs (to improve speed)
        sl, Vl = block_svd(A, Lnrm)

        # Estimate size of universe if not provided
        if numObj is None:
            numObj, _ = estimate_num_obj(sl, eigval=True, numSmp=numSmp)

        # Get the null space
        U0 = Vl[:, -numObj:]  # Kernel
        U = U0 / np.linalg.norm(U0, axis=1, keepdims=True)  # Normalize each row of U0

        # Find cluster center candidates
        C = pivot_rows(U, numObj)

        # Distance to cluster centers
        F = 1 - np.matmul(U, C.T)

        # Solve linear assignment
        X = np.zeros(F.shape)

        for i in range(numAgt):
            l, r = idxSum[i], idxSum[i + 1]
            Fi = F[l:r, :]  # Component of F associated to agent i

            # Suboptimal linear assignment
            Xi = suboptimal_assignment(Fi)

            X[l:r, :] = Xi  # Store results

        # Pairwise assignments
        XX = np.matmul(X, X.T)

        return XX

    def create_computation_graph(self) -> Delayed:
        """Create Dask task graph for correspondences."""
        return dask.delayed(self.run)()


# --------------------------------------------------------------------------
# Code for converting the aggregate permutation and Laplacian matrices
# --------------------------------------------------------------------------


def P2L(P):
    """
    Makes a Laplacian matrix out of a aggregate permutation matrix
    """
    # Generate Laplacian matrix
    P = P - np.diag(np.diag(P))
    L = np.diag(np.sum(P, axis=1)) - P
    return L


def L2P(L):
    """
    Makes an aggregate permutation matrix given a Laplacian matrix
    """
    # Generate aggregate permutation matrix
    P = -L
    P = P = np.diag(np.diag(P)) + np.eye(P.shape[0])
    return P


# --------------------------------------------------------------------------
# Code for normalizing the Laplacian matrix
# --------------------------------------------------------------------------


def normalize_lap(L, normalize="DI", multtype="sym", makesym=False):
    """
    Normalize Laplacian matrix
    Parameters
    ----------
    L : numpy.ndarray
        Laplacian matrix
    normalize : str, optional
        Degree+I or only degree matrix. The default is "DI".
    multtype : str, optional
        Multiplicaion type: symmetric or random walk. The default is "sym".
    makesym : bool, optional
        Make L symmetric or not. The default is False.
    Returns
    -------
    Lnrm : numoy.ndarray
        Normalized Laplacian matrix
    """
    # Take symmetric part of L if option specified
    if makesym:
        L = (L + L.T) / 2

    # Normalize L according to option specified
    Ldig = np.diag(L)  # Diagonal of L
    if normalize == "DI":
        Ddig = 1.0 / (Ldig + 1) ** (1 / 2)  # Diagonal of normalizer
    elif normalize == "D":
        Ddig - 1.0 / (Ldig) ** (1 / 2)  # Diagonal of normalizer
        Ddig[Ldig == 0] = 0

    rIdx, cIdx = np.where(L)  # row and column location of off-diagonal elements

    Lnrm = L  # Preallocate
    for i in range(rIdx.shape[0]):
        r, c = rIdx[i], cIdx[i]

        # Multiplication type based on option
        if multtype == "sym":
            Lnrm[r, c] = L[r, c] * Ddig[r] * Ddig[c]
        elif multtype == "randwalk":
            Lnrm[r, c] = L[r, c] * Ddig[r] * Ddig[r]

    return Lnrm


# --------------------------------------------------------------------------
# Code for singular value decomposition
# --------------------------------------------------------------------------


def block_svd(A, Lnrm):
    """
    Compute SVD using union of connected components' SVDs (to improve speed)
    Parameters
    ----------
    A : numpy.ndarray
        SQUARE adjacency matrix
        (this can be extended to non-square if desired)
    Lnrm : numpy.ndarray
        Laplacian matrix corresponding to A
    Returns
    -------
    sl : numpy.ndarray
        Vector of singular values
    Vl : numpy.ndarray
        V-matrix in SVD decomposition Lnrm = U*S*V.T
    """
    # Find graph communities
    numCom, labels = csgraph.connected_components(A)

    V = np.zeros(A.shape)  # Initialize matrix of eigenvectors
    sv = np.zeros((A.shape[0], 1))  # Vector of singular values
    for i in range(numCom):
        idx = labels == i  # Nodes that belong to community i
        idxgrid = np.ix_(idx, idx)  # Broadcasting helper
        Li = Lnrm[idxgrid]  # Part of matrix corresponding to the community
        _, Si, Vi = np.linalg.svd(Li)  # SVD of Li block
        Vi = Vi.T  # svd returns transpose of Vi, need to convert
        V[idxgrid] = Vi  # Store Vi in corresponding part of Vl
        sv[idx, :] = Si.reshape(-1, 1)  # Store singular values

    # Sort eigenvalues and vectors
    srtIdx = np.argsort(sv, axis=0)[::-1].reshape((-1,))

    sl, Vl = sv[srtIdx, :], V[:, srtIdx]
    # print(Vl[:, -2:])
    return sl, Vl


# --------------------------------------------------------------------------
# Code for estimating the size of the universe
# --------------------------------------------------------------------------


def estimate_num_obj(
    inp, method="fixed", multtype="sym", thresh=0.5, normalize="DI", makesym=False, eigval=False, numSmp=np.array([[]])
):
    """
    Estimate Number of objects in the universe
    Parameters
    ----------
    inp : np.ndarray
        Laplacian matrix 'L', or vector of eigenvalues 'sl'.
    method : str, optional
        'fixed' or 'gap'; Fixed threshold or Eigengap. The default is "fixed".
    multtype : str, optional
        'sym' or 'randwalk; Multiplicaion type: symmetric or random walk. The default is "sym".
    thresh : float, optional
        Threshold value for fixed method. The default is 0.5.
    normalize : str, optional
        Degree+I or only degree matrix. The default is "DI".
    makesym : bool, optional
        Make L symmetric or not. The default is False.
    eigval : bool, optional
        If eigenvalues of Laplacian are provided directly. The default is False.
    numSmp : np.ndarray, optional
        Number of observations for each agent. The default is np.array([[]]).
    Returns
    -------
    numObjEst : int
        Estimated number of objects in the universe.
    sl : numpy.ndarray
        Singularvalues of L.
    """

    if not eigval:  # If L is provided compute the eigenvalues first
        L = inp  # If input is a Laplacian matrix

        # Normalize L
        Lnrm = normalize_lap(L, normalize, multtype, makesym)

        P = L2P(L)
        A = P - np.diag(np.diag(P))  # Adjacency matrix of induced graph
        sl = block_svd(A, Lnrm)  # Use block SVD to improve speed
    else:
        sl = inp  # If eigenvalues are provided directly

    if method == "fixed":
        numObjEst = np.count_nonzero(sl < thresh)  # The number of eigenvalue < thresh

        # Limit the estimate
        numObjMin = np.max(numSmp)
        numObjEst = max(numObjEst, numObjMin)  # Minimum # of objects must not be less than max # of samples
    elif method == "gap":
        sd = np.abs(np.diff(sl, axis=0))  # Spectral gap
        srtIdx = np.argsort(sd, axis=0)[::-1].reshape((-1,))
        numObjIdx = sl.shape[0] - 1 - srtIdx

        if numSmp.size > 0:
            # Limit the estimate
            numObjMin = np.max(numSmp)
            numObjIdx = np.delete(
                numObjIdx, np.where(numObjIdx < numObjMin)
            )  # Remove indices where size of universe is less than # of observations of an agent

        numObjEst = numObjIdx[0]

    return numObjEst, sl


# --------------------------------------------------------------------------
# Code for finding the pivot rows
# --------------------------------------------------------------------------


def pivot_rows(U, numObj):
    """
    Choose pivot rows (rows that are orthogonal)
    Parameters
    ----------
    U : numpy.ndarray
        Matrix of eigenvectors.
    numObj : int
        Number of objects.
    Returns
    -------
    C : numpy.ndarray
        Matrix of pivot rows.
    """

    UU = np.abs(np.matmul(U, U.T))  # Matrix of all inner products
    pivIdx = np.zeros(numObj, dtype=int)  # Index of pivot rows

    pivIdx[0] = 0  # Take the first row as pivot
    sumVec = UU[:, 0]  # Column of NN associated to the pivot
    sumVec[0] = np.nan  # Use nan to avoid choosing the same pivot in future iterations

    # Find remaining pivots
    for i in range(1, numObj):
        idx = np.nanargmin(sumVec)  # Index of vector with smallest inner product to previous pivots
        pivIdx[i] = idx
        sumVec += UU[:, idx]  # Sum of inner products corresponding to chosen pivots
        sumVec[idx] = np.nan

    C = U[pivIdx, :]
    return C


# --------------------------------------------------------------------------
# Code for linear assignment
# --------------------------------------------------------------------------


def suboptimal_assignment(Fi):
    """
    Suboptimal assignment (instead of Hungarian) to improve speed
    Parameters
    ----------
    Fi : numpy.ndarray
        Distance matrix.
    Returns
    -------
    Xi : np.ndarray
        Assignment matrix.
    """
    numObsi = Fi.shape[0]  # Number of observations of agent i

    assign = np.zeros(numObsi, dtype=int)  # Assignment labels
    assign[:] = -2  # Set all initials ot inf
    Fi0 = Fi
    while np.any(assign == -2):
        rowid = np.where(assign == -2)[0]  # Index of rows not assigned
        idx = np.argmin(Fi0[rowid, :], axis=1)  # Find column index of smallest element in each row
        for i in range(rowid.size):
            if assign[rowid[i]] == -2:  # If no prior assignment
                idset = np.where(idx == idx[i])[0]  # Find if we have repeated assignment
                if idset.size == 1:  # Check if assignment is unique
                    assign[rowid[i]] = idx[i]  # Assign column index to observation
                    Fi0[:, idx[i]] = np.inf  # Set elements to inf to indicate column is assigned
                else:  # Repeated assignment indices
                    idx1 = np.argmin(Fi0[rowid[idset], idx[i]])  # Find min elements to resolve conflict
                    assign[rowid[idset]] = -1  # Temporarily set assignment for all conflicts to -1
                    assign[rowid[idset[idx1]]] = idx[i]  # Set assignment for best candidate
                    Fi0[:, idx[i]] = np.inf  # Set elements to inf to indicate column is assinged
        assign[assign == -1] = -2  # Set undetermined assignment to -2

    Xi = np.zeros(Fi.shape, dtype=int)  # Preallocate assignment matrix
    for i in range(numObsi):
        Xi[i, assign[i]] = 1

    return Xi
