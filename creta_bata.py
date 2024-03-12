import numpy as np
import networkx as nx
import h5py
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def RunTA(edges, RT, T_avg, maxImages, param, TAMethod="BATA"):
    # Relabelling nodes
    print(edges.shape)
    nodes = np.unique(edges - 1).astype(int)
    print(nodes.shape)
    G = nx.Graph()
    G.add_weighted_edges_from(np.hstack((edges, np.arange(edges.shape[0]).reshape((-1, 1)))))
    D = dict(G.degree())
    loc = max(D, key=D.get)
    idxConst = np.where(nodes == loc)[0][0]
    G1 = G.subgraph(nodes)
    P = G1.edges
    edges_sg = np.array(P)
    idx = np.where(edges[:, 0] > edges[:, 1])[0]
    edge_idx_map = np.array([int(d["weight"]) for (_, _, d) in G1.edges(data=True)])
    RT[:, idx] = -RT[:, idx]
    RT_sg = RT[:, edge_idx_map]

    if len(T_avg) == 0:
        C_avg_temp = np.array([])
    else:
        C_avg_temp = T_avg[:, nodes]

    # Translation Averaging
    # if TAMethod == 'RLUD':
    #     C_avg_temp, ed_ret_idx = RLUD(edges_sg.T, -RT_sg, param, C_avg_temp)
    #     tacost = LUDCost(RT_sg, edges_sg, C_avg_temp)
    if TAMethod == 'BATA':
        C_avg_temp, ed_ret_idx = BATA(edges_sg.T, -RT_sg, param, C_avg_temp, idxConst)
        tacost = BATACost(RT_sg, edges_sg, C_avg_temp)
    else:
        raise ValueError('Invalid TA method')

    # Converting back to original labels
    T_avg = np.full((3, maxImages), np.nan)
    T_avg[:, nodes] = C_avg_temp
    ed_ret_idx[edge_idx_map] = ed_ret_idx

    return T_avg, tacost, ed_ret_idx

def RLUD(edges_sg, RT_sg, param, C_avg_temp):
    # RLUD implementation
    # (Implement RLUD function according to your needs)
    # This function should update C_avg_temp and return ed_ret_idx
    # based on the RLUD translation averaging method.
    # ...
    pass

def LUDCost(RT_sg, edges_sg, C_avg_temp):
    # LUDCost implementation
    # (Implement LUDCost function according to your needs)
    # This function should calculate and return the cost of
    # translation averaging based on the RLUD method.
    # ...
    pass

def BATACost(RT, edges, T):
    # This function determines BATA cost for a given solution T
    # Inputs:
    # RT: 3XN matrix of (Cj-Ci)/norm(Cj-Ci)
    # edges: MX2 matrix of edge
    # T: 3XN matrix of camera locations
    # Outputs:
    # cost: Mean error of all the edge costs

    T_Diff = T[:, edges[:, 1]] - T[:, edges[:, 0]]
    T_DiffNorm = np.sum(T_Diff * RT, axis=0) / np.sum(T_Diff * T_Diff, axis=0)
    T_DiffNorm[T_DiffNorm < 0] = 0
    costall = np.linalg.norm(T_Diff * T_DiffNorm - RT, axis=0, ord=2)
    costall = costall[~np.isnan(costall)]
    cost = np.mean(costall)

    return cost

def BATA(tij_index, tij_observe, param, t_init_given=None, idxConst=None):
    if t_init_given is None:
        t_init_given = np.empty((3, 0))
    
    numofcam = int(np.max(tij_index) + 1)
    numofobser = int(tij_observe.shape[1])
    Ri_T = np.tile(np.eye(3), numofobser).T
    Rj_T = Ri_T.copy()

    index_ti_I = np.column_stack((np.arange(0, 3 * numofobser),
                                  np.arange(0, 3 * numofobser),
                                  np.arange(0, 3 * numofobser))).astype(int)

    index_ti_J = (tij_index[1] - 1) * 3
    index_ti_J = np.concatenate((index_ti_J, index_ti_J + 1, index_ti_J + 2))
    index_ti_J = np.tile(index_ti_J, (3, 1)).T.astype(int)

    index_tj_I = np.column_stack((np.arange(0, 3 * numofobser),
                                  np.arange(0, 3 * numofobser),
                                  np.arange(0, 3 * numofobser))).astype(int)

    index_tj_J = (tij_index[1] - 1) * 3
    index_tj_J = np.concatenate((index_tj_J, index_tj_J + 1, index_tj_J + 2))
    index_tj_J = np.tile(index_tj_J, (3, 1)).T.astype(int)

    At0_full = csr_matrix((Ri_T.T.flatten(), (index_ti_I.flatten(), index_ti_J.flatten())),
                          shape=(3 * numofobser, 3 * numofcam)) - \
               csr_matrix((Rj_T.T.flatten(), (index_tj_I.flatten(), index_tj_J.flatten())),
                          shape=(3 * numofobser, 3 * numofcam))
    
    Z = np.zeros((4, 4))
    if t_init_given.size == 0:
        Aeq1 = np.sum(np.tile(tij_observe, (1, 3)).reshape(-1, numofobser) * At0_full, axis=0)
        Aeq2 = np.eye(3)
        Aeq = np.vstack([Aeq1, Aeq2])
        beq1 = numofobser
        beq2 = np.zeros(3)
        beq = np.hstack([beq1, beq2])
        
        # Initialization with LUDRevised
        Svec = np.random.rand(numofobser) + 0.5
        Svec = Svec / np.sum(Svec) * numofobser
        S = np.repeat(Svec, 3)
        W = np.ones(3 * numofobser)
        ii = 1
        
        errPrev = 1
        errCurr = 0
        while ii <= param['numofiterinit'] and abs(errPrev - errCurr) / errPrev > 1e-5:
            A = csr_matrix(np.diag(W)) * At0_full
            B = W * S * tij_observe.flatten()
            X = np.linalg.lstsq(np.vstack([A.T @ A, Aeq.T, np.hstack([Aeq, Z])]),
                                np.hstack([A.T @ B, beq]), rcond=None)[0]
            t = X[:3 * numofcam]
            Aij = At0_full @ t
            tij_T = tij_observe.flatten()
            Svec = np.sum(Aij * tij_T.reshape(-1, 1), axis=0) / np.sum(tij_T ** 2, axis=0)
            Svec[Svec < 0] = 0
            tmp3 = np.tile(Svec, 3)
            S = tmp3
            errPrev = errCurr
            tmp = np.sum((Aij - S * tij_T.reshape(-1, 1)) ** 2, axis=0)
            Wvec = 1 / (1 + (tmp / param['delta']) ** 2)
            errCurr = np.sum(np.sqrt(tmp))
            W = np.tile(np.sqrt(Wvec), 3)
            ii += 1
    else:
        t = t_init_given.flatten()

    # RLUD init
    ii = 1
    node_ret_idx = np.ones((3, numofcam), dtype=bool)
    errPrev = 1
    errCurr = 0
    beq2 = np.zeros(3)
    I = np.eye(3)
    
    while ii <= param['numofouteriter'] and abs(errPrev - errCurr) / errPrev > 1e-5:
        Aij = At0_full @ t
        tij_T = tij_observe.flatten()
        Svec = np.sum(Aij.reshape(3, -1) * tij_T.reshape(3, -1), axis=0) / \
               np.sum(tij_T.reshape(3, -1) ** 2, axis=0)
        print("Svec", Svec)
        
        ed_ret_idx, node_ret_idx = extractLargestConnComp(Svec, tij_index, node_ret_idx)
        numEdgesRet = np.sum(ed_ret_idx[0, :])
        numNodesRet = np.sum(node_ret_idx[0, :])
        
        t2 = t[node_ret_idx.flatten()]

        Svec = Svec[ed_ret_idx.flatten()]
        At0_fullr = At0_full[ed_ret_idx.flatten(), :][:, node_ret_idx.flatten()]
        
        tij_T = tij_observe[:, ed_ret_idx.flatten()]
        tmp3 = np.tile(Svec, 3)
        S_red = tmp3
        Wvec2 = Wvec[ed_ret_idx.flatten()]
        A = csr_matrix(np.diag(Wvec2)) * At0_fullr
        B = Wvec2 * Svec * tij_T.flatten()
        Aeq1 = np.sum(tij_T * At0_fullr, axis=0)
        Aeq2 = np.tile(I, numNodesRet)
        X = np.linalg.lstsq(np.vstack([A.T @ A, Aeq1, Aeq2.T]),
                            np.hstack([A.T @ B, np.sum(tij_T, axis=1), beq2]), rcond=None)[0]
        t[node_ret_idx.flatten()] = X[:numNodesRet]
        ii += 1

    # BATA
    t = t - np.tile(t[3 * idxConst - 2:3 * idxConst], numofcam)
    node_ret_idx_temp = np.ones((3, numofcam), dtype=bool)
    node_ret_idx_temp[:, idxConst - 1] = False
    t = t[node_ret_idx_temp.flatten()]
    At0 = At0_full[:, node_ret_idx_temp.flatten()]
    
    errPrev = 1
    errCurr = 0
    ii = 1
    
    while ii <= param['numofouteriter'] and abs(errCurr - errPrev) / errPrev > 1e-5:
        Aij = At0 @ t
        tij_T_weighted = tij_observe.flatten()
        Svec = np.sum(Aij ** 2, axis=0) / np.sum(Aij * tij_T_weighted.reshape(3, -1), axis=0)
        tmp3 = np.tile(Svec, 3)
        S = tmp3
        
        A = csr_matrix(np.diag(1 / S)) * At0
        B = tij_observe.flatten()
        tmp = np.sqrt(np.sum((A @ t - B) ** 2, axis=0))
        errPrev = errCurr
        errCurr = np.sum(tmp[ed_ret_idx.flatten()])
        Wvec = 1 / (1 + ((tmp / param['robustthre']) ** 2))
        
        jj = 1
        while jj <= param['numofinneriter']:
            Aij = At0 @ t
            tij_T_weighted = tij_observe.flatten()
            Svec = np.sum(Aij ** 2, axis=0) / np.sum(Aij * tij_T_weighted.reshape(3, -1), axis=0)
            
            ed_ret_idx, node_ret_idx_temp = extractLargestConnComp(Svec, tij_index, node_ret_idx)
            node_ret_idx_temp2 = node_ret_idx_temp[:, np.setdiff1d(np.arange(node_ret_idx_temp.shape[1]), idxConst - 1)]
            Svec = Svec[ed_ret_idx.flatten()]
            tmp3 = np.tile(Svec, 3)
            S_red = tmp3
            Wvec2 = Wvec[ed_ret_idx.flatten()]
            At0_fullr = At0_full[ed_ret_idx.flatten(), :][:, node_ret_idx_temp2.flatten()]
            tij_T = tij_observe[:, ed_ret_idx.flatten()]
            W = np.tile(np.sqrt(Wvec2), 3)
            
            A = csr_matrix(np.diag(W / S_red)) * At0_fullr
            B = W * tij_T.flatten()
            t2 = spsolve(A.T @ A, A.T @ B)
            t[node_ret_idx_temp2.flatten()] = t2
            jj += 1
        
        ii += 1
    
    t = t.reshape(3, -1)
    t[:, :idxConst - 1] = t[:, np.arange(idxConst - 1) - 1]
    t[:, idxConst - 1:] = t[:, np.arange(idxConst - 1, t.shape[1] - 1)]
    t[~node_ret_idx.flatten()] = np.nan
    t = t - np.nanmean(t, axis=1, keepdims=True)
    nodes = np.where(node_ret_idx[0, :])[0]
    ed_ret_idx = np.isin(tij_index[0, :], nodes) & np.isin(tij_index[1, :], nodes)
    
    return t, ed_ret_idx


def extractLargestConnComp(Svec, tij_index, node_ret_idx):
    ed_ret_idx = np.ones((3, tij_index.shape[1]), dtype=bool)
    ed_ret_idx[:, Svec < 0] = False
    nodes = np.where(node_ret_idx[0])[0]
    eidx = np.isin(tij_index[0], nodes) & np.isin(tij_index[1], nodes)
    ed_ret_idx[:, ~eidx] = False
    edges = tij_index[:, ed_ret_idx[0]]

    G = nx.Graph()
    G.add_edges_from(edges.T)
    bins = list(nx.connected_components(G))
    largest_comp = np.array(list(max(bins, key=len))).astype(int)
    ed_ret_idx_temp = np.isin(tij_index[0], largest_comp) & np.isin(tij_index[1], largest_comp) & ed_ret_idx[0]
    ed_ret_idx = np.tile(ed_ret_idx_temp, (3, 1))
    node_ret_idx[...] = False
    node_ret_idx[:, largest_comp] = True

    return ed_ret_idx, node_ret_idx



extractParallelRigidSubgraph = True
extractParallelRigidSubgraphIter = False
loadGTComponent = False
loadRavgFilteredGraph = True

computeInitialRelTransIRLS = False
computeInitialRelTransRANSAC = True

stepNormConvTol = 1e-6
TACostRelConvTol = 1e-5

TAMethod = 'BATA'

# Translation Averaging Parameters
param = {}
if TAMethod == 'RLUD':
    maxIters = 50
    # RLUD Parameters
    delta = 1e-5
    numofiterinit = 50
    numofouteriter = 20
    robustthre = 1e-1
elif TAMethod == 'BATA':
    maxIters = 10
    # Parameters for RLUD-BATA
    param["delta"] = 1e-5
    param["numofiterinit"] = 50
    param["numofouteriter"] = 20
    param["numofinneriter"] = 5
    param["robustthre"] = 1e-1
else:
    raise ValueError('Invalid TA method')

# Relative translation estimation parameters
SIGMA = 1e-2

# Filter Edges in Iteration
maxAngleDeg = 40

# Load data
#data = scipy.io.loadmat('dummy_data.mat')
f = h5py.File('dummy_data.mat','r')
print(list(f.keys()))
data = {"RT": f["RT"].value.T, "R_avg": f.get("R_avg").value.T, "edges": f.get("edges").value.T, "matches": f.get("matches").value, "maxImages": int(f.get("maxImages").value[0][0])}
print(data["R_avg"].shape)
matches = []
for _mtch in data["matches"].flatten():
    name = h5py.h5r.get_name(_mtch, f.id)
    matches.append(f[name].value.T)
data["matches"] = matches
# The data should contain the following variables:
# Graph has N nodes and M edges.
# RT: 3XM matrix of relative translation directions (Tij=Rj*(Ti-Tj)).
# edges: Mx2 matrix of camera pairs.
# matches: Mx1 list each containing Kx4 matrix.
#   Each matrix contains the point correspondences for each edge.
#   Each row in the matrix contains (xi,yi,xj,yj) where xi,yi and xj,yj
#   are coordinates for images i,j respectively after correcting for camera intrinsics.
# R_avg: 3X3XN matrix of absolute rotations.
# maxImages: Maximum No. of images in the dataset (used for indexing)
# NOTE: Ensure that the graph is connected and has the maximal parallel
# rigid component.

# Compute unit norm feature vectors
for i in range(len(data['matches'])):
    tempMat = data['matches'][i]
    numInliers = tempMat.shape[0]
    x1 = np.concatenate([tempMat[:, :2], np.ones((numInliers, 1))], axis=1)
    x1 = x1 / np.linalg.norm(x1, axis=1, keepdims=True)
    x2 = np.concatenate([tempMat[:, 2:4], np.ones((numInliers, 1))], axis=1)
    x2 = x2 / np.linalg.norm(x2, axis=1, keepdims=True)
    data['matches'][i] = np.concatenate([x1, x2], axis=1)

# Pre-processing matches
for i in range(len(data['matches'])):
    tempMat = data['matches'][i]
    tempMat[:, :3] = tempMat[:, :3] @ data['R_avg'][:, :, int(data['edges'][i, 0] - 1)]
    tempMat[:, 3:] = tempMat[:, 3:] @ data['R_avg'][:, :, int(data['edges'][i, 1] - 1)]
    data['matches'][i] = np.cross(tempMat[:, 3:], tempMat[:, :3])

# Pre-processing relative translations
for i in range(data['edges'].shape[0]):
    data['RT'][:, i] = -data['R_avg'][:, :, int(data['edges'][i, 1] - 1)].T @ data['RT'][:, i]

# CReTA
iter = 1
stepNorm = 1
tacostprev = 100
tacost = 1
print(data['maxImages'])
T_avg = np.array([])

while (iter <= maxIters and stepNorm > stepNormConvTol and abs(tacostprev - tacost) / tacostprev > TACostRelConvTol):
    # Run Translation Averaging
    C_avg_prev = T_avg.copy()
    tacostprev = tacost
    T_avg, tacost, ed_ret_idx = RunTA(data['edges'], data['RT'], T_avg, data['maxImages'], param)    

    # Remove edges that were filtered out
    eidx = ~ed_ret_idx
    data['edges'] = np.delete(data['edges'], eidx, axis=0)
    data['RT'] = np.delete(data['RT'], eidx, axis=1)
    data['matches'] = [data['matches'][i] for i in range(len(data['matches'])) if not eidx[i]]

    # Filter Edges in iteration
    RT_camSol = T_avg[:, data['edges'][:, 1]] - T_avg[:, data['edges'][:, 0]]
    RT_camSol = RT_camSol / np.linalg.norm(RT_camSol, axis=0)
    dotProduct = np.sum(data['RT'] * RT_camSol, axis=0)
    angles = np.abs(np.arccos(dotProduct))
    SolCalcDiffAngles = angles * 180 / np.pi

    eidx = SolCalcDiffAngles > maxAngleDeg
    data['edges'] = np.delete(data['edges'], eidx, axis=0)
    data['RT'] = np.delete(data['RT'], eidx, axis=1)
    data['matches'] = [data['matches'][i] for i in range(len(data['matches'])) if not eidx[i]]

    # Extract largest connected component
    G = nx.Graph()
    G.add_edges_from(data['edges'])
    bins = list(nx.connected_components(G))
    nodes = max(bins, key=len)  # Map from new (idx) to old (value)

    eidxRet = np.isin(data['edges'][:, 0], list(nodes)) & np.isin(data['edges'][:, 1], list(nodes))
    eidx = ~eidxRet
    data['edges'] = np.delete(data['edges'], eidx, axis=0)
    data['RT'] = np.delete(data['RT'], eidx, axis=1)
    data['matches'] = [data['matches'][i] for i in range(len(data['matches'])) if not eidx[i]]

    # Compute relative translations
    for i in range(data['edges'].shape[0]):
        tij = (T_avg[:, data['edges'][i, 1]] - T_avg[:, data['edges'][i, 0]])
        tij = tij / np.linalg.norm(tij)
        tempMat = data['matches'][i]
        err_vec = np.abs(tempMat @ tij)
        # Remove points at 1st iteration
        if iter == 1:
            s_err = np.sort(err_vec)
            remPtsIdx = err_vec > s_err[int(np.ceil(len(s_err) * 0.75))]
            err_vec = np.delete(err_vec, remPtsIdx)
            tempMat = np.delete(tempMat, remPtsIdx, axis=0)
            data['matches'][i] = tempMat
        w_vec = 1 / (1 + (err_vec / SIGMA) ** 2)
        _, _, v = svd(tempMat.T @ (w_vec * tempMat))
        tij_ref = v[:, -1]
        if tij @ tij_ref < 0:
            tij_ref = -tij_ref
        data['RT'][:, i] = tij_ref

    # Update parameters
    if iter > 1:
        stepDiff = C_avg_prev - T_avg
        stepDiff = np.delete(stepDiff, np.any(np.isnan(stepDiff), axis=0), axis=1)
        stepNorm = np.mean(np.linalg.norm(stepDiff, axis=0))
    print(f'Iteration: {iter} done')
    iter += 1



