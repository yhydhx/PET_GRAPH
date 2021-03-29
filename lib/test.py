


import numpy as np
import scipy.sparse
import sklearn.metrics
from sklearn.model_selection import train_test_split


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def HEM_one_level(rr,cc,vv,rid,weights):

    nnz = rr.shape[0]
    N = rr[nnz-1] + 1

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    
                    # First approach
                    if 2==1:
                        tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                    
                    # Second approach
                    if 1==1:
                        Wij = vv[rs+jj]
                        Wii = vv[rowstart[tid]]
                        Wjj = vv[rowstart[nid]]
                        di = weights[tid]
                        dj = weights[nid]
                        tval = (2.*Wij + Wii + Wjj) * 1./(di+dj+1e-9)
                    
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id



def HEM(W, levels, rid=None):
    """
    Coarsen a graph multiple times using the Heavy Edge   (HEM).

    Input
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    Output
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

    Note
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """

    N, N = W.shape
    print(N)
    if rid is None:
        rid = np.random.permutation(range(N))
     
    ss = np.array(W.sum(axis=0)).squeeze()
    print(ss)
    rid = np.argsort(ss)
        
        
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)

    print('Heavy Edge Matching coarsening with Xavier version')

    for _ in range(levels):

        # CHOOSE THE WEIGHTS FOR THE PAIRING
        # weights = ones(N,1)       # metis weights
        weights = degree            # graclus weights
        # weights = supernode_size  # other possibility
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = scipy.sparse.find(W)
        cc = idx_row
        rr = idx_col
        vv = val
        
        # TO BE SPEEDUP
        if not (list(cc)==list(np.sort(cc))):
            tmp=cc
            cc=rr
            rr=tmp

        cluster_id = HEM_one_level(cc,rr,vv,rid,weights) # cc is ordered
        parents.append(cluster_id)

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = scipy.sparse.csr_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
        W.eliminate_zeros()
        
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        degree = W.sum(axis=0)
        #degree = W.sum(axis=0) - W.diagonal()

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        #[~, rid]=sort(ss);     # arthur strategy
        #[~, rid]=sort(supernode_size);    #  thomas strategy
        #rid=randperm(N);                  #  metis/graclus strategy
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents


from util import *

N = 69
rid = np.random.permutation(range(N))
print(rid)
print(len(rid))
num = 120
data = np.load("../mat%d.npy" %num)
print(data.shape)
label = np.load("../label%d.npy" %num)
print(label.shape)

data = data[label<=2]
label = label[label<=2]
print(label)
exit()

# MCI & NC
if True:
	data = data[label<=2]
	label = label[label<=2]
	label[label>=1] = 1


X_train, X_test, y_train, y_test = train_test_split(
data, label, test_size=1-0.7, random_state=55)

g_mod = np.corrcoef(X_train.transpose()) 
print(X_train)
print(X_train.transpose())

g_mod[np.isnan(g_mod)]=0 
np.fill_diagonal(g_mod, 1)
#g_mod = (g - np.min(g))/(np.max(g) - np.min(g))
g_mod[g_mod<=0.7] = 0
g_sparse = scipy.sparse.coo_matrix(g_mod)

graphs, parents = HEM(g_sparse, 4)