import random
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy
import nibabel as nib
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from lib.coarsening import *


# def load_mi_data(ep):
#     excelpath = ep
#     df = pd.read_excel(excelpath)

#     MCI_1 = df.index[df['MCI'] == 1].tolist()
#     MCI_0 = df.index[df['MCI'] == 0].tolist()
#     random.shuffle(MCI_1)
#     MCI_1_train = MCI_1[0:68]
#     MCI_1_test = MCI_1[68:]
#     random.shuffle(MCI_0)
#     MCI_0_train = MCI_0[0:43]
#     MCI_0_test = MCI_0[43:]

#     # print(df.loc[MCI_1_test])
#     mat_0_train = df.loc[MCI_0_train]['subject ID'].tolist()
#     mat_0_test = df.loc[MCI_0_test]['subject ID'].tolist()
#     mat_1_train = df.loc[MCI_1_train]['subject ID'].tolist()
#     mat_1_test = df.loc[MCI_1_test]['subject ID'].tolist()
#     print(len(mat_0_train), len(mat_0_test))
#     print(len(mat_1_train), len(mat_1_test))
#     return mat_0_train, mat_0_test, mat_1_train, mat_1_test


# def CalGraph(matlist, filepath):
#     g_list = []
#     for mt in range(len(matlist)):
#         matfile = filepath + 'MI_ResOC_suj' + str(matlist[mt]) + '.mat'
#         mat = sio.loadmat(matfile)
#         graph = np.asarray(mat['mi'])[:, :, 5, :]
#         # print(graph.shape)
#         for ind in range(graph.shape[2]):
#         # for ind in range(1):
#             g_list.append(graph[:, :, ind])
#         print(len(g_list))
#     return g_list

def load_data(train_rate=0.8, thresh=0.4, binary=True, num=180, state="fixed", random_seed=None):
    data = np.load("mat%d.npy" %num)
    print(data.shape)
    label = np.load("label%d.npy" %num)
    print(label.shape)
    # MCI & NC
    if binary:
        data = data[label<=2]
        label = label[label<=2]
        label[label>=1] = 1
    else :
        #trible label 
        data = data[label<=2]
        label = label[label<=2]
        pass
        #four labels

        
    
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=1-train_rate, random_state=random_seed)

    
    if state == "random":
        from numpy import random
        g = random.random([num, num])
        g = (g + g.transpose()) / 2
        value = np.sort(g.reshape(num*num))[int(num*num*0.78)]
        print(value)
        g[g<value] = value
        g_mod = (g - np.min(g))/(np.max(g) - np.min(g)) * (1 - thresh) + thresh
        g_mod[g_mod<=thresh] = 0
        print(len(g_mod[g_mod>thresh]))
        np.fill_diagonal(g_mod, 1)
    elif state == "eye":
        g_mod = np.eye(num)

    elif state == "corr":
        g_mod = np.corrcoef(data.transpose()) 
        g_mod[np.isnan(g_mod)]=0 
        np.fill_diagonal(g_mod, 1)
        #g_mod = (g - np.min(g))/(np.max(g) - np.min(g))
        g_mod[g_mod<=thresh] = 0
        print(len(g_mod[g_mod>thresh]))
    else:
        raise NotImplementedError

    # g_mod[g_mod>=thresh] = 1
    g_sparse = scipy.sparse.coo_matrix(g_mod)

    return X_train, X_test, y_train, y_test, g_sparse


def process_data(num=180):
    # fp = "/home/jiaming/Desktop/ADNI_AV45PET/"
    # # fp = "/Users/zhaoxuandong/Public/Dropbox (Partners HealthCare)/MImadrid/"
    # # ep = "/Users/zhaoxuandong/Public/Dropbox (Partners HealthCare)/MImadrid/meta/metaData.xlsx"
    print("loading data...")

    if num == 120:

        # 

        meta = np.load("AAL2.npy")
        coo_dict = {}

        # for i in range(1, 49):
        #     coo_dict[i] = np.where(meta_cort == i)

        # for i in range(1, 22):
        #     coo_dict[i + 48] = np.where(meta_sub == i)
        m = np.unique(meta)
        for i in range(1, 121):
            coo_dict[i] = np.where(meta == m[i])
            # print(coo_dict[i])

        vector_dict = []

        for i in tqdm(range(1, num + 1)):
            vector = []
            for folder in ["NC", "EMCI", "LMCI", "AD"]:
                for root, dirs, files in os.walk(folder):
                    for name in files:
                        if name[-1] != 'i':
                            continue
                        img = nib.load(os.path.join(root, name)).get_data()
                        mask = np.load('mask.npy')
                        img = img * mask
                        #img = (img - np.min(img))/(np.max(img) - np.min(img))
                        img = img[coo_dict[i]]
                        vector.append(np.average(img))
            vector_dict.append(np.array(vector))

        matrix_69 = np.array(vector_dict).transpose()

        np.save("mat120", matrix_69)

        print(matrix_69.shape) # 419 * 180

        label = [0 for i in range(100)] + [1 for i in range(96)] + [2 for i in range(131)] + [3 for i in range(92)]

        label = np.array(label)
        np.save("label120", label)

    elif num == 180:

        # 

        meta = np.load("mmp.npy")
        coo_dict = {}

        # for i in range(1, 49):
        #     coo_dict[i] = np.where(meta_cort == i)

        # for i in range(1, 22):
        #     coo_dict[i + 48] = np.where(meta_sub == i)

        for i in range(1, 181):
            coo_dict[i] = np.where(meta == i)
            # print(coo_dict[i])

        vector_dict = []

        for i in tqdm(range(1, num + 1)):
            vector = []
            for folder in ["NC", "EMCI", "LMCI", "AD"]:
                for root, dirs, files in os.walk(folder):
                    for name in files:
                        if name[-1] != 'i':
                            continue
                        img = nib.load(os.path.join(root, name)).get_data()
                        mask = np.load('mask.npy')
                        img = img * mask
                        #img = (img - np.min(img))/(np.max(img) - np.min(img))
                        img = img[coo_dict[i]]
                        vector.append(np.average(img))
            vector_dict.append(np.array(vector))

        matrix_69 = np.array(vector_dict).transpose()

        np.save("mat180", matrix_69)

        print(matrix_69.shape) # 419 * 180

        label = [0 for i in range(100)] + [1 for i in range(96)] + [2 for i in range(131)] + [3 for i in range(92)]

        label = np.array(label)
        np.save("label180", label)

    elif num == 69:
        # fp = "/home/jiaming/Desktop/ADNI_AV45PET/"
        meta_cort = nib.load("HarvardOxford-cort-maxprob-thr25-2mm.nii").get_data() # 48
        meta_sub = nib.load("HarvardOxford-sub-maxprob-thr25-2mm.nii").get_data() # 21

        coo_dict = {}

        for i in range(1, 49):
            coo_dict[i] = np.where(meta_cort == i)

        for i in range(1, 22):
            coo_dict[i + 48] = np.where(meta_sub == i)

        vector_dict = []

        for i in tqdm(range(1, num + 1)):
            vector = []
            for folder in ["NC", "EMCI", "LMCI", "AD"]:
                for root, dirs, files in os.walk(folder):
                    for name in files:
                        if name[-1] != 'i':
                            continue
                        img = nib.load(os.path.join(root, name)).get_data()
                        mask = np.load('mask.npy')
                        img = img * mask
                        #img = (img - np.min(img))/(np.max(img) - np.min(img))
                        img = img[coo_dict[i]]
                        vector.append(np.average(img))
            vector_dict.append(np.array(vector))

        matrix_69 = np.array(vector_dict).transpose()

        np.save("mat69", matrix_69)

        print(matrix_69.shape) # 419 * 69

        label = [0 for i in range(100)] + [1 for i in range(96)] + [2 for i in range(131)] + [3 for i in range(92)]

        label = np.array(label)
        np.save("label69", label)

    elif num == 264:
        from nilearn import datasets 
        adhd = datasets.fetch_adhd(n_subjects=1)  
        power = datasets.fetch_coords_power_2011()
        mask = power.rois
        mask = np.array([mask.x, mask.y, mask.z]).transpose()
        mask[:, 0] = mask[:, 0] + 90
        mask[:, 1] = mask[:, 1] + 130
        mask[:, 2] = mask[:, 2] + 70
        mask = mask / 2
        mask = mask.astype(int)
        print(mask.shape)

        coo_dict = {}

        assert mask.shape[0] == num

        mask_264 = np.zeros([91, 109, 91])

        for i in range(num):
            xl = []
            yl = []
            zl = []
            cx, cy, cz = mask[i]
            #print(cx, cy, cz)
            for x in [cx-2, cx-1, cx, cx+1, cx+2]:
                for y in [cy-2, cy-1, cy, cy+1, cy+2]:
                    for z in [cz-2, cz-1, cz, cz+1, cz+2]:
                        if x >= 0 and y >= 0 and z >= 0 and x < 91 and y < 109 and z < 91:
                            if (x-cx) **2 + (y-cy) **2 + (z-cz) **2 < 10.25:
                                xl.append(x)
                                yl.append(y)
                                zl.append(z)
                                mask_264[x, y, z] = i + 1

            coo_dict[i] = (np.array(xl), np.array(yl), np.array(zl))
        #print(coo_dict)

        print(np.unique(mask_264))

        np.save("mask264", mask_264)

        vector_dict = []

        np.set_printoptions(suppress=True)

        for i in tqdm(range(num)):
            vector = []
            for folder in ["NC", "EMCI", "LMCI", "AD"]:
                for root, dirs, files in os.walk(folder):
                    for name in files:
                        if name[-1] != 'i':
                            continue
                        img = nib.load(os.path.join(root, name)).get_data()
                        mask = np.load('mask.npy')
                        img = img * mask
                        #img = (img - np.min(img))/(np.max(img) - np.min(img))
                        print(folder, img[[26, 57, 69]])

                        img = img[coo_dict[i]]
                        # print(img)
                        vector.append(np.average(img))
            vector_dict.append(np.array(vector))

        matrix_69 = np.array(vector_dict).transpose()

        r, c = matrix_69.nonzero()
        c_unique = np.unique(c)
        matrix_69 = matrix_69[:, c_unique]

        print(matrix_69.shape) # 419 * 264

        np.save("mat264", matrix_69)

        label = [0 for i in range(100)] + [1 for i in range(96)] + [2 for i in range(131)] + [3 for i in range(92)]

        label = np.array(label)
        np.save("label264", label)

    else:
        raise NotImplementedError

# def construct_graph(M):
#     # data shape: (N, 10404)
#     I = []
#     J = []

#     for i in range(M):
#         for j in range(M):
#             for k in range(M):
#                 I.append(i * M + j)
#                 J.append(i * M + k)
#             for k in range(M):
#                 I.append(i * M + j)
#                 J.append(k * M + j)
#     V = np.ones(M * M * M * 2)
#     W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M * M, M * M))

#     W.setdiag(0)

#     # assert type(W) is scipy.sparse.csr.csr_matrix
#     return W


def signal(A):
    m, n = A.shape
    X = np.array(A).reshape(m * n, 1)
    return X

def update_graph_aftercoarse(L):
    coarsening_levels = 4
    #print("updating the graph")
    deepth = len(L)
    # only change the first layer
    #print(L[0])
    node_size = L[0].shape[0]
    #randomly remove or add edge
    #print(L[0][0,0])
    x = random.choice(range(node_size))
    y = random.choice(range(node_size))
    while x ==y:
        x = random.choice(range(node_size))
        y = random.choice(range(node_size))
    
    if(L[0][x,y]) == 0.0:
        L[0][x,y] = 1
        L[0][y,x] = 1
    else:
        L[0][x,y] = 0
        L[0][y,x] = 0
    


    return L


def update_graph(A):
    print("updating the graph")
    coarsening_levels = 4
    node_size,node_size = A.shape
    dense_A = A.tocsr()
    #print(dense_A)
   
    #print(A.shape)
    

    #randomly remove or add edge
    #print(L[0][0,0])
    x = random.choice(range(node_size))
    y = random.choice(range(node_size))
    while x ==y:
        x = random.choice(range(node_size))
        y = random.choice(range(node_size))
    #print("========================")
    #print(x,y,dense_A[x,y],dense_A[x,y] == 0.0)
    if dense_A[x,y] == 0.0:
        dense_A[x,y] = 1
        dense_A[y,x] = 1
    else:
        dense_A[x,y] = 0
        dense_A[y,x] = 0
    #print(x,y,dense_A[x,y],dense_A[x,y] == 0.0)
    A = scipy.sparse.coo_matrix(dense_A)
    #print(A)
    #exit()
    L, perm = coarsen(A, coarsening_levels)

   
    # Compute max eigenvalue of graph Laplacians
    lmax = []
    for i in range(coarsening_levels):
        lmax.append(lmax_L(L[i]))
    #print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))
    
    return A, L, lmax

def  graph_similiarity(A, B):
    '''
    Consider the edge weight 
    
    T : total edge of both graphs
    D : difference between two graphs

    '''
    T,D = (0.0,0.0)
    print(A)
    if A.shape != B.shape:
        return 0

    size,size = A.shape

    for i in range(size):
        for j in range(size):
            if A[i][j] == 0.0 and B[i][j] == 0.0:
                continue
            T += 1
            if A[i][j] == 0.0 and B[i][j] != 0.0:
                D += 1
            elif A[i][j] != 0.0 and B[i][j] == 0.0:
                D += 1
    return D/T

def main():
    process_data(num=120)




if __name__ == '__main__':
    #main()
    pass
