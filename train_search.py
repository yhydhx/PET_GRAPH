# coding: utf-8

# # Graph Convolutional Neural Networks
# ## Graph LeNet5 with PyTorch
# ### Xavier Bresson, Oct. 2017

# Implementation of spectral graph ConvNets<br>
# Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering<br>
# M Defferrard, X Bresson, P Vandergheynst<br>
# Advances in Neural Information Processing Systems, 3844-3852, 2016<br>
# ArXiv preprint: [arXiv:1606.09375](https://arxiv.org/pdf/1606.09375.pdf) <br>

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb  # pdb.set_trace()
import collections
import time
import numpy as np
import os
import sys
# from tensorflow.examples.tutorials.mnist import input_data
from lib.grid_graph import grid_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L
from lib.coarsening import perm_data
from lib.coarsening import rescale_L
from network import Graph_ConvNet_LeNet5
from util import *
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


# # Graph ConvNet LeNet5
# ### Layers: CL32-MP4-CL64-MP4-FC512-FC10


# parameters
binary = False # true for binary classification; false for 4-classification
num = 120 # options: 120, 180, 69, 264
dataset = 'adni' # options: 'mnist' or 'adni'
train_rate = 0.7 # how to split train/test
thresh = 0.7 # keep edges in the graph with weight > thresh
state = "corr" # 'corr', 'random' or 'eye'
random_seed = None # None or an integer
verbose = False # for feature imp & graph visualization 
layer1 = (144,144)
saved_path = "1/120_RD_2_1"

def prepare(dataset):
    # # MNIST
    if dataset == 'mnist':
        mnist = input_data.read_data_sets('datasets', one_hot=False)  # load data in folder datasets/
        train_data = mnist.train.images.astype(np.float32)
        val_data = mnist.validation.images.astype(np.float32)
        test_data = mnist.test.images.astype(np.float32)
        train_labels = mnist.train.labels
        val_labels = mnist.validation.labels
        test_labels = mnist.test.labels
        print(train_data.shape)

        print(train_labels.shape)
        print(val_data.shape)
        print(val_labels.shape)
        print(test_data.shape)
        print(test_labels.shape)

        # Construct graph
        t_start = time.time()
        grid_side = 280
        number_edges = 8
        metric = 'euclidean'
        A = grid_graph(grid_side, number_edges, metric)  # create graph of Euclidean grid
        print(A.shape)

    elif dataset == 'adni':
        t_start = time.time()
        train_data, test_data, train_labels, test_labels, A = load_data(train_rate=train_rate, thresh=thresh, binary=binary, num=num, state=state) # 0.35
        #A = np.load(saved_path+"_graph_A_2.npy")
        #A = np.load(saved_path+"_init_graph.npy")
        #A = np.load("UTA/multi_30_20_3_300_found_graph.npy")
        #A = np.load("UTA/multi_30_20_3_300_found_graph.npy")
        #A = np.load("UTA/multi_30_20_3_800_found_graph_left_candidate_465.npy")
        #A = scipy.sparse.coo_matrix(A)
        # print(A)
        # print( scipy.sparse.coo_matrix(A))
        #np.save(saved_path+"_init_graph.npy",A.toarray())

        # print("data shape ====")
        # print(train_data.shape)
        # print(train_data[0][0])
        # print(train_labels.shape)
        # print(test_data.shape)
        # print(test_labels.shape)
        # print("data shape end")
        if verbose:
            fig = plt.figure()
            #定义画布为1*1个划分，并在第1个位置上进行作图
            ax = fig.add_subplot(111)
            #定义横纵坐标的刻度
            # ax.set_yticks(range(len(yLabel)))
            # ax.set_yticklabels(yLabel, fontproperties=font)
            # ax.set_xticks(range(len(xLabel)))
            # ax.set_xticklabels(xLabel)
            #作图并选择热图的颜色填充风格，这里选择hot
            #print("A===")
            #print(A)
            im = ax.imshow(A.toarray(), cmap=plt.cm.hot_r)
            #增加右侧的颜色刻度条
            plt.colorbar(im)
            #增加标题
            plt.title("This is the original graph")
            #show
            plt.show()


        # print(train_data.shape)
        # print(test_data.shape)
        # print("baseline: ", sum(test_labels)/test_labels.shape[0])
        

        # grid_side = 180 # 102
        # number_edges = 101
        # metric = 'euclidean'
        # print(A)

    # Compute coarsened graphs

    coarsening_levels = 4
    L, perm = coarsen(A, coarsening_levels)

    #print(L)
    
    global layer1
    layer1 = (L[0].shape)
    #print(perm)
    #print(set(perm))
    # Compute max eigenvalue of graph Laplacians
    lmax = []
    for i in range(coarsening_levels):
        lmax.append(lmax_L(L[i]))
    #print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

    # # Reindex nodes to satisfy a binary tree structure
    train_data = perm_data(train_data, perm)
    # val_data = perm_data(val_data, perm)
    test_data = perm_data(test_data, perm)
    #
    #print(train_data.shape)
    #print(val_data.shape)
    #print(test_data.shape)
    '''
    test part for update graph
    '''
    #a,b = update_graph(A)
    #print(a[0].shape == (144,144))
    #exit()
    '''
    test part for update graph
    '''
    #print('Execution time: {:.2f}s'.format(time.time() - t_start))
    del perm
    return train_data, train_labels, test_data, test_labels, L, lmax,A


def train(train_data, train_labels, test_data, test_labels, L, lmax,A,learning_rate):
    sys.path.insert(0, 'lib/')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    #print(A)
    #print("available")
    #exit()
    if torch.cuda.is_available():
        #print('cuda available')
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        torch.cuda.manual_seed(1)
    else:
        #print('cuda not available')
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        torch.manual_seed(1)

    # network parameters
    D = train_data.shape[1]
    CL1_F = 32
    CL1_K = 25
    CL2_F = 64 # 64
    CL2_K = 25
    CL3_F = 64
    CL3_K = 25
    FC1_F = 512 # 512
    FC2_F = 3

    CL1_F = 64
    CL1_K = 75
    CL2_F = 256 # 64
    CL2_K = 75
    CL3_F = 64
    CL3_K = 25
    FC1_F = 1024 # 512
    FC2_F = 3
    net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, CL3_F, CL3_K, FC1_F, FC2_F]

    # instantiate the object net of the class
    net = Graph_ConvNet_LeNet5(net_parameters)
    if torch.cuda.is_available():
        net.cuda()
    #print(net)

    # Weights
    L_net = list(net.parameters())

    # learning parameters
    
    dropout_value = 0.4
    l2_regularization = 1e-4
    batch_size = 20 #20
    num_epochs = 300
    train_size = train_data.shape[0]
    nb_iter = int(num_epochs * train_size) // batch_size
    #print('num_epochs=', num_epochs, ', train_size=', train_size, ', nb_iter=', nb_iter)

    # Optimizer
    global_lr = learning_rate
    global_step = 0
    decay = 0.1
    decay_steps = train_size
    lr = learning_rate
    optimizer = net.update(lr)

    # loop over epochs
    indices = collections.deque()
    train_acc_list = []
    test_acc_list = []
    last = 0 
    last_loss = 0.6
    flag = True
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        # reshuffle
        indices.extend(np.random.permutation(train_size))  # rand permutation

        # reset time
        t_start = time.time()

        # extract batches
        running_loss = 0.0
        running_accuray = 0
        running_total = 0
        while len(indices) >= batch_size:

            # extract batches
            batch_idx = [indices.popleft() for i in range(batch_size)]
            train_x, train_y = train_data[batch_idx, :], train_labels[batch_idx]
            train_x = Variable(torch.FloatTensor(train_x).type(dtypeFloat), requires_grad=False)
            train_y = train_y.astype(np.int64)
            train_y = torch.LongTensor(train_y).type(dtypeLong)
            train_y = Variable(train_y, requires_grad=False)

            # Forward
            y = net.forward(train_x, dropout_value, L, lmax)
            loss = net.loss(y, train_y, l2_regularization)
            loss_train = loss.data

            # Accuracy
            acc_train = net.evaluation(y, train_y.data)
            #print(train_y)

            # backward
            loss.backward()

            # Update
            global_step += batch_size  # to update learning rate
            optimizer.step()
            optimizer.zero_grad()

            # loss, accuracy
            running_loss += loss_train
            running_accuray += acc_train
            running_total += 1

            # print
            if not running_total % 100:  # print every x mini-batches
                print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (
                    epoch + 1, running_total, loss_train, acc_train))




        # print
        t_stop = time.time() - t_start
        # print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f %%, time= %.3f, lr= %.5f' %
        #       (epoch + 1, running_loss / running_total, running_accuray / running_total, t_stop, lr))
        if (verbose and ((epoch + 0) % 100 == 0)):
            net.feature_importances(L)

        train_acc_list.append(running_accuray/running_total)    

        # update learning rate
        # lr = global_lr * pow(decay, float(global_step / decay_steps))
        if running_loss / running_total < last_loss * 0.5 and flag == True:
            lr = np.maximum(lr * 0.5, 1e-5)
            flag = False
            last = epoch + 1
            last_loss = running_loss / running_total
            #print("lr decay to %.5f" %lr)
        
        if flag == False and last + 150 < epoch:
            flag = True

        optimizer = net.update_learning_rate(optimizer, lr)

        # Test set
        running_accuray_test = 0
        running_total_test = 0
        indices_test = collections.deque()
        indices_test.extend(range(test_data.shape[0]))
        t_start_test = time.time()
        # while len(indices_test) >= 1:
        batch_idx_test = [indices_test.popleft() for i in range(len(indices_test))]
        # print(len(batch_idx_test))
        test_x, test_y = test_data[batch_idx_test, :], test_labels[batch_idx_test]
        test_x = Variable(torch.FloatTensor(test_x).type(dtypeFloat), requires_grad=False)
        y = net.forward(test_x, 0.0, L, lmax)
        test_y = test_y.astype(np.int64)
        test_y = torch.LongTensor(test_y).type(dtypeLong)
        test_y = Variable(test_y, requires_grad=False)
        acc_test = net.evaluation(y, test_y.data)
        running_accuray_test += acc_test
        running_total_test += 1
        t_stop_test = time.time() - t_start_test
        print('  accuracy(test) = %.3f%%, time= %.3f' % (running_accuray_test / running_total_test, t_stop_test))
        test_acc_list.append(running_accuray_test / running_total_test)

        if binary:
            labels = ["NC", "MCI"]
        else:
            #labels = ["NC", "LMCI", "EMCI", "AD"]
            labels = ["NC", "LMCI", "EMCI"]

        if (epoch + 1) % 100 == 0:
            _, class_predicted = torch.max(y.data, 1)
            print(classification_report(test_y.data.cpu(), class_predicted.cpu(), target_names=labels))
    step1_acc = running_accuray_test / running_total_test
 
    '''
    nwe graph ended
    '''    

    length = len(train_acc_list)
    # e = [i for i in range(1, length+1)]
    x = np.arange(1, length + 1)
    l1 = plt.plot(x, train_acc_list, 'r--', label='train')
    l2 = plt.plot(x, test_acc_list, 'g--', label='test')
    plt.title('The training and test accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim((50, 102))
    plt.legend()
    #plt.show()
    #print(L)

    if  verbose:
        fig = plt.figure()
        #定义画布为1*1个划分，并在第1个位置上进行作图
        ax = fig.add_subplot(111)
        #定义横纵坐标的刻度
        #ax.set_yticks(range(len(yLabel)))
        #ax.set_yticklabels(yLabel, fontproperties=font)
        #ax.set_xticks(range(len(xLabel)))
        #ax.set_xticklabels(xLabel)
        #作图并选择热图的颜色填充风格，这里选择hot
        #print("A===")
        #print(A)
        im = ax.imshow(A.toarray(), cmap=plt.cm.hot_r)
        #增加右侧的颜色刻度条
        plt.colorbar(im)
        #增加标题
        plt.title("This is the original graph")
        #show
        plt.show()

    return step1_acc



def main():
    acc_list = []

    learning_rate = 1e-4
    for i in range(10):
        train_data, train_label, test_data, test_label, L, lmax,A = prepare(dataset)  # mi or minist
        acc = train(train_data, train_label, test_data, test_label, L, lmax,A,learning_rate)
        acc_list.append(acc)
    print(acc_list)

    learning_rate = 1e-3
    for i in range(10):
        train_data, train_label, test_data, test_label, L, lmax,A = prepare(dataset)  # mi or minist
        acc = train(train_data, train_label, test_data, test_label, L, lmax,A,learning_rate)
        acc_list.append(acc)
    print(acc_list)

    learning_rate = 1e-2
    for i in range(10):
        train_data, train_label, test_data, test_label, L, lmax,A = prepare(dataset)  # mi or minist
        acc = train(train_data, train_label, test_data, test_label, L, lmax,A,learning_rate)
        acc_list.append(acc)
    print(acc_list)

    learning_rate = 0.1
    for i in range(10):
        train_data, train_label, test_data, test_label, L, lmax,A = prepare(dataset)  # mi or minist
        acc = train(train_data, train_label, test_data, test_label, L, lmax,A,learning_rate)
        acc_list.append(acc)
    print(acc_list)


if __name__ == '__main__':
    main()
