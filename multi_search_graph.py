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
from sklearn.metrics import classification_report,confusion_matrix
import copy
import matplotlib.pyplot as plt


# # Graph ConvNet LeNet5
# ### Layers: CL32-MP4-CL64-MP4-FC512-FC10

class Train:
    def __init__(self):


        # parameters
        self.binary = False # true for binary classification; false for 4-classification
        self.num = 120 # options: 120, 180, 69, 264
        self.dataset = 'adni' # options: 'mnist' or 'adni'
        self.train_rate = 0.7 # how to split train/test
        self.thresh = 0.7 # keep edges in the graph with weight > thresh
        self.state = "corr" # 'corr', 'random' or 'eye'
        self.random_seed = None # None or an integer
        self.verbose = False # for feature imp & graph visualization 
        self.layer1 = (144,144)
        self.saved_path = "graph_search/"
        self.CL1_F = 64
        self.CL1_K = 75
        self.CL2_F = 256 # 64
        self.CL2_K = 75
        self.CL3_F = 64
        self.CL3_K = 25
        self.FC1_F = 1024 # 512
        self.FC2_F = 3   #output
        self.D = 0
        self.net_parameters = [self.D, self.CL1_F, self.CL1_K, self.CL2_F, self.CL2_K, self.CL3_F, self.CL3_K, self.FC1_F, self.FC2_F]
        self.learning_rate = 1e-3 # 1e-3
        self.dropout_value = 0.4
        self.l2_regularization = 1e-4
        self.batch_size = 20 #20
        self.num_epochs = 100
        self.top_num = 0


    def prepare(self,dataset):
        # # MNIST
        if dataset == 'mnist':
            mnist = input_data.read_data_sets('datasets', one_hot=False)  # load data in folder datasets/
            train_data = mnist.train.images.astype(np.float32)
            val_data = mnist.validation.images.astype(np.float32)
            test_data = mnist.test.images.astype(np.float32)
            train_labels = mnist.train.labels
            val_labels = mnist.validation.labels
            test_labels = mnist.test.labels

            # Construct graph
            t_start = time.time()
            grid_side = 280
            number_edges = 8
            metric = 'euclidean'
            A = grid_graph(grid_side, number_edges, metric)  # create graph of Euclidean grid
            #print(A.shape)

        elif dataset == 'adni':
            t_start = time.time()
            train_data, test_data, train_labels, test_labels, A = load_data(train_rate=self.train_rate, thresh= self.thresh, binary=self.binary, num=self.num, state=self.state) # 0.35
            #print(train_labels)
            
            #A = np.load(self.saved_path+"76.npy")
            A = np.load("UGA/multi_30_20_3_800_found_graph_left_candidate_465.npy")
            A = scipy.sparse.coo_matrix(A)
            # print(A)
            # print( scipy.sparse.coo_matrix(A))
            old_A = copy.copy(A)
            np.save(self.saved_path+"init_graph.npy",A.toarray())
            A, candidate = self.generate_candidates(A, 0.7 , 0.7)
            

            if  self.verbose:
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


            print(train_data.shape)
            print(test_data.shape)
            #print("baseline: ", sum(test_labels)/test_labels.shape[0])
            

            # grid_side = 180 # 102
            # number_edges = 101
            # metric = 'euclidean'
            # print(A)

        # Compute coarsened graphs

        coarsening_levels = 4
        L, perm = coarsen(A, coarsening_levels)

        #print(L[4])

        
        
        self.layer1 = (L[0].shape)
        self.D = self.layer1[0]
        # print(perm)
        # print(set(perm))
        # Compute max eigenvalue of graph Laplacians
        lmax = []
        for i in range(coarsening_levels+1):
            lmax.append(lmax_L(L[i]))
        #print('lmax: ' + str([lmax[i] for i in range(coarsening_levels+1)]))

        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_labels
        self.test_label = test_labels
        self.old_A = old_A
        self.A = A
        self.candidate = candidate
        # # Reindex nodes to satisfy a binary tree structure
        train_data = perm_data(train_data, perm)
        # val_data = perm_data(val_data, perm)
        test_data = perm_data(test_data, perm)

        #
        # print(train_data.shape)
        # #print(val_data.shape)
        # print(test_data.shape)
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
        
        return train_data, train_labels, test_data, test_labels, L, lmax,A,candidate,old_A


    def train(self, train_data, train_labels, test_data, test_labels, L, lmax,A):
        sys.path.insert(0, 'lib/')

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        
        net_parameters = [self.D, self.CL1_F, self.CL1_K, self.CL2_F, self.CL2_K, self.CL3_F, self.CL3_K, self.FC1_F, self.FC2_F]
        
        # instantiate the object net of the class
        net = Graph_ConvNet_LeNet5(net_parameters)
        if torch.cuda.is_available():
            net.cuda()
        #print(net)

        # Weights
        L_net = list(net.parameters())

        # learning parameters
        learning_rate =self.learning_rate 
        dropout_value =self.dropout_value 
        l2_regularization =self.l2_regularization 
        batch_size =self.batch_size
        num_epochs =self.num_epochs 
        train_size =train_data.shape[0]
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
                if not running_total % 1000:  # print every x mini-batches
                    print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (
                        epoch + 1, running_total, loss_train, acc_train))




            # print
            t_stop = time.time() - t_start
            #print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f %%, time= %.3f, lr= %.5f' %
            #      (epoch + 1, running_loss / running_total, running_accuray / running_total, t_stop, lr))
            if (self.verbose and ((epoch + 0) % 100 == 0)):
                net.feature_importances(L)

            train_acc_list.append(running_accuray/running_total)    

            # update learning rate
            # lr = global_lr * pow(decay, float(global_step / decay_steps))
            if running_loss / running_total < last_loss * 0.5 and flag == True:
                lr = np.maximum(lr * 0.5, 1e-5)
                flag = False
                last = epoch + 1
                last_loss = running_loss / running_total
                print("lr decay to %.5f" %lr)
            
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
            #print('  accuracy(test) = %.3f%%, time= %.3f' % (running_accuray_test / running_total_test, t_stop_test))
            test_acc_list.append(running_accuray_test / running_total_test)

            if self.binary:
                labels = ["NC", "MCI"]
            else:
                #labels = ["NC", "LMCI", "EMCI", "AD"]
                labels = ["NC", "LMCI", "EMCI"]

            if (epoch +1 ) % 100 == 0:
                _, class_predicted = torch.max(y.data, 1)
                print(classification_report(test_y.data.cpu(), class_predicted.cpu(), target_names=labels))
                a = confusion_matrix(test_y.data.cpu(), class_predicted.cpu(),labels=[0,1,2])
                print(a)
        step1_acc = running_accuray_test / running_total_test
        return step1_acc
    
    def set_params(self, param_flag):
        if param_flag == 0:
            self.CL1_F = 32
            self.CL1_K = 25
            self.CL2_F = 64 # 64
            self.CL2_K = 25
            self.CL3_F = 64
            self.CL3_K = 25
            self.FC1_F = 512 # 512
            self.FC2_F = 3
            
        else:
            self.CL1_F = 64
            self.CL1_K = 75
            self.CL2_F = 256 # 64
            self.CL2_K = 75
            self.CL3_F = 64
            self.CL3_K = 25
            self.FC1_F = 1024 # 512
            self.FC2_F = 3   #output


    def generate_candidates(self, A, thresh1, thresh2):
        
        A = A.toarray()
        
        A[A<thresh2] = 0

        
        fat_A = copy.copy(A)
        fat_A[fat_A>0] = 1
        skeleton_A = copy.copy(A)
        skeleton_A[skeleton_A<thresh1] = 0


        #keep the original weight 
        result_A = copy.copy(skeleton_A)
        result_A = scipy.sparse.coo_matrix(result_A)
        skeleton_A[skeleton_A>=thresh1] = 1
        
        skeleton_A = scipy.sparse.coo_matrix(skeleton_A)
        fat_A = scipy.sparse.coo_matrix(fat_A)
        candidate = fat_A!=skeleton_A

        
        #print(candidate)
        candidate = candidate.nonzero()
        candidate_list = []

        #print(candidate[0].shape,candidate[1].shape)
        for i in range(len(candidate[0])):
            if candidate[0][i] < candidate[1][i]:
                candidate_list.append([candidate[0][i],candidate[1][i]])
        print("candidate number:  %d" % len(candidate_list))

        
        return result_A, candidate_list

    def train_top3(self, A):
        acc_list = []
        for i in range(10):
            train_data, test_data, train_label, test_label, L, lmax = self.gene_graph(A)
        
            acc = self.train(train_data, train_label, test_data, test_label, L, lmax,A)
            acc_list.append(acc)
        return acc_list
    def train_one(self, A):
        train_data, test_data, train_label, test_label, L, lmax = self.gene_graph(A)
    
        acc = self.train(train_data, train_label, test_data, test_label, L, lmax,A)
        return acc



    def gene_graph(self, A):
        coarsening_levels = 4
    
        L, perm = coarsen(A, coarsening_levels)
        train_data = perm_data(copy.copy(self.train_data), perm)
        test_data = perm_data(copy.copy(self.test_data), perm)
        self.layer1 = (L[0].shape)
        self.D = self.layer1[0]
        print(self.D)
        lmax = []
        for i in range(coarsening_levels+1):
            lmax.append(lmax_L(L[i]))
        return train_data, test_data,self.train_label,self.test_label, L, lmax
    
    def gene_combs(self,candidate):
        comb_num = 20
        result = []
        for i in range(comb_num):
            comb = self.multichoice(candidate,200)
            if comb not in result:
                result.append(comb)
            #print(comb)
        return result

    def multichoice(self,candidate,num):
        numbers = copy.copy(candidate)
        result = []
        for i in range(num):
            slt_num = random.choice(numbers)
            result.append(slt_num)
            numbers.remove(slt_num)
        return result

    def action_graph(self,action):
        new_A = copy.copy(self.A)
        new_A = new_A.tocsr()
        old_A = self.old_A.toarray()
        #print(action)
        for index in range(len(action)):
            if action[index] == 0:
                continue
            a,b  = self.candidate[index]
            new_A[a,b] = old_A[a,b]
            new_A[b,a] = old_A[b,a]
        new_A = scipy.sparse.coo_matrix(new_A)
        return new_A

    def main(self):
        train_data, train_label, test_data, test_label, L, lmax, A, candidate , old_A = self.prepare(self.dataset)  # mi or minist
        old_A = old_A.toarray()
        #old_A is used to save the weight
        stop_flag = 0
        count = 0
        max_acc = self.train_top3(A)
        print("max_acc : %.3f"%max_acc)
        #max_acc = 0
        while stop_flag == 0:
            max_candidate = [0,0]
            max_A = A
            
            acc_dict = {}
            combs = self.gene_combs(candidate)
            for c_atom in combs:
                count += 1
                new_A = copy.copy(A)
                new_A = new_A.tocsr()

                for a,b in c_atom:
                    new_A[a,b] = old_A[a,b]
                    new_A[b,a] = old_A[a,b]
                new_A = scipy.sparse.coo_matrix(new_A)
                
                acc = self.train_top3(new_A)
                #acc_dict[(c_atom[0],c_atom[1])] = acc
                print("count :  %d  acc: %f"% (count,acc))
                if acc > max_acc:
                    print("better edges.")
                    max_candidate = c_atom
                    max_acc = acc
                    max_A = copy.copy(new_A)

            #end of the loop , select the best candidate and remove the best candidate.
            if max_candidate == [0,0]:
                #which means no better graph here
                stop_flag = 1
                break
            else:
                A = max_A
                
                for atom in max_candidate:
                    candidate.remove(atom)
                # for k,v in acc_dict.items():
                #     if v < 85:
                #         candidate.remove([k[0],k[1]])
                print(max_acc)
                print(max_candidate)
                print("found a new graph, the rest of candidate:  %d" % len(candidate))
                print("saved the new graph")
                np.save(self.saved_path+"found_graph_left_candidate_%d.npy"%len(candidate),max_A.toarray())
        #store the graph 
        np.save(self.saved_path+"found_graph.npy",max_A.toarray())

if __name__ == '__main__':
    #import multi_search_graph.Train
    test = Train()
    #test.main()
    train_data, train_label, test_data, test_label, L, lmax, A, candidate , old_A = test.prepare(test.dataset)
    #0 or 1  default 1
    test.set_params(1)

    #different actions 
    action = [random.choice([0,1]) for _ in range(len(candidate))]
    action = [0 for _ in range(len(candidate))]

    #generate graph and then compute the accs. 
    new_A = test.action_graph(action)
    acc = test.train_top3(new_A)
    print(acc)
    acc = test.train_top3(old_A)
    print(acc)
