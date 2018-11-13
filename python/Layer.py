# -*- coding: utf-8 -*-
"""
Layer base class

@author: jorchard
"""

import numpy as np
import copy
import torch

dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda:0") # Uncomment this to run on GPU
else:
    device = torch.device("cpu")
#device = torch.device("cpu")

class PELayer:
    '''
    Later structure is
       ==> [(v)<-->(e)] <==> [(v)<-->(e)] <==
    '''

    def __init__(self, n=0):
        self.n = n  # Number of nodes in layer
        self.idx = []

        # Node activities
        # Allocated dynamically to accommodate different batch sizes
        self.v = [] #torch.zeros(self.n, device=device)
        self.e = [] #torch.zeros(self.n, device=device)
        self.dvdt = [] #torch.FloatTensor(self.n).zero_().to(device)
        self.dedt = [] #torch.FloatTensor(self.n).zero_().to(device)

        # Node biases
        if n>0:
            self.b = torch.zeros(self.n, device=device)
            self.dbdt = torch.FloatTensor(self.n).zero_().to(device)
        else:
            self.b = []
            self.dbdt = []

        # Error node variance for feedback
        #self.Sigma = torch.eye(self.n, dtype=torch.float32).to(device)

        # Misc. parameters
        #self.type = ''
        #self.trans_fcn = 0  # 0=logistic, 1=identity
        self.is_input = False
        self.is_top = False
        self.layer_above = []
        self.layer_below = []

        # self.alpha = torch.tensor(1.).float().to(device)  # forward weight
        # self.beta = torch.tensor(1.).float().to(device)   # backward weight
        self.alpha = 1.
        self.beta = 1.

        self.tau = 0.1 #torch.tensor(0.1).float().to(device)  # Dynamic time constant
        self.probe_on = False
        self.v_history = []
        self.e_history = []
        self.activation_function = 'tanh'
        self.sigma = tanh  # activation function
        self.sigma_p = tanh_p
        self.batch_size = 0

    def SetActivationFunction(self, fcn):
        if fcn=='tanh':
            self.activation_function = 'tanh'
            self.sigma = tanh
            self.sigma_p = tanh_p
        elif fcn=='logistic':
            self.activation_function = 'logistic'
            self.sigma = logistic
            self.sigma_p = logistic_p
        elif fcn=='identity':
            self.activation_function = 'identity'
            self.sigma = identity
            self.sigma_p = identity_p
        elif fcn=='softmax':
            self.activation_function = 'softmax'
            self.sigma = softmax
            self.sigma_p = softmax_p
        else:
            print('Activation function not recognized, using logistic')
            self.activation_function = 'logistic'
            self.sigma = logistic
            self.sigma_p = logistic_p


    def Save(self, fp):
        np.save(fp, self.n)
        np.save(fp, self.is_input)
        np.save(fp, self.is_top)
        np.save(fp, self.activation_function)
        np.save(fp, self.alpha)
        np.save(fp, self.beta)
        np.save(fp, self.tau)
        np.save(fp, self.b)

    def Load(self, fp):
        self.n = np.asscalar( np.load(fp) )
        self.is_input = np.asscalar( np.load(fp) )
        self.is_top = np.asscalar( np.load(fp) )
        self.activation_function = str( np.load(fp) )
        self.SetActivationFunction(self.activation_function)
        self.alpha = torch.tensor( np.load(fp) ).float().to(device)
        self.beta = torch.tensor( np.load(fp) ).float().to(device)
        self.tau = torch.tensor( np.load(fp) ).float().to(device)
        self.b = torch.tensor( np.load(fp) ).float().to(device)
        self.dbdt = torch.zeros_like( self.b ).float().to(device)

    def Allocate(self, batch_size=1):
        if batch_size!=self.batch_size:
            self.batch_size = batch_size
            del self.v, self.e, self.dvdt, self.dedt, self.v_history, self.e_history
            self.v_history = []
            self.e_history = []
            if batch_size>1:
                self.v = torch.zeros([batch_size, self.n], device=device)
                self.e = torch.zeros([batch_size, self.n], device=device)
                self.dvdt = torch.zeros([batch_size, self.n], device=device)
                self.dedt = torch.zeros([batch_size, self.n], device=device)
            else:
                self.v = torch.zeros([self.n], device=device)
                self.e = torch.zeros([self.n], device=device)
                self.dvdt = torch.zeros([self.n], device=device)
                self.dedt = torch.zeros([self.n], device=device)

    def Release(self):
        del self.v, self.e, self.dvdt, self.dedt, self.b

    def SetBias(self, b):
        for k in range(len(self.b)):
            self.b[k] = b[k]

    def ShowState(self):
        print('  v = '+str(np.array(self.v)))

    def ShowError(self):
        print('  e = '+str(np.array(self.e)))

    def ShowBias(self):
        print('  b = '+str(np.array(self.b)))

    def Reset(self):
        del self.v_history, self.e_history
        self.v_history = []
        self.e_history = []
        if isinstance(self.v, (torch.Tensor)):
            self.v.zero_()
        if isinstance(self.e, (torch.Tensor)):
            self.e.zero_()

    def SetFF(self):
        self.alpha = 1. #torch.tensor(1.).float().to(device)
        self.beta =  0. #torch.tensor(0.).float().to(device)

    def SetFB(self):
        self.alpha = 0. #torch.tensor(0.).float().to(device)
        self.beta =  1. #torch.tensor(1.).float().to(device)

    def SetBidirectional(self):
        self.alpha = 1. #torch.tensor(1.).float().to(device)
        self.beta =  1. #torch.tensor(1.).float().to(device)

    def Step(self, dt=0.01):
        k = dt/self.tau
        self.v = torch.add(self.v, k, self.dvdt)
        self.e = torch.add(self.e, k, self.dedt)
        self.dvdt.zero_()
        self.dedt.zero_()

    def Probe(self, bool):
        self.probe_on = bool

    def Record(self):
        if self.probe_on:
            self.v_history.append(np.array(self.v))
            self.e_history.append(np.array(self.e))


#***************************************************
#
#  InputPELayer
#
#***************************************************
class InputPELayer(PELayer):

    def __init__(self, n=10):
        PELayer.__init__(self, n=n)
        self.is_input = True
        self.is_top = False
        #self.sensory = [] #torch.zeros(n).to(device)  # container for constant input

    def Allocate(self, batch_size=1):
        old_batch_size = self.batch_size
        PELayer.Allocate(self, batch_size=batch_size)

    def SetInput(self, x):
        #self.sensory = torch.tensor(copy.deepcopy(x)).float().to(device)
        self.v = torch.tensor(copy.deepcopy(x)).float().to(device)

    def Step(self, dt=0.01):
        k = dt/self.tau
        self.e = torch.add(self.e, k, self.dedt)
        self.dedt.zero_()

    def Record(self):
        self.v_history.append(np.array(self.v))
        self.e_history.append(np.array(self.e))


#***************************************************
#
#  TopPELayer
#
#***************************************************
class TopPELayer(PELayer):

    def __init__(self, n=10):
        PELayer.__init__(self, n=n)
        self.is_top = True
        self.is_input = False
        self.expectation = [] #torch.zeros(n).float().to(device)  # container for constant input
        self.beta = torch.tensor(0., dtype=torch.float32).to(device)  # relative weight of FF inputs (vs FB)
        self.sigma = softmax
        self.sigma_p = softmax_p

    def Allocate(self, batch_size=1):
        old_batch_size = self.batch_size
        PELayer.Allocate(self, batch_size=batch_size)
        if batch_size!=old_batch_size:
            if old_batch_size!=0:
                del self.expectation
            if batch_size==1:
                self.expectation = torch.zeros([self.n], dtype=torch.float, device=device)
            else:
                self.expectation = torch.zeros([batch_size, self.n], dtype=torch.float, device=device)

    def SetExpectation(self, t):
        self.expectation = torch.tensor(copy.deepcopy(t)).float().to(device)

    def Record(self):
        self.v_history.append(np.array(self.v))


#***************************************************
#
#  TopAugPELayer
#
#***************************************************
# class TopAugPELayer(PELayer):

#     def __init__(self, n=10, a=30):
#         PELayer.__init__(self, n=n)
#         self.a = a  # This is the number of augmenting neurons
#         self.va = []  # array of augmenting nodes
#         self.dvadt = []  # and their derivatives
#         self.is_topAug = True
#         self.is_top = False
#         self.is_input = False
#         self.expectation = [] #torch.zeros(n).float().to(device)  # container for constant input
#         self.beta = torch.tensor(0., dtype=torch.float32).to(device)  # relative weight of FF inputs (vs FB)
#         # The classification nodes use sigma
#         self.sigma = softmax
#         self.sigma_p = softmax_p
#         # The augmenting nodes use Aug_sigma
#         self.Aug_sigma = tanh
#         self.Aug_sigma_p = tanh_p
#         self.Ma

#     def Allocate(self, batch_size=1):
#         old_batch_size = self.batch_size
#         PELayer.Allocate(self, batch_size=batch_size)
#         if batch_size!=old_batch_size:
#             del self.expectation, self.va
#             if batch_size==1:
#                 self.expectation = torch.zeros([self.n], dtype=torch.float, device=device)
#                 self.va = torch.zeros([self.a], dtype=torch.float, device=device)
#             else:
#                 self.expectation = torch.zeros([batch_size, self.n], dtype=torch.float, device=device)
#                 self.va = torch.zeros([batch_size, self.a], dtype=torch.float, device=device)

#     def SetExpectation(self, t):
#         self.expectation = torch.tensor(copy.deepcopy(t)).float().to(device)

#     def Record(self):
#         self.v_history.append(np.array(self.v))



def logistic(v):
    return torch.reciprocal( torch.add( torch.exp(-v), 1.) )

def logistic_p(v):
    lv = logistic(v)
    return lv*(1.-lv)
    #return torch.addcmul( torch.zeros_like(v) , lv , torch.neg(torch.add(lv, -1)) ) 

def tanh(v):
    return torch.tanh(v)

def tanh_p(v):
    return 1. - torch.pow(torch.tanh(v),2)
    #return torch.add( torch.neg( torch.pow( torch.tanh(v),2) ) , 1.)

def identity(v):
    return v

def identity_p(v):
    return torch.ones_like(v)

def softmax(v):
    z = torch.exp(v)
    if len(np.shape(z))==1:
        s = torch.sum(z)
        return z/s
    else:
        s = torch.sum(z, dim=1)
        return z/s[np.newaxis,:].transpose(1,0).repeat([1,np.shape(v)[1]])

def softmax_p(v):
    z = softmax(v)
    return z*(1.-z)



#