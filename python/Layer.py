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
        self.variance = 1. #torch.eye(self.n, dtype=torch.float32).to(device)

        # Misc. parameters
        #self.type = ''
        #self.trans_fcn = 0  # 0=logistic, 1=identity
        self.is_input = False
        self.is_top = False
        self.is_rf = False
        self.layer_above = []
        self.layer_below = []

        # self.alpha = torch.tensor(1.).float().to(device)  # forward weight
        # self.beta = torch.tensor(1.).float().to(device)   # backward weight
        # alpha and beta control the influence of inputs to the state node (v)
        #  -> alpha controls the input being mapped by W from the adjacent layer
        #  -> beta controls the input from this layer's error nodes
        self.alpha = torch.ones(self.n, device=device)  # FF weight (-> v from e below)
        self.beta = torch.ones(self.n, device=device)   # FB influence (v<- from corresponding e)
        self.Sigma = 1.
        self.v_decay = 0.

        self.tau = 0.1 #torch.tensor(0.1).float().to(device)  # Dynamic time constant
        self.probe_on = False
        self.v_history = []
        self.e_history = []
        self.batch_size = 0


    def Save(self, fp):
        np.save(fp, self.n)
        np.save(fp, self.is_input)
        np.save(fp, self.is_top)
        np.save(fp, self.is_rf)
        np.save(fp, self.variance)
        #np.save(fp, self.activation_function)
        np.save(fp, self.alpha.cpu())
        np.save(fp, self.beta.cpu())
        np.save(fp, self.tau.cpu())
        np.save(fp, self.b.cpu())
        

    def Load(self, fp):
        self.n = np.asscalar( np.load(fp) )
        self.is_input = np.asscalar( np.load(fp) )
        self.is_top = np.asscalar( np.load(fp) )
        self.is_rf = np.asscalar( np.load(fp))
        self.variance = np.asscalar( np.load (fp) )
        # self.activation_function = str( np.load(fp) )
        # self.SetActivationFunction(self.activation_function)
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
        del self.v, self.e, self.dvdt, self.dedt, self.b, self.variance

    def SetBias(self, b):
        for k in range(len(self.b)):
            self.b[k] = b[k]

    def ShowState(self):
        print('  v = '+str(np.array(self.v)))

    def ShowError(self):
        print('  e = '+str(np.array(self.e)))

    def ShowBias(self):
        print('  b = '+str(np.array(self.b)))

    def PEError(self):
        return torch.sum(self.e**2) / self.batch_size / self.n

    def Reset(self):
        del self.v_history, self.e_history
        self.v_history = []
        self.e_history = []
        if isinstance(self.v, (torch.Tensor)):
            self.v.zero_()
        if isinstance(self.e, (torch.Tensor)):
            self.e.zero_()

    def Clamp(self):
        self.alpha = torch.zeros(self.n).float().to(device)
        self.beta  = torch.zeros(self.n).float().to(device)
        
    def SetFF(self):
        self.alpha = torch.ones(self.n).float().to(device)
        self.beta  = torch.zeros(self.n).float().to(device)

    def SetFB(self):
        self.alpha = torch.zeros(self.n).float().to(device)
        self.beta  = torch.ones(self.n).float().to(device)

    def SetBidirectional(self):
        self.alpha = torch.ones(self.n).float().to(device)
        self.beta  = torch.ones(self.n).float().to(device)

    def SetFixed(self):
        self.alpha = torch.zeros(self.n).float().to(device)
        self.beta  = torch.zeros(self.n).float().to(device)

    def SetVariance(self, variance=1.0):
        self.variance = variance
        
    def Step(self, dt=0.01):
        k = dt/self.tau
        self.v += k * self.dvdt
        self.e += k * self.dedt
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
        self.is_rf = False
        #self.sensory = [] #torch.zeros(n).to(device)  # container for constant input

    def Allocate(self, batch_size=1):
        old_batch_size = self.batch_size
        PELayer.Allocate(self, batch_size=batch_size)

    def SetInput(self, x):
        #self.sensory = torch.tensor(copy.deepcopy(x)).float().to(device)
        self.v = x.clone().detach()

    # def Step(self, dt=0.01):
    #     k = dt/self.tau
    #     self.v = torch.add(self.v, k, self.dvdt)
    #     self.e = torch.add(self.e, k, self.dedt)
    #     self.dvdt.zero_()
    #     self.dedt.zero_()

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
        self.is_rf = False
        self.expectation = [] #torch.zeros(n).float().to(device)  # container for constant input
        self.beta = torch.zeros(self.n, device=device)   # FB influence (v<- from corresponding e)
        # self.sigma = softmax
        # self.sigma_p = softmax_p

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

    def Reset(self):
        PELayer.Reset(self)
        if isinstance(self.expectation, (torch.Tensor)):
            self.expectation.zero_()

    def SetExpectation(self, t):
        self.expectation = t.clone().detach()

    def Record(self):
        self.v_history.append(np.array(self.v))

#***************************************************
#
#  RFLayer
#
#***************************************************
class RFLayer(PELayer):

    def __init__(self, scale, RF, prev_dim):
        '''
        Initializes a receptive field layer based on the scaling from input, receptive field size, 
        and dimensions from the previous layer.

        scale: the amount to scale from the input, or previous, layer (ex. 2)
        RF: receptive field dimension (assumed square)
        prev_dim: a list of dimensions of the previous layer
        '''
        self.scale = scale

        #Calculate number of neurons in this layer
        self.side_dim = scale*prev_dim[0] #Assume square previous layer for now
        self.n = self.side_dim**2
        PELayer.__init__(self, n=self.n)

        #Size of receptive field
        self.RF = RF 

        #Other parameters, may or may not be used 
        self.is_top = False
        self.is_input = False
        self.is_rf = True

    def Allocate(self, batch_size=1):
        old_batch_size = self.batch_size

        if batch_size != old_batch_size:
            self.batch_size = batch_size
            del self.v, self.e, self.dvdt, self.dedt, self.v_history, self.e_history
            self.v_history = []
            self.e_history = []

            if batch_size>1:
                #First, create vector representation of the layer
                self.v = torch.zeros([batch_size, self.n], device=device)
                self.e = torch.zeros([batch_size, self.n], device=device)
                self.dvdt = torch.zeros([batch_size, self.n], device=device)
                self.dedt = torch.zeros([batch_size, self.n], device=device)

                #Create matrix representation of the layer
                self.v_grid = torch.reshape(self.v, (batch_size, self.side_dim, self.side_dim))
                self.e_grid = torch.reshape(self.e, (batch_size, self.side_dim, self.side_dim))
                self.dvdt_grid = torch.reshape(self.dvdt, (batch_size, self.side_dim, self.side_dim))
                self.dedt_grid = torch.reshape(self.dedt, (batch_size, self.side_dim, self.side_dim))

            else:
                #Vector representation
                self.v = torch.zeros([self.n], device=device)
                self.e = torch.zeros([self.n], device=device)
                self.dvdt = torch.zeros([self.n], device=device)
                self.dedt = torch.zeros([self.n], device=device)

                #Matrix representation
                self.v_grid = torch.reshape(self.v, (self.side_dim, self.side_dim))
                self.e_grid = torch.reshape(self.e, (self.side_dim, self.side_dim))
                self.dvdt_grid = torch.reshape(self.dvdt, (self.side_dim, self.side_dim))
                self.dedt_grid = torch.reshape(self.dedt, (self.side_dim, self.side_dim))

    def RFUpdate(self, val, state=True):
        '''
        Sets the .v to 'val' if state=True, otherwise sets the .e to 'val'. Basically an update layer function.

        val: the value to update .v or .e to
        state: updates .v if True, else updates .e

        Undefined function. I may decide to handle this in NeuralNetwork instead.
        '''

        #Apply the update .v or .e

        #Apply the update to .v_grid or .e_grid

        return 0


    def Reset(self):
        PELayer.Reset(self)

        if isinstance(self.e_grid, (torch.Tensor)):
            self.e_grid.zero_()
        if isinstance(self.v_grid, (torch.Tensor)):
            self.v_grid.zero_()

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






#