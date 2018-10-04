# -*- coding: utf-8 -*-
"""
Layer base class

@author: jorchard
"""

import numpy as np
import copy
import torch


class PELayer:
    '''
    Later structure is
       ==> [(v)<-->(e)] <==> [(v)<-->(e)] <==
    '''

    def __init__(self, n=10):
        self.n = n
        # self.v = np.zeros(self.n)
        # self.e = np.zeros(self.n)
        # self.zv = np.zeros(self.n)
        # self.ze = np.zeros(self.n)
        self.v = torch.FloatTensor(self.n).zero_()
        self.b = torch.FloatTensor(self.n).zero_()
        #self.b = torch.tensor([0.2384]*self.n)
        #self.b = torch.tensor([0.2384]*self.n)
        #self.zv = torch.FloatTensor(self.n).zero_()
        self.e = torch.FloatTensor(self.n).zero_()
        #self.ze = torch.FloatTensor(self.n).zero_()
        self.Sigma = torch.eye(self.n)
        self.type = ''
        self.trans_fcn = 0  # 0=logistic, 1=identity
        self.is_input = False
        self.is_top = False
        self.layer_above = []
        self.layer_below = []
        self.dvdt = torch.FloatTensor(self.n).zero_()
        self.dedt = torch.FloatTensor(self.n).zero_()
        self.dbdt = torch.FloatTensor(self.n).zero_()
        self.tau = 0.1
        self.v_history = []
        self.e_history = []

    # def Set(self, v):
    #     self.v = copy.deepcopy(v)

    def SetBias(self, b):
        for k in range(len(self.b)):
            self.b[k] = b[k]

    def ShowState(self):
        print('  v = '+str(np.array(self.v)))

    def ShowError(self):
        print('  e = '+str(np.array(self.e)))

    def ShowBias(self):
        print('  b = '+str(np.array(self.b)))

    def Step(self, dt=0.001):
        k = dt/self.tau
        self.v = torch.add(self.v, k, self.dvdt)
        self.e = torch.add(self.e, k, self.dedt)
        self.dvdt.zero_()
        self.dedt.zero_()

    def Record(self):
        self.v_history.append(np.array(self.v))
        self.e_history.append(np.array(self.e))


class InputPELayer(PELayer):

    def __init__(self, n=10):
        PELayer.__init__(self, n=n)
        self.is_input = True
        self.is_top = False
        self.sensory = torch.zeros(n)  # container for constant input
        self.beta = 1.  # relative weight of FF inputs (vs FB)

    def SetInput(self, x):
        self.sensory = torch.tensor(copy.deepcopy(x)).float()
        #self.v = torch.tensor(copy.deepcopy(x))

    def Step(self, dt=0.001):
        # No update to the state, v
        k = dt/self.tau
        self.v = self.v + k*( self.beta*(self.sensory - self.v) +
                              (1.-self.beta)*self.dvdt )
        self.dvdt.zero_()
        self.e = torch.add(self.e, dt, self.dedt)
        self.dedt.zero_()

    def Record(self):
        self.v_history.append(np.array(self.v))
        self.e_history.append(np.array(self.e))


class TopPELayer(PELayer):

    def __init__(self, n=10):
        PELayer.__init__(self, n=n)
        self.is_top = True
        self.is_input = False
        self.expectation = torch.zeros(n)  # container for constant input
        #self.v = torch.FloatTensor(self.n).zero_()
        #self.dvdt = torch.FloatTensor(self.n).zero_()

        self.beta = 0.  # relative weight of FF inputs (vs FB)

    def SetExpectation(self, x):
        self.expectation = torch.tensor(copy.deepcopy(x)).float()
        #self.v = torch.tensor(copy.deepcopy(x))
        self.e = torch.zeros_like(self.v)

    def Output_Down(self):
        return self.v

    def Step(self, dt=0.01):
        '''
        TopPELayer.Step updates v using a weighted sum of the expectation
        and the input from below.
        '''
        k = dt/self.tau
        self.v = self.v + k*( self.beta*self.dvdt +
                              (1.-self.beta)*(self.expectation - self.v) )
        self.dvdt.zero_()

    def Integrate(self):
        return

    def Record(self):
        self.v_history.append(np.array(self.v))



def logistic(v):
    return torch.reciprocal( torch.add( torch.exp(-v), 1.) )

def logistic_p(v):
    return torch.addcmul( torch.zeros_like(v) , v , torch.neg(torch.add(v, -1)) ) 

def tanh(v):
    return torch.tanh(v)

def tanh_p(v):
    return torch.add( torch.neg( torch.pow( torch.tanh(v),2) ) , 1.)


#