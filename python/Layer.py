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
        #self.zv = torch.FloatTensor(self.n).zero_()
        self.e = torch.FloatTensor(self.n).zero_()
        #self.ze = torch.FloatTensor(self.n).zero_()
        self.Sigma = torch.eye(self.n)
        self.type = ''
        self.trans_fcn = 0  # 0=logistic, 1=identity
        self.is_input = False
        self.layer_above = []
        self.layer_below = []
        self.dvdt = torch.FloatTensor(self.n).zero_()
        self.dedt = torch.FloatTensor(self.n).zero_()
        self.tau = 0.05
        self.v_history = []
        self.e_history = []

    def Set(self, v):
        self.v = copy.deepcopy(v)

    def SetBias(self, b):
        for k in range(len(self.b)):
            self.b[k] = b

    def ShowState(self):
        print('  v = '+str(np.array(self.v)))

    def ShowError(self):
        print('  e = '+str(np.array(self.e)))

    def get_v(self):
            return self.v

    def Softmax(self):
        self.v.div( torch.sum(self.v) )
        #self.v = self.v / np.sum(self.v)

    def Output_Up(self):
        return self.e

    def Output_Down(self):
        return logistic( self.v )

    def IntegrateFromBelow(self, W, x):
        '''
        IntegrateFromBelow involves the input to the v nodes.
        phi_dot = theta*eps_u - eps_p
        I've also added a bias term, self.b
        '''
        self.dvdt += torch.mv(W, x) - self.e + self.b

    def IntegrateFromAbove(self, M, x):
        '''
        IntegrateFromAbove involves the input to the e (error) nodes.
        '''
        self.dedt += logistic(self.v) - torch.mv(M, x) - torch.mv(self.Sigma, self.e)

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

    def SetInput(self, x):
        self.v = torch.tensor(copy.deepcopy(x))

    def Step(self, dt=0.001):
        # No update to the state, v
        self.e = torch.add(self.e, dt, self.dedt)
        self.dedt.zero_()

    def Record(self):
        self.e_history.append(np.array(self.e))

def logistic(v):
    return torch.reciprocal( torch.add( torch.exp(-v), 1.) )







#