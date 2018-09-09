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
                #self.zv = torch.FloatTensor(self.n).zero_()
                self.e = torch.FloatTensor(self.n).zero_()
                #self.ze = torch.FloatTensor(self.n).zero_()
                self.type = ''
                self.trans_fcn = 0  # 0=logistic, 1=identity
                self.is_input = False
                self.layer_above = []
                self.layer_below = []

        def Set(self, v):
            self.v = copy.deepcopy(v)

        def DisplayState(self):
            print(self.v)

        def get_v(self):
                return self.v

        def Softmax(self):
            self.v.div( torch.sum(self.v) )
            #self.v = self.v / np.sum(self.v)

        def Output_Up(self):
            return self.e

        def Output_Down(self):
            return self.v

        def IntegrateFromBelow(self, below):
            '''
            IntegrateFromBelow involves the input to the v nodes.
            '''
            dvdt += 0



