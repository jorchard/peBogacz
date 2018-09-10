# NeuralNetwork

import numpy as np
import Layer
import torch


class NeuralNetwork(object):

    def __init__(self):
        self.layers = []
        self.W = []
        self.M = []
        self.cross_entropy = False
        self.weight_decay = 0.

    def AddLayer(self, L):
        self.layers.append(L)
        if len(self.layers)>1:
            # from m nodes -> to n nodes
            m = self.layers[-2].n
            n = L.n
            if False:
                self.W.append( torch.randn( n, m ) ) # forward weights
                self.M.append( torch.randn( m, n ) ) # backward weights
            else:
                self.W.append( torch.FloatTensor(n, m).zero_() )
                self.M.append( torch.FloatTensor(m, n).zero_() )
            self.layers[-1].layer_below = self.layers[-2]
            self.layers[-2].layer_above = self.layers[-1]


    def Integrate(self):
        for i in range(1, len(self.layers)):
            below_i = self.layers[i-1]
            layer_i = self.layers[i]
            layer_i.IntegrateFromBelow( self.W[i-1], below_i.Output_Up() )
            below_i.IntegrateFromAbove( self.M[i-1], layer_i.Output_Down() )
            print(i) #self.layer[i]

    def Step(self, dt=0.001):
        for i in range(0, len(self.layers)):
            self.layers[i].Step(dt=dt)
            print(i) #self.layer[i].Step(dt=dt)



    def Cost(self, target):
        return 0. #self.layer[-1].Cost(target)









#