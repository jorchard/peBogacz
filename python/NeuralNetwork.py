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
            self.W.append( torch.randn( m, n ) ) # forward weights
            self.M.append( torch.randn( m, n ) ) # backward weights
            self.layers[-1].layer_below = self.layers[-2]
            self.layers[-2].layer_above = self.layers[-1]


    def Integrate(self):
        for i in range(1, len(self.layers)):
            b = self.layers[i-1]
            a = self.layers[i]
            a.IntegrateFromBelow( torch.mv( self.W, b.Output_Up()) )
            b.IntegrateFromAbove( torch.vm( self.M, a.Output_Down()) )
            print(i) #self.layer[i]

    def Step(self, dt=0.001):
        for i in range(0, len(self.layers)):
            print(i) #self.layer[i].Step(dt=dt)



    def Cost(self, target):
        return 0. #self.layer[-1].Cost(target)









#