# NeuralNetwork

import numpy as np
import Layer
import torch
from copy import deepcopy


class NeuralNetwork(object):

    def __init__(self):
        self.layers = []
        self.W = []
        self.M = []
        self.dWdt = []
        self.dMdt = []
        self.cross_entropy = False
        self.weight_decay = 0.
        self.t = 0.
        self.t_history = []
        self.learning_tau = 2.
        self.learn = False
        self.learn_weights = self.learn
        self.learn_biases = self.learn


    def AddLayer(self, L):
        self.layers.append(L)
        if len(self.layers)>1:
            # from m nodes -> to n nodes
            m = self.layers[-2].n
            n = L.n
            if True:
                self.W.append( torch.randn( n, m )*0.1 ) # forward weights
                self.M.append( torch.randn( m, n )*0.1 ) # backward weights
            else:
                self.W.append( torch.FloatTensor(n, m).zero_() )
                self.M.append( torch.FloatTensor(m, n).zero_() )
            #self.W = deepcopy(self.M)
            # Weight gradients
            self.dWdt.append( torch.FloatTensor(n, m).zero_() )
            self.dMdt.append( torch.FloatTensor(m, n).zero_() )
            self.layers[-1].layer_below = self.layers[-2]
            self.layers[-2].layer_above = self.layers[-1]
            if self.layers[-1].is_top:
                self.W[-1] = torch.eye(n)
                self.M[-1] = torch.eye(n)


    def SetIdentityWeights(self):
        for idx, wm in enumerate(zip(self.W, self.M)):
            w = wm[0]
            m = wm[1]
            nn = self.layers[idx].n
            mm = self.layers[idx+1].n
            for a in range(nn):
                for b in range(mm):
                    w[a,b] = 0.
                    m[b,a] = 0.
            n_min = min(nn, mm)
            for c in range(n_min):
                w[c,c] = 1.
                m[c,c] = 1.


    def SetBias(self, b):
        for layer in self.layers:
            layer.SetBias(b)


    def SetInput(self, x):
        self.layers[0].SetInput(x)

    def SetExpectation(self, x):
        self.layers[-1].SetExpectation(x)

    def Integrate(self):
        for i in range(1, len(self.layers)):
            below_i = self.layers[i-1]
            layer_i = self.layers[i]
            W = self.W[i-1]
            M = self.M[i-1]

            #layer_i.IntegrateFromBelow( self.W[i-1], below_i.Output_Up() )
            layer_i.dvdt = torch.mv(W,below_i.e) * Layer.tanh_p(layer_i.v) - layer_i.e
            #below_i.IntegrateFromAbove( self.M[i-1], layer_i.Output_Down() )
            below_i.dedt = below_i.v - torch.mv(M, Layer.tanh(layer_i.v) ) - below_i.b - torch.mv(below_i.Sigma, below_i.e)

            # Now the weight gradients
            # M first... I think this is the right order. We'll know once we use a 
            # different # of neurons in each layer.
            # Based on equation (29) in "A tutorial on ..." by Bogacz.
            # self.dMdt[i-1] = torch.addr(self.dMdt[i-1],
            #            self.layers[i-1].Output_Up(),
            #            self.layers[i].Output_Down(), alpha=1 )
            self.dMdt[i-1] = torch.addr(torch.zeros_like(self.M[i-1]), below_i.e, Layer.tanh(layer_i.v), alpha=1 )

            # Have to do this for self.W now. Not sure what the equation should be,
            # since it should presumably be different than that for M.

            # And what about the bias? It's stored in the Layer data.
            below_i.dbdt = deepcopy(below_i.e)
            #print(str(np.array(below_i.dbdt)))

        # And process the FF input to the top layer
        self.layers[-1].dvdt = torch.mv(W,self.layers[-2].e) * Layer.tanh_p(self.layers[-1].v)


    def Step(self, dt=0.001):
        k = dt/self.learning_tau
        for i in range(0, len(self.layers)):
            self.layers[i].Step(dt=dt)
        if self.learn:
            for i in range(1, len(self.layers)-1):
                # Update W and M
                if self.learn_weights:
                    #self.M[i-1] = torch.add(self.M[i-1], -k, self.dMdt[i-1])
                    self.M[i-1] += k*self.dMdt[i-1]
                    self.W[i-1] = torch.transpose(self.M[i-1], 0, 1)
                    self.dMdt[i-1].zero_()
            for i in range(0, len(self.layers)-2):
                if self.learn_biases:
                    self.layers[i].b = torch.add(self.layers[i].b, k, self.layers[i].dbdt)
                    #print('Update to bias of layer '+str(i)+' '+str(np.array(self.layers[i].dbdt)))
                    self.layers[i].dbdt.zero_()


    def ShowState(self):
        for idx, layer in enumerate(self.layers):
            if layer.is_input:
                print('Layer '+str(idx)+' (input):')
                layer.ShowState()
                layer.ShowError()
            elif layer.is_top:
                print('Layer '+str(idx)+' (expectation):')
                layer.ShowState()
            else:
                print('Layer '+str(idx)+':')
                layer.ShowState()
                layer.ShowError()


    def ShowWeights(self):
        for idx in range(len(self.layers)-1):
            print('  W'+str(idx)+str(idx+1)+' = ')
            print(str(np.array(self.W[idx])))
            print('  M'+str(idx+1)+str(idx)+' = ')
            print(str(np.array(self.M[idx])))

    def ShowBias(self):
        for idx, layer in enumerate(self.layers):
            layer.ShowBias()

    def Cost(self, target):
        return 0. #self.layer[-1].Cost(target)


    def Record(self):
        '''
        Records the state of the network.
        '''
        self.t_history.append(self.t)
        for layer in self.layers:
            layer.Record()


    def Run(self, T, dt):
        tt = np.arange(self.t, self.t+T, dt)
        for t in tt:
            self.t = t
            self.Integrate()
            self.Step(dt=dt)
            self.Record()






#