# NeuralNetwork

import numpy as np
import Layer
import torch
from copy import deepcopy

dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda:0") # Uncomment this to run on GPU
else:
    device = torch.device("cpu")
#device = torch.device("cpu")

def OneHot(z):
    y = np.zeros(np.shape(z))
    y[np.argmax(z)] = 1.
    return y



class Connection(object):

    def __init__(self, b, a, W=None, M=None):
        '''
        Connection(b, a, W=None, M=None)

        Create a connection between two layers.

        Inputs:
          b is the layer (object) below
          a is the later (object) abover
          W and M can be supplied (optional)
        '''
        self.below = b
        self.above = a

        self.below_idx = self.below.idx
        self.above_idx = self.above.idx

        Wgiven = hasattr(W, '__len__')
        Mgiven = hasattr(M, '__len__')

        if Wgiven:
            self.W = torch.tensor(W).float().to(device)
        else:
            self.W = torch.randn( b.n, a.n, dtype=torch.float32, device=device)*0.1

        if Mgiven:
            self.M = torch.tensor(M).float().to(device)
        else:
            self.M = torch.randn( a.n, b.n, dtype=torch.float32, device=device)*0.1

        self.dWdt = torch.zeros( b.n, a.n, dtype=torch.float32, device=device)
        self.dMdt = torch.zeros( a.n, b.n, dtype=torch.float32, device=device)




class NeuralNetwork(object):

    def __init__(self):
        self.layers = []
        self.n_layers = 0
        self.connections = []
        #self.W = []
        #self.M = []
        #self.dWdt = []
        #self.dMdt = []
        self.cross_entropy = False
        self.weight_decay = 0.
        self.t = 0.
        self.t_history = []
        self.learning_tau = 2.
        self.learn = False
        self.learn_weights = self.learn
        self.learn_biases = self.learn
        self.batch_size = 0


    def Save(self, fp):
        if type(fp)==str:
            fp = open(fp, 'wb')
        np.save(fp, len(self.layers))
        for k in range(len(self.layers)-1):
            np.save(fp, self.layers[k].b)
            np.save(fp, np.array(self.W[k]))
            np.save(fp, np.array(self.M[k]))
        np.save(fp, self.layers[-1].b)
        fp.close()

    def Load(self, fp):
        if type(fp)==str:
            fp = open(fp, 'rb')
        n_layers = np.load(fp)
        b = np.load(fp)
        w = np.load(fp)
        m = np.load(fp)
        L = Layer.InputPELayer(n=len(b))
        self.AddLayer(L, W=w, M=m)
        self.layers[0].b = torch.tensor(b).float().to(device)
        for k in range(1,n_layers-1):
            b = np.load(fp)
            w = np.load(fp)
            m = np.load(fp)
            L = Layer.PELayer(n=len(b))
            self.AddLayer(L, W=w, M=m)
            self.layers[-1].b = torch.tensor(b).float().to(device)
        b = np.load(fp).copy()
        L = Layer.TopPELayer(n=len(b))
        self.AddLayer(L)
        self.layers[-1].b = torch.tensor(b).float().to(device)
        fp.close()



    def Connect(self, prei, posti, W=None, M=None):
        '''
        Connect(prei, posti, W=None, M=None)

        Connect two layers.

        Inputs:
          prei is the index of the lower layer
          posti is the index of the upper layer
        '''
        preL = self.layers[prei]
        postL = self.layers[posti]
        preL.layer_above = postL
        postL.layer_below = preL
        self.connections.append(Connection(preL, postL, W=W, M=M))


    def AddLayer(self, L):
        self.layers.append(L)
        self.n_layers += 1
        L.idx = self.n_layers-1


    def ConnectNextLayer(self, post):
        self.AddLayer(post)
        self.Connect(self.n_layers-2, self.n_layers-1)


    # def AddLayer(self, L, pre=None, W=None, M=None):
    #     self.layers.append(L)

    #     if len(self.layers)>1:
    #         # from m nodes -> to n nodes
    #         # We assume RIGHT-MULTIPLICATION, as in
    #         #   v * W
    #         m = self.layers[-2].n
    #         n = L.n


    #         if False:
    #             self.W.append( torch.FloatTensor(m, n).zero_().to(device) )
    #             self.M.append( torch.FloatTensor(n, m).zero_().to(device) )
    #         #self.W = deepcopy(self.M)
    #         # Weight gradients
    #         self.dWdt.append( torch.FloatTensor(m, n).zero_().to(device) )
    #         self.dMdt.append( torch.FloatTensor(n, m).zero_().to(device) )
    #         self.layers[-1].layer_below = self.layers[-2]
    #         self.layers[-2].layer_above = self.layers[-1]
    #         # if self.layers[-1].is_top:
    #         #     self.W[-1] = torch.eye(n)
    #         #     self.M[-1] = torch.eye(n)
    #     else:
    #         # first layer
    #         if Wgiven:
    #             self.W.append( torch.tensor(W).float().to(device))
    #         if Mgiven:
    #             self.M.append( torch.tensor(M).float().to(device))


    # def SetIdentityWeights(self):
    #     for idx, wm in enumerate(zip(self.W, self.M)):
    #         w = wm[0]
    #         m = wm[1]
    #         nn = self.layers[idx].n
    #         mm = self.layers[idx+1].n
    #         for a in range(nn):
    #             for b in range(mm):
    #                 w[a,b] = 0.
    #                 m[b,a] = 0.
    #         n_min = min(nn, mm)
    #         for c in range(n_min):
    #             w[c,c] = 1.
    #             m[c,c] = 1.


    def SetBias(self, b):
        for layer in self.layers:
            layer.SetBias(b)


    def Allocate(self, x):
        dims = len(np.shape(x))
        proposed_batch_size = 1
        if dims==2:
            proposed_batch_size = np.shape(x)[0]
        if proposed_batch_size!=self.batch_size:
            self.batch_size = proposed_batch_size
            #print('Allocating')
            for l in self.layers:
                l.Allocate(batch_size=proposed_batch_size)


    def SetInput(self, x):
        self.Allocate(x)
        self.layers[0].SetInput(x)


    def SetExpectation(self, x):
        self.Allocate(x)
        self.layers[-1].SetExpectation(x)


    def Integrate(self):
        # Loop through connections
        # Then loop throug layers

        # First, address only the connections between layers
        for c in self.connections:
            b = c.below
            a = c.above
            # e <-- v
            b.dedt -= a.sigma(a.v)@c.M
            # e --> v
            if a.is_top:
                a.dvdt += a.alpha*(b.e@c.W)*a.sigma_p(a.v)  
            else:
                a.dvdt += (b.e@c.W)*a.sigma_p(a.v)

            if self.batch_size==1:
                c.dMdt += a.sigma(a.v.reshape([a.n,1])) @ b.e.reshape([1,b.n])
                c.dWdt += b.e.reshape([b.n,1]) @ a.sigma(a.v.reshape([1,a.n]))
            else:
                c.dMdt += a.sigma(a.v).transpose(1,0) @ b.e
                c.dWdt += b.e.transpose(1,0) @ a.sigma(a.v)

            b.dbdt += torch.sum(b.e, dim=0)


        # Next, address the connections inside each layer
        for l in self.layers:
            #l.dedt += l.sigma(l.v) - l.b - l.e
            if l.is_input:
                l.dvdt += l.alpha*(l.sensory - l.v) - l.beta*l.e
                l.dedt += l.sigma(l.v) - l.e
            elif l.is_top:
                l.dvdt -= l.beta*l.e
                l.dedt += l.sigma(l.v) - l.expectation - l.e
            else:
                l.dvdt -= l.e
                l.dedt += l.sigma(l.v) - l.e



    # def old_Integrate(self):
    #     # Bottom layer, use beta to balance FF and FB operation
    #     self.layers[0].dvdt = self.layers[0].beta*(self.layers[0].sensory - self.layers[0].v) - (1.-self.layers[0].beta)*self.layers[0].e
    #     for i in range(1, len(self.layers)):
    #         # For i, update:
    #         #   layer[i-1].e
    #         #   layer[i].v
    #         below_i = self.layers[i-1]
    #         layer_i = self.layers[i]
    #         W = self.W[i-1]
    #         M = self.M[i-1]

    #         #layer_i.IntegrateFromBelow( self.W[i-1], below_i.Output_Up() )
    #         if i==len(self.layers)-1:
    #             # Top layer -- use convex combination
    #             layer_i.dvdt = layer_i.beta * (below_i.e@W) * layer_i.sigma_p(layer_i.v) - (1.-layer_i.beta)*layer_i.e
    #         else:
    #             # NOT the top layer
    #             #print(i,np.shape(W))
    #             #print(np.shape(below_i.e))
    #             layer_i.dvdt = (below_i.e@W) * layer_i.sigma_p(layer_i.v) - layer_i.e
    #         #below_i.IntegrateFromAbove( self.M[i-1], layer_i.Output_Down() )

    #         below_i.dedt = below_i.sigma(below_i.v) - layer_i.sigma(layer_i.v)@M - below_i.b - below_i.e #torch.mv(below_i.Sigma, below_i.e)

    #         # Now the weight gradients
    #         # M first...
    #         # Based on equation (29) in "A tutorial on ..." by Bogacz.
    #         if self.batch_size==1:
    #             #self.dMdt[i-1] = torch.addr(torch.zeros_like(self.M[i-1]), below_i.e, layer_i.sigma(layer_i.v), alpha=1 )
    #             self.dMdt[i-1] =  layer_i.sigma(layer_i.v.reshape([layer_i.n,1])) @ below_i.e.reshape([1,below_i.n])
    #         else:
    #             self.dMdt[i-1] = layer_i.sigma(layer_i.v).transpose(1,0) @ below_i.e

    #         # And what about the bias? It's stored in the Layer data.
    #         #below_i.dbdt = deepcopy(below_i.e)
    #         below_i.dbdt = torch.sum(below_i.e, dim=0)
    #         #print(str(np.array(below_i.dbdt)))

    #     # And process the FF input to the top layer
    #     #self.layers[-1].dvdt = torch.mv(W,self.layers[-2].e) * self.layers[-1].sigma_p(self.layers[-1].v)
    #     self.layers[-1].dedt = self.layers[-1].sigma(self.layers[-1].v) - self.layers[-1].expectation - self.layers[-1].e #torch.mv(self.layers[-1].Sigma, self.layers[-1].e)


    def Step(self, dt=0.01):
        k = dt/self.learning_tau
        for l in self.layers:
            l.Step(dt=dt)
            #l.dvdt.zero_()
            #l.dedt.zero_()

        if self.learn:
            for c in self.connections:
                c.M += k*c.dMdt
                c.W += k*c.dWdt
                #c.dMdt.zero_()
                #c.dWdt.zero_()
                c.below.b += k*c.below.dbdt
                #c.below.b.zero_()


    # def old_Step(self, dt=0.001):
    #     k = dt/self.learning_tau
    #     for i in range(0, len(self.layers)):
    #         self.layers[i].Step(dt=dt)
    #     if self.learn:
    #         for i in range(1, len(self.layers)):
    #             # Update W and M
    #             if self.learn_weights:
    #                 #self.M[i-1] = torch.add(self.M[i-1], -k, self.dMdt[i-1])
    #                 self.M[i-1] += k*self.dMdt[i-1]
    #                 self.W[i-1] = torch.transpose(self.M[i-1], 0, 1)
    #                 self.dMdt[i-1].zero_()
    #         for i in range(0, len(self.layers)-1):
    #             if self.learn_biases:
    #                 #******************
    #                 # This might be faster using +=
    #                 #******************
    #                 #self.layers[i].b = torch.add(self.layers[i].b, k, self.layers[i].dbdt)
    #                 self.layers[i].b += k * self.layers[i].dbdt
    #                 #print('Update to bias of layer '+str(i)+' '+str(np.array(self.layers[i].dbdt)))
    #                 self.layers[i].dbdt.zero_()


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

    def Reset(self):
        for l in self.layers:
            l.Reset()

    def ResetGradients(self):
        for l in self.layers:
            l.dvdt.zero_()
            l.dedt.zero_()
            l.dbdt.zero_()
        for c in self.connections:
            c.dWdt.zero_()
            c.dMdt.zero_()

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
            self.ResetGradients()
            self.Integrate()
            self.Step(dt=dt)
            #self.Record()


    def Infer(self, T, x, y):
        self.learn = True
        self.layers[0].SetFF()
        self.layers[-1].SetFB()
        self.SetInput(x)
        self.SetExpectation(y)
        self.Run(T, dt=0.01)

    def Predict(self, T, x):
        self.learn = False
        self.layers[0].SetFF()
        self.layers[-1].SetFF()
        self.SetInput(x)
        self.Run(T, dt=0.01)
        return self.layers[-1].sigma(self.layers[-1].v)

    def Generate(self, T, y):
        self.learn = False
        self.layers[0].SetFB()
        self.layers[-1].SetFB()
        self.SetExpectation(y)
        self.Run(T, dt=0.01)
        return self.layers[0].v



#