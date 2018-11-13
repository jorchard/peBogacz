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



#============================================================
#
# Connection class
#
#============================================================
class Connection(object):

	def __init__(self, b=None, a=None, W=None, M=None):
	    '''
	    Connection(b, a, W=None, M=None)

	    Create a connection between two layers.

	    Inputs:
	      b is the layer (object) below
	      a is the later (object) abover
	      W and M can be supplied (optional)
	    '''
	    ltypes = (Layer.PELayer, Layer.InputPELayer, Layer.TopPELayer)
	    if isinstance(b, ltypes) or isinstance(a, ltypes):
	        self.below = b
	        self.above = a

	        self.below_idx = self.below.idx
	        self.above_idx = self.above.idx

	        Wgiven = hasattr(W, '__len__')
	        Mgiven = hasattr(M, '__len__')

	        if Wgiven:
	            self.W = torch.tensor(W).float().to(device)
	        else:
	            self.W = torch.randn( b.n, a.n, dtype=torch.float32, device=device) / np.sqrt(b.n)

	        if Mgiven:
	            self.M = torch.tensor(M).float().to(device)
	        else:
	            self.M = torch.randn( a.n, b.n, dtype=torch.float32, device=device) / np.sqrt(b.n)

	        self.dWdt = torch.zeros( b.n, a.n, dtype=torch.float32, device=device)
	        self.dMdt = torch.zeros( a.n, b.n, dtype=torch.float32, device=device)
	    else:
	    	self.below = []
	    	self.above = []
	    	self.below_idx = -99
	    	self.above_idx = -99
	    	self.W = []
	    	self.M = []
	    	self.dWdt = []
	    	self.dMdt = []
	    self.learn = True

	def MakeIdentity(self):
		if self.below.n!=self.above.n:
			print('Connection matrix is not square.')
		else:
			self.W = torch.eye(self.below.n).float().to(device)
			self.M = torch.eye(self.below.n).float().to(device)

	def Nonlearning(self):
		self.learn = False
		del self.dWdt, self.dMdt

	def Release(self):
	    self.below = []
	    self.above = []
	    self.below_idx = []
	    self.above_idx = []
	    del self.W, self.M
	    self.W = []
	    self.M = []
	    del self.dMdt, self.dWdt
	    self.dMdt = []
	    self.dWdt = []

	def Save(self, fp):
		np.save(fp, self.below_idx)
		np.save(fp, self.above_idx)
		np.save(fp, self.W)
		np.save(fp, self.M)
		np.save(fp, self.learn)



#============================================================
#
# NeuralNetwork class
#
#============================================================
class NeuralNetwork(object):

	def __init__(self):
	    self.layers = []
	    self.n_layers = 0
	    self.connections = []
	    self.cross_entropy = False
	    self.weight_decay = 0.
	    self.t = 0.
	    self.t_runstart = 0.
	    self.learning_blackout = 1.0 # how many seconds to wait before turning learning on
	    self.t_history = []
	    self.learning_tau = 2.
	    self.learn = False
	    self.learn_weights = True
	    self.learn_biases = True
	    self.batch_size = 0
	    self.probe_on = False


	def Release(self):
	    for l in self.layers:
	        l.Release()
	        del l
	    for c in self.connections:
	        c.Release()
	        del c
	    self.layers = []
	    self.connections = []


	def Save(self, fp):
	    if type(fp)==str:
	        fp = open(fp, 'wb')
	    np.save(fp, len(self.layers))
	    for l in self.layers:
	        l.Save(fp)
	    np.save(fp, len(self.connections))
	    for c in self.connections:
	    	c.Save(fp)
	    fp.close()

	def Load(self, fp):
		if type(fp)==str:
			fp = open(fp, 'rb')
		n_layers = np.asscalar( np.load(fp) )
		self.Release()
		for k in range(n_layers):
			L = Layer.PELayer()
			L.Load(fp)
			if L.is_input:
				L.__class__ = Layer.InputPELayer
			elif L.is_top:
				L.__class__ = Layer.TopPELayer
			self.AddLayer(L)

		n_connections = np.asscalar( np.load(fp) )
		for k in range(n_connections):
			bi = np.asscalar( np.load(fp) ) # below_idx
			ai = np.asscalar( np.load(fp) ) # above_idx
			W = np.load(fp)
			M = np.load(fp)
			learn = np.asscalar( np.load(fp) )
			self.Connect(bi, ai, W=W, M=M)
			self.connections[-1].learn = learn
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


	def SetBias(self, b):
	    for layer in self.layers:
	        layer.SetBias(b)

	def SetTau(self, tau):
		for l in self.layers:
			l.tau = torch.tensor(tau).float().to(device)

	def Allocate(self, x):
	    dims = len(np.shape(x))
	    proposed_batch_size = 1
	    if dims==2:
	        proposed_batch_size = np.shape(x)[0]
	    if proposed_batch_size!=self.batch_size:
	        self.batch_size = proposed_batch_size
	        del self.t_history
	        self.t_history = []
	        self.t = 0.
	        print('Allocating')
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
		    blw = c.below
		    abv = c.above
		    # e <-- v
		    blw.dedt -= abv.sigma(abv.v)@c.M + blw.b
		    # e --> v
		    abv.dvdt += abv.alpha*(blw.e@c.W)*abv.sigma_p(abv.v)
		    # if a.is_top:
		    #     a.dvdt += a.alpha*(b.e@c.W)*a.sigma_p(a.v)
		    # else:
		    #     a.dvdt += (b.e@c.W)*a.sigma_p(a.v)

		    if c.learn==True:
		        if self.batch_size==1:
		            c.dMdt += abv.sigma(abv.v.reshape([abv.n,1])) @ blw.e.reshape([1,blw.n])
		            c.dWdt += blw.e.reshape([blw.n,1]) @ abv.sigma(abv.v.reshape([1,abv.n]))
		        else:
		            c.dMdt += abv.sigma(abv.v).transpose(1,0) @ blw.e
		            c.dWdt += blw.e.transpose(1,0) @ abv.sigma(abv.v)

		        blw.dbdt += torch.sum(blw.e, dim=0)


		# Next, address the connections inside each layer
		for l in self.layers:
		    #l.dedt += l.sigma(l.v) - l.b - l.e
		    if l.is_input:
		    	# The state node is not updated in a bottom input layer
		        #l.dvdt += l.alpha*(l.sensory - l.v) - l.beta*l.e
		        #l.dedt += l.sigma(l.v) - l.e
		        l.dedt += l.v - l.e
		    elif l.is_top:
		        l.dvdt -= l.beta*l.e
		        #l.dedt += l.sigma(l.v) - l.expectation - l.e
		        l.dedt += l.v - l.expectation - l.e
		    else:
		        l.dvdt -= l.beta*l.e
		        #l.dedt += l.sigma(l.v) - l.e
		        l.dedt += l.v - l.e


	def Step(self, dt=0.01):
	    k = dt/self.learning_tau
	    for l in self.layers:
	        l.Step(dt=dt)

	    # Only update weights if learning is one, and we are past
	    # the "blackout" transient period.
	    if self.learn and self.t-self.t_runstart>=self.learning_blackout:
	        for c in self.connections: # The first connection is fixed
	        	if self.learn_weights:
		            c.M += k*c.dMdt
		            c.W += k*c.dWdt
		        if self.learn_biases:
		            c.below.b += k*c.below.dbdt


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
		del self.t_history
		self.t_history = []
		self.t = 0.
		for l in self.layers:
		    l.Reset()

	def ResetGradients(self):
	    for l in self.layers:
	        l.dvdt.zero_()
	        l.dedt.zero_()
	        l.dbdt.zero_()
	    for c in self.connections:
	    	if c.learn:
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
	    if self.batch_size==1:
		    self.t_history.append(self.t)
		    for layer in self.layers:
		        layer.Record()

	def BackprojectExpectation(self, y):
		'''
		Initialize state nodes from the top-layer expection, skipping the
		error nodes.
		'''
		self.Allocate(y)
		self.layers[-1].v = torch.tensor(y).float().to(device)
		for idx in range(self.n_layers-2,0,-1):
			v = self.layers[idx+1].sigma(self.layers[idx+1].v)@self.connections[idx].M + self.layers[idx].b
			self.layers[idx].v = torch.tensor(v).float().to(device)


	def Run(self, T, dt):
		self.probe_on = False
		for l in self.layers:
			if l.probe_on:
				self.probe_on = True

		self.t_runstart = self.t
		tt = np.arange(self.t, self.t+T, dt)
		for t in tt:
		    self.t = t
		    self.ResetGradients()
		    self.Integrate()
		    self.Step(dt=dt)
		    if self.probe_on:
		    	self.Record()


	def Infer(self, T, x, y, dt=0.01, learning=False):
	    self.learn = learning
	    self.layers[1].SetFF()
	    self.layers[-1].SetBidirectional()
	    self.SetInput(x)
	    self.SetExpectation(y)

	    self.Run(T, dt=dt)

	def Predict(self, T, x, dt=0.01):
	    self.learn = False
	    self.layers[1].SetFF()
	    self.layers[-1].SetFF()
	    self.SetInput(x)
	    self.Run(T, dt=dt)
	    return self.layers[-1].v #self.layers[-1].sigma(self.layers[-1].v)

	def Generate(self, T, y, dt=0.01):
	    self.learn = False
	    self.layers[1].SetFB()
	    self.layers[-1].SetBidirectional()
	    self.SetExpectation(y)
	    self.Run(T, dt=dt)
	    mu0 = self.layers[1].sigma(self.layers[1].v) @ self.connections[0].M + self.layers[0].b
	    return mu0


#============================================================
#
# Untility functions
#
#============================================================
def MakeBatches(data_in, data_out, batch_size=10):
    '''
    batches = MakeBatches(data_in, data_out, batch_size=10)
    
    Breaks up the dataset into batches of size batch_size.
    
    Inputs:
      data_in    is a list of inputs
      data_out   is a list of outputs
      batch_size is the number of samples in each batch
      
    Output:
      batches is a list containing batches, where each batch is:
                 [in_batch, out_batch]
    '''
    N = len(data_in)
    batches = []
    for k in range(0, N, batch_size):
        din = data_in[k:k+batch_size]
        dout = data_out[k:k+batch_size]
        if isinstance(din, (list, tuple)):
            batches.append( [torch.stack(din, dim=0).float().to(device) , torch.stack(dout, dim=0).float().to(device)] )
        else:
            batches.append( [din.float().to(device) , dout.float().to(device)] )
    return batches

def OneHot(z):
    y = np.zeros(np.shape(z))
    y[np.argmax(z)] = 1.
    return y







#