# NeuralNetwork

import numpy as np
import Layer
import torch
from torch.autograd import Variable
from copy import deepcopy
from IPython.display import display
from ipywidgets import FloatProgress


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

	def __init__(self, b=None, a=None, act=None, W=None, M=None, symmetric=False):
		'''
		Connection(b, a, act=None, W=None, M=None, symmetric=False)

		Create a connection between two layers.

		Inputs:
		  b is the layer (object) below
		  a is the later (object) abover
		  act is the activation function, 'tanh', 'logistic', 'identity', 'softmax'
		  W and M can be supplied (optional)
		  symmetric is True if you want W = M^T
		'''
		ltypes = (Layer.PELayer, Layer.InputPELayer, Layer.TopPELayer)
		if isinstance(b, ltypes) or isinstance(a, ltypes):
			self.below = b
			self.above = a

			self.below_idx = self.below.idx
			self.above_idx = self.above.idx

			Wgiven = hasattr(W, '__len__')
			Mgiven = hasattr(M, '__len__')
			af_given = hasattr(act, '__len__')


			if Mgiven:
			    self.M = torch.tensor(M).float().to(device)
			else:
			    #self.M = torch.randn( a.n, b.n, dtype=torch.float32, device=device) / np.sqrt(b.n)
			    self.M = torch.randn( a.n, b.n, dtype=torch.float32, device=device) / np.sqrt(b.n) / 10.
			    
			if Wgiven:
			    self.W = torch.tensor(W).float().to(device)
			elif symmetric:
				self.W = deepcopy(self.M.transpose(1,0))
			else:
			    self.W = torch.randn( b.n, a.n, dtype=torch.float32, device=device) / np.sqrt(b.n)/10.
			    # This distribution is taken from the caption of Fig. 6 in Wittington and Bogacz
			    #self.W = ( torch.rand( b.n, a.n, dtype=torch.float32, device=device) - 0.5 ) *8*np.sqrt(6)/np.sqrt(a.n+b.n)

			self.dWdt = torch.zeros( b.n, a.n, dtype=torch.float32, device=device)
			self.dMdt = torch.zeros( a.n, b.n, dtype=torch.float32, device=device)

			self.W_decay = 0.0
			self.M_decay = 0.0
			self.lam = 0.

			if af_given:
				self.SetActivationFunction(act)
				print(act)
			else:
				self.activation_function = 'tanh'
				self.sigma = tanh  # activation function
				self.sigma_p = tanh_p
		else:
			self.below = []
			self.above = []
			self.below_idx = -99
			self.above_idx = -99
			self.W = []
			self.M = []
			self.dWdt = []
			self.dMdt = []
			self.activation_function = 'tanh'
			self.sigma = tanh  # activation function
			self.sigma_p = tanh_p
		self.learn = True

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
	    elif fcn=='ReLU':
	    	self.activation_function = 'ReLU'
	    	self.sigma = ReLU
	    	self.sigma_p = ReLU_p
	    elif fcn=='softmax':
	        self.activation_function = 'softmax'
	        self.sigma = softmax
	        self.sigma_p = softmax_p
	    else:
	        print('Activation function not recognized, using logistic')
	        self.activation_function = 'logistic'
	        self.sigma = logistic
	        self.sigma_p = logistic_p

	def SetDecay(self, W_decay=0.0, M_decay=0.0):
		self.W_decay = W_decay
		self.M_decay = M_decay
            
	def MakeIdentity(self, noise=0.):
		if self.below.n!=self.above.n:
			print('Connection matrix is not square.')
		else:
			n = self.below.n
			self.W = torch.eye(n).float().to(device) + noise*torch.normal(mean=torch.zeros((n,n)), std=1.).float().to(device)
			self.M = torch.eye(n).float().to(device) + noise*torch.normal(mean=torch.zeros((n,n)), std=1.).float().to(device)

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
		np.save(fp, self.W.cpu())
		np.save(fp, self.M.cpu())
		np.save(fp, self.W_decay)
		np.save(fp, self.M_decay)
		np.save(fp, self.learn)
		np.save(fp, self.activation_function)




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
	    self.l_rate = 0.2
	    #self.lam = 0.0
	    self.learn = False
	    self.learn_weights = True
	    self.learn_biases = True
	    self.batch_size = 0
	    self.probe_on = False
	    self.rms_history = []
	    self.pe_error_history = []
	    self.test_error_history = []        
	    self.train_error_history = []


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
				L.expectation = []
			self.AddLayer(L)

		n_connections = np.asscalar( np.load(fp) )
		for k in range(n_connections):
			bi = np.asscalar( np.load(fp) ) # below_idx
			ai = np.asscalar( np.load(fp) ) # above_idx
			W = np.load(fp)
			M = np.load(fp)
			W_decay = np.asscalar( np.load(fp) )
			M_decay = np.asscalar( np.load(fp) )
			learn = np.asscalar( np.load(fp) )
			activation_function = str( np.load(fp) )
			self.Connect(bi, ai, W=W, M=M)
			self.connections[-1].learn = learn
			self.connections[-1].SetActivationFunction(activation_function)
			self.connections[-1].SetDecay(W_decay=W_decay, M_decay=M_decay)
		fp.close()


	def Connect(self, prei, posti, act=None, W=None, M=None, symmetric=False):
	    '''
	    Connect(prei, posti, act=af, W=None, M=None, symmetric=False)

	    Connect two layers.

	    Inputs:
	      prei is the index of the lower layer
	      posti is the index of the upper layer
	      af is one of 'identity', 'logistic', 'tanh', or 'softmax'
	      W and M are connection weight matrices
	      symmetric is True if you want W = M^T
	    '''
	    preL = self.layers[prei]
	    postL = self.layers[posti]
	    preL.layer_above = postL
	    postL.layer_below = preL
	    self.connections.append(Connection(preL, postL, act=act, W=W, M=M, symmetric=symmetric))


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

	def SetvDecay(self, v_decay):
		for l in self.layers:
			l.v_decay = torch.tensor(v_decay).float().to(device)

	def SetWeightDecay(self, w_decay):
		for c in self.connections:
			c.lam = torch.tensor(w_decay).float().to(device)

	def Allocate(self, x):
		'''
		Allocate(x)

		Creates zero-vectors for the state and error nodes of all layers.

		Input:
		  x can either be the number of samples in a batch, or it can be
		    a batch.
		'''
		proposed_batch_size = 1
		if type(x) in (int, float, ):
		    proposed_batch_size = x
		    dims = 1
		else:
		    dims = len(np.shape(x))
		if dims==2:
		    proposed_batch_size = np.shape(x)[0]
		if proposed_batch_size!=self.batch_size:
			self.batch_size = proposed_batch_size
			del self.t_history
			self.t_history = []
			self.t = 0.
			#print('Allocating')
			for idx,l in enumerate(self.layers):
				l.Allocate(batch_size=proposed_batch_size)
			#print('Re-Allocating to '+str(proposed_batch_size))


	def SetInput(self, x):
	    self.Allocate(x)
	    self.layers[0].SetInput(x)


	def SetExpectation(self, x):
	    self.Allocate(x)
	    self.layers[-1].SetExpectation(x)


	def SetExpectationState(self, v):
		self.Allocate(v)
		self.layers[-1].v = torch.tensor(v).float().to(device)



	def ShowState(self):
	    for idx, layer in enumerate(self.layers):
	        if layer.is_input:
	            print('Layer '+str(idx)+' (input):')
	            layer.ShowState()
	            layer.ShowError()
	        elif layer.is_top:
	            print('Layer '+str(idx)+' (expectation):')
	            layer.ShowState()
	            layer.ShowError()
	        else:
	            print('Layer '+str(idx)+':')
	            layer.ShowState()
	            layer.ShowError()

	def ShowWeights(self):
	    for idx in range(len(self.layers)-1):
	        print('  W'+str(idx)+str(idx+1)+' = ')
	        print(str(np.array(self.connections[idx].W)))
	        print('  M'+str(idx+1)+str(idx)+' = ')
	        print(str(np.array(self.connections[idx].M)))

	def ShowBias(self):
	    for idx, layer in enumerate(self.layers):
	        layer.ShowBias()

	def Reset(self):
		del self.t_history
		self.t_history = []
		self.t = 0.
		for l in self.layers:
		    l.Reset()

	def ResetErrors(self):
		for l in self.layers:
			l.e.zero_()

	def ResetGradients(self):
	    for l in self.layers:
	        l.dvdt.zero_()
	        l.dedt.zero_()
	        l.dbdt.zero_()
	    for c in self.connections:
	    	if c.learn:
		        c.dWdt.zero_()
		        c.dMdt.zero_()

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


	def rms_error(self, x, y):
		self.BackprojectExpectation(y)
		mu0 = self.GenerateSamples()
		rms = torch.sqrt(torch.mean(torch.sum(torch.pow(x-mu0, 2), 1)/np.shape(x)[1]))
		#print(mu0)
		return rms


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


	def Integrate(self):
		# Loop through connections
		# Then loop through layers

		# First, address only the connections between layers
		for c in self.connections:
		    blw = c.below
		    abv = c.above
		    # e <-- v
		    blw.dedt -= c.sigma(abv.v)@c.M + blw.b
		    # e --> v
		    #abv.dvdt += abv.alpha*(blw.e@c.W)*c.sigma_p(abv.v)
		    abv.dvdt += abv.alpha*(blw.e@c.W)  # *** exclude the derivative of the act fcn ***

		    if c.learn==True:
		        if self.batch_size==1:
		            c.dMdt += c.sigma(abv.v.reshape([abv.n,1])) @ blw.e.reshape([1,blw.n])
		            c.dWdt += blw.e.reshape([blw.n,1]) @ c.sigma(abv.v.reshape([1,abv.n]))
		        else:
		            c.dMdt += c.sigma(abv.v).transpose(1,0) @ blw.e
		            c.dWdt += blw.e.transpose(1,0) @ c.sigma(abv.v)

		        blw.dbdt += torch.sum(blw.e, dim=0)


		# Next, address the connections inside each layer
		for l in self.layers:
		    if l.is_input:
		    	# The input state only updated when beta>0 (eg. FB mode)
		    	# Not updated in FF mode, when beta=0
		        l.dvdt -= l.beta*l.e + l.v_decay*l.beta*l.v
		        l.dedt += l.v - l.Sigma*l.e
		    elif l.is_top:
		        l.dvdt -= l.beta*l.e + l.v_decay*l.alpha*l.v
		        l.dedt += l.v - l.expectation - l.Sigma*l.e
		    else:
		        l.dvdt -= l.beta*l.e + l.v_decay*l.v
		        l.dedt += l.v - l.Sigma*l.e


	def Step(self, dt=0.01):
	    k = dt/self.learning_tau
	    for l in self.layers:
	        l.Step(dt=dt)

	    # Only update weights if learning is one, and we are past
	    # the "blackout" transient period.
	    if self.learn and self.t-self.t_runstart>=self.learning_blackout:
	        for c in self.connections: # The first connection is fixed
	        	if self.learn_weights:
		            c.M += k*c.dMdt / self.batch_size - k*c.lam*c.M
		            c.W += k*c.dWdt / self.batch_size - k/2.*c.lam*c.W
		        if self.learn_biases:
		            c.below.b += k*c.below.dbdt / self.batch_size



	def PEError(self):
		'''
		pe_error = NN.PEError()

		Returns the sum of the squares of all the error nodes.
		'''
		total_pe_error = 0.
		#l_counter = 0
		for l in self.layers:
			#print('Layer '+str(l_counter))
			#l_counter += 1
			total_pe_error += l.PEError()
		return total_pe_error

	def GenerateSamples(self):
		mu0 = self.connections[0].sigma(self.layers[1].v) @ self.connections[0].M + self.layers[0].b
		return mu0

	def Learn(self, x, t, T=2., epochs=5, dt=0.01, batch_size=10, shuffle=True):
		#net.learning_tau = 30. #torch.tensor(batch_size).float().to(device) * 10.
		self.SetBidirectional()
		self.layers[0].SetFF()
		fp = FloatProgress(min=0,max=epochs*len(x))
		#self.rms_history.append(self.rms_error(x,t))
		display(fp)
		for k in range(epochs):
			batches = MakeBatches(x, t, batch_size=batch_size, shuffle=shuffle)

			for samp in batches:
				#net.Reset()
				if batch_size==1 and len(np.shape(samp[0]))==2:
					#self.BackprojectExpectation(samp[1][0])
					#self.PropagateErrors(samp[0][0])
					self.Infer(T, samp[0][0], samp[1][0], dt=dt, learn=True)
				else:
					#self.BackprojectExpectation(samp[1])
					#self.PropagateErrors(samp[0])
					self.Infer(T, samp[0], samp[1], dt=dt, learn=True)
				fp.value += batch_size
			#self.rms_history.append(self.rms_error(x, t))


	def BackprojectExpectation(self, y):
		'''
		Initialize all the state nodes from the top-layer expection.

		This does not overwrite the state of layer[0].
		'''
		self.Allocate(y)
		# State nodes
		self.layers[-1].v = y.clone().detach()
		for idx in range(self.n_layers-2, 0,-1):
			v = self.connections[idx].sigma(self.layers[idx+1].v)@self.connections[idx].M + self.layers[idx].b
			#v = self.layers[idx+1].sigma(self.layers[idx+1].v)@self.connections[idx].M + self.layers[idx].b
			self.layers[idx].v = v.clone().detach()
		mu0 = self.connections[0].sigma(self.layers[1].v) @ self.connections[0].M + self.layers[0].b
		#mu0 = self.layers[1].sigma(self.layers[1].v) @ self.connections[0].M + self.layers[0].b
		return mu0


	def OverwriteErrors(self):
		'''
		OverwriteErrors()

		Uses current states to overwrite the error nodes; it sets them to their equilibria, assuming
		the states are held constant.

		This method does NOT update the error nodes in the top layer.
		'''
		for idx in range(0, self.n_layers-1):
			self.layers[idx].e = (self.layers[idx].v - (self.connections[idx].sigma(self.layers[idx+1].v) @ self.connections[idx].M) - self.layers[idx].b) / self.layers[idx].variance


	def OverwriteStates(self):
		'''
		NN.OverwriteStates()

		Updates the states, incrementing them by the incoming connections.
		Note that the layer-wise alpha and beta are used to weigh the forward and
		backward inputs, respectively.

		This method potentially updates the state nodes in ALL layers, including the bottom and top.
		'''
		self.layers[0].v -= 0.2*self.layers[0].beta * ( self.layers[0].e + self.layers[0].v_decay*self.layers[0].v )
		for idx in range(1, self.n_layers):
			# abv.alpha*(blw.e@c.W)*c.sigma_p(abv.v)

			# Original version in the W&B paper
			#self.layers[idx].v += 0.2*( self.layers[idx].alpha*(self.layers[idx-1].e @ self.connections[idx-1].W) * self.connections[idx-1].sigma_p(self.layers[idx].v) - self.layers[idx].beta*self.layers[idx].e )

			# Removing sigma'
			self.layers[idx].v += 0.2*( self.layers[idx].alpha*(self.layers[idx-1].e @ self.connections[idx-1].W) - self.layers[idx].beta*self.layers[idx].e )
			if self.layers[idx].is_top==False:
				self.layers[idx].v -= 0.2*self.layers[idx].v_decay*self.layers[idx].v



	def FastLearn(self, x, t, test=False, T=20, epochs=5, Beta_one=0.9, Beta_two=0.999, ep=0.00000001, batch_size=10, noise=False, freeze=10, shuffle=True):
		'''
		Implementation of Whittington & Bogacz 2017

		'''
		if test:
			test_length = len(test[0])
			self.test_error_history.append(self.dataset_error(test[0], test[1], test_length))
		train_length = len(x)
		#self.train_error_history.append(self.dataset_error(t, x, train_length))
        
		fp = FloatProgress(min=0,max=epochs)
		display(fp)
		#self.batch_size = batch_size  # this needs to be set in the Allocate function
        
		#Initialize Adam parameters
		alpha = self.l_rate

		freeze_W = False

		m = [] #1st momentum vector containing each layer's dMdt, dWdt, and dbdt in that order
		v = [] #2nd momentum vector with elements identical to above
		g = [] #vector of all gradients with elements identical to above

		OG_W_decay = [] #Saves the original decay values since they are set to 0 after 11 training epochs
		OG_M_decay = []

		self.Allocate(batch_size)
		self.ResetGradients()

		for c in self.connections:
			m.append(c.dMdt)
			m.append(c.dWdt)
			m.append(c.below.dbdt)

			v.append(c.dMdt)
			v.append(c.dWdt)
			v.append(c.below.dbdt)

			g.append(c.dMdt)
			g.append(c.dWdt)
			g.append(c.below.dbdt)
            
			OG_W_decay.append(c.W_decay)
			OG_M_decay.append(c.M_decay)

		time=0 #time step

		for k in range(0, epochs):
			#Remove multiplicative noise after 13 epochs
			if k > 12:
				noise = False
            
			#Remove decay after 11 epochs
			# if k > 10 and self.connections[0].W_decay > 0.0:
			# 	for c in self.connections:
			# 		c.W_decay = 0.0
			# 		c.M_decay = 0.0
                
			#Do not update W after 'freeze' # of epochs 
			if k > freeze: 
				freeze_W = True
            
			epoch_pe_error = 0.
			batches = MakeBatches(x, t, batch_size=batch_size, shuffle=shuffle)
			self.Allocate(batch_size)

			self.ResetGradients()

			for j in range(0, len(batches)):
				mb = batches[j]

				#Perform Adam while loop
				time += 1

				#Evaluate gradients with respect to objective function at time step t
				self.BackprojectExpectation(mb[1])

				# 2. Set the desired output
				self.SetInput(mb[0])

				# 3. Run infer_ps
				# This involves fixing the input in the bottom and top layers.
				self.layers[0].SetFF()  # Don't update state of bottom layer
				self.layers[-1].SetFB() # Don't update state of top layer
				self.OverwriteErrors()
				for k in range(0, T):
					self.OverwriteStates()
					self.OverwriteErrors()

				epoch_pe_error += self.PEError()

				# 4. Calculate gradients from the error nodes
				idx = 0 #keeps track of idx in g

				for c in self.connections:
					blw = c.below
					abv = c.above
                    
					if blw.variance > 1:
						blw.e *= blw.variance
                    
					if self.batch_size == 1:
						g[idx] = c.sigma(abv.v.reshape([abv.n,1])) @ blw.e.reshape([1,blw.n]) - c.M_decay*c.M
						if noise == True:
							noise_M = Variable(c.M.data.new(c.M.size()).normal_(1, 0.25))
							g[idx] *= noise_M
						idx += 1

						g[idx] = blw.e.reshape([blw.n,1]) @ c.sigma(abv.v.reshape([1,abv.n])) - c.W_decay*c.W
						if noise == True:
							noise_W = Variable(c.W.data.new(c.W.size()).normal_(1, 0.25))
							g[idx] *= noise_W                                  
						idx += 1
					else:
						g[idx] = c.sigma(abv.v).transpose(1,0) @ blw.e - c.M_decay*c.M
						if noise == True:
							noise_M = Variable(c.M.data.new(c.M.size()).normal_(1, 0.25))
							g[idx] *= noise_M
						idx += 1

						g[idx] = blw.e.transpose(1,0) @ c.sigma(abv.v) - c.W_decay*c.W
						if noise == True:
							noise_W = Variable(c.W.data.new(c.W.size()).normal_(1, 0.25))
							g[idx] *= noise_W      
						idx += 1

					# Gradient w.r.t. biases
					g[idx] = torch.sum(blw.e, dim=0)
					idx += 1

				for i in range (0, len(m), 3):
					c = self.connections[i // 3]
                    
					m[i] = Beta_one*m[i] + (1 - Beta_one)*g[i]
					v[i] = Beta_two*v[i] + (1 - Beta_two)*(g[i]*g[i])
					m_hat = m[i] / (1 - (Beta_one**time))
					v_hat = v[i] / (1 - (Beta_two**time))
					c.M += alpha * (m_hat*(torch.reciprocal(torch.sqrt(v_hat) + ep)) - c.lam*c.M)
                    
                    
					m[i+1] = Beta_one*m[i+1] + (1 - Beta_one)*g[i+1]
					v[i+1] = Beta_two*v[i+1] + (1 - Beta_two)*(g[i+1]*g[i+1])
					m_hat = m[i+1] / (1 - (Beta_one**time))
					v_hat = v[i+1] / (1 - (Beta_two**time))
					if not freeze_W:
						c.W += alpha * (m_hat*(torch.reciprocal(torch.sqrt(v_hat) + ep)) - c.lam*c.W)
                    
					m[i+2] = Beta_one*m[i+2] + (1 - Beta_one)*g[i+2]
					v[i+2] = Beta_two*v[i+2] + (1 - Beta_two)*(g[i+2]*g[i+2]) 
					m_hat = m[i+2] / (1 - (Beta_one**time))
					v_hat = v[i+2] / (1 - (Beta_two**time))
					if self.learn_biases:
						c.below.b += alpha*m_hat*(torch.reciprocal(torch.sqrt(v_hat) + ep)) #- c.lam*c.below.b

			if test:
				self.test_error_history.append(self.dataset_error(test[0], test[1], test_length))
			#self.train_error_history.append(self.dataset_error(t, x, train_length))
        
			fp.value += 1

		#Restore original decay values
		for i in range (0, len(self.connections)):
			self.connections[i].W_decay = OG_W_decay[i]     
			self.connections[i].M_decay = OG_M_decay[i]
            
	def dataset_error(self, dataset_in, dataset_out, dataset_length):
		self.BackprojectExpectation(dataset_in)
		z = self.connections[0].sigma(self.layers[1].v)@self.connections[0].M + self.layers[0].b
		y_classes = np.argmax(z,1)
		t_classes = np.argmax(dataset_out, 1)
		correct = np.count_nonzero((y_classes - t_classes)==0)
		return 1.0 - (correct / dataset_length)
            
	def FastPredict(self, x, T=100):
		'''
		y = NN.FastPredict(x, T=100)

		Run network to equilibrium with input clamped to x. This method uses the
		fast convergence method of Whittington & Bogacz [2017].
		'''
		# Set the input layer
		self.Allocate(x)
		self.layers[0].SetInput(x)

		self.layers[0].SetFF()
		self.layers[-1].SetFF()

        # 3. Run infer_ps
		self.OverwriteErrors()
		for k in range(0, T):
			self.OverwriteStates()
			self.OverwriteErrors()
            
		g = self.connections[-1].sigma(self.layers[-2].v)@self.connections[-1].W 
		return g #self.layers[-1].sigma(self.layers[-1].v)
        
        
	def SetBidirectional(self):
		for l in self.layers:
			l.SetBidirectional()
			
	def SetFF(self):
		for l in self.layers:
			l.SetFF()
	

	def Infer(self, T, x, y, dt=0.01, learn=False):
	    self.learn = learn
	    #self.layers[1].SetBidirectional()
	    #self.layers[-1].SetBidirectional()
	    #self.layers[-1].SetFB()
	    self.Allocate(x)
	    self.SetInput(x)
	    self.SetExpectation(y)
	    self.Run(T, dt=dt)

	def Predict(self, T, x, dt=0.01):
	    self.learn = False
	    #self.layers[1].SetFF()
	    #self.layers[1].SetBidirectional()
	    #self.layers[-1].alpha = torch.tensor(mask).float().to(device)
	    #self.layers[-1].beta = 1.-self.layers[-1].alpha
	    #self.layers[-1].SetFF()
	    self.Allocate(x)
	    self.SetInput(x)
	    self.Run(T, dt=dt)
	    return self.layers[-1].v #self.layers[-1].sigma(self.layers[-1].v)

	def Generate(self, T, y, dt=0.01):
	    self.learn = False
	    self.layers[0].SetFB()
	    self.layers[-1].SetBidirectional()
	    self.Allocate(y)
	    self.SetExpectation(y)
	    self.Run(T, dt=dt)
	    #mu0 = self.connections[0].sigma(self.layers[1].v) @ self.connections[0].M + self.layers[0].b
	    return self.layers[0].v


#============================================================
#
# Untility functions
#
#============================================================
def MakeBatches(data_in, data_out, batch_size=10, shuffle=True):
    '''
	    batches = MakeBatches(data_in, data_out, batch_size=10)
	    
	    Breaks up the dataset into batches of size batch_size.
	    
	    Inputs:
	      data_in    is a list of inputs
	      data_out   is a list of outputs
	      batch_size is the number of samples in each batch
	      shuffle    shuffle samples first (True)
	      
	    Output:
	      batches is a list containing batches, where each batch is:
	                 [in_batch, out_batch]


	    Note: The last batch might be incomplete (smaller than batch_size).
    '''
    N = len(data_in)
    r = range(N)
    if shuffle:
        r = np.random.permutation(N)
    batches = []
    for k in range(0, N, batch_size):
        if k+batch_size<=N:
            din = data_in[r[k:k+batch_size]]
            dout = data_out[r[k:k+batch_size]]
        else:
            din = data_in[r[k:]]
            dout = data_out[r[k:]]
        if isinstance(din, (list, tuple)):
            batches.append( [np.stack(din, dim=0) , np.stack(dout, dim=0)] )
        else:
            batches.append( [din , dout] )
    return batches

def logistic(v):
    return torch.reciprocal( torch.add( torch.exp(-v), 1.) )

def logistic_p(v):
    lv = logistic(v)
    return lv*(1.-lv)
    #return torch.addcmul( torch.zeros_like(v) , lv , torch.neg(torch.add(lv, -1)) ) 

def ReLU(v):
	return torch.clamp(v, min=0.)

def ReLU_p(v):
	return torch.clamp(torch.sign(v), min=0.)

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
    sftmax = torch.nn.Softmax()
    return sftmax(v)
    #if len(np.shape(z))==1:
    #    s = torch.sum(z)
    #    return z/s
    #else:
    #    s = torch.sum(z, dim=1)
    #    return z/s[np.newaxis,:].transpose(1,0).repeat([1,np.shape(v)[1]])

def softmax_p(v):
    z = softmax(v)
    return z*(1.-z)

def OneHot(z):
    y = np.zeros(np.shape(z))
    y[np.argmax(z)] = 1.
    return y







#
