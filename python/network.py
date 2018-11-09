# Network.py class

import numpy as np
from IPython.display import display
from ipywidgets import FloatProgress  
import time

def Logistic(z):
	return 1. / (1 + np.exp(-z) )

def Logistic_Primed_y(y):
	return y * (1. - y)


def ReLU(z):
	return np.clip(z, a_min=0., a_max=None)
	#return np.max(0, z)

def ReLU_Primed_y(y):
	return (np.sign(y) + 1.)/2.


def Identity(z):
	return z

def Identity_Primed_y(y):
	return np.ones(shape=np.shape(y))


def OneHot(z):
	y = np.zeros(np.shape(z))
	y[np.argmax(z)] = 1.
	return y


class Network():

	def __init__(self, sizes, type='classifier'):
		'''
		    net = Network.Network(sizes, type='classifier')
		    
		    Creates a Network and saves it in the variable 'net'.
		    
		    Inputs
		      sizes is a list of integers specifying the number
		          of nodes in each layer
		          eg. [5, 20, 3] will create a 3-layer network
		              with 5 input, 20 hidden, and 3 output nodes
		      type can be either 'classifier' or 'regression', and
		          sets the activation function on the output layer,
		          as well as the loss function.
		          'classifier': logistic, cross entropy
		          'regression': linear, mean squared error
		'''
		self.n_layers = len(sizes)
		self.N = sizes   # array of number of nodes per layer
		self.h = []      # node activities, one array per layer
		self.z = []      # input current, one array per layer
		self.b = []      # biases, one array per layer
		self.W = []      # Weight matrices, indexed by the layer below it
		self.dEdb = []   # Gradient of loss w.r.t. biases
		self.dEdW = []   # Gradient of loss w.r.t. weights
		self.dEdbdt = []   # Velocity of Gradient of loss w.r.t. biases
		self.dEdWdt = []   # Velocity of Gradient of loss w.r.t. weights
		for n in self.N:
		    self.h.append(np.zeros(n))
		    self.z.append(np.zeros(n))
		    self.dEdb.append(np.zeros(n))
		    self.dEdbdt.append(np.zeros(n))
		    self.b.append(np.random.normal(size=n)/np.sqrt(n))
		for k in range(len(self.N)-1):
		    self.W.append(np.random.normal(size=[self.N[k+1], self.N[k]])/np.sqrt(self.N[k]))
		    self.dEdW.append(np.zeros([self.N[k+1], self.N[k]]))
		    self.dEdWdt.append(np.zeros([self.N[k+1], self.N[k]]))

		# Two common types of networks
		# The member variable self.Loss refers to one of the implemented
		# loss functions: MSE, or CrossEntropy.
		# Call it using self.Loss(t)
		self.Activation = Logistic
		self.Activation_Primed = Logistic_Primed_y
		if type=='classifier':
			self.classifier = True
			self.Loss = self.CrossEntropy
			#self.Loss_Primed = self.CrossEntropy_Primed
			self.OutputActivation = Logistic
			self.TopGradient = self.TopGradient_Logistic_CE
			#self.OutputActivation_Primed = Logistic_Primed_y
		else:
			self.classifier = False
			self.Loss = self.MSE
			#self.Loss = self.MSE_Primed
			self.OutputActivation = Identity
			self.TopGradient = self.TopGradient_Identity_MSE
			#self.OutputActivation_Primed = Identity_Primed_y

	def FeedForward(self, x):
		'''
			y = net.FeedForward(x)

			Runs the network forward, starting with x as input.
			Returns the activity of the output layer.

			All node use 
			Note: The activation function used for the output layer
			depends on what self.Loss is set to.
		'''
		self.h[0] = x # set input layer

		# For each layer...
		for l in range(1, self.n_layers):
			# Calc. input current to next layer
			z = np.dot(self.W[l-1], self.h[l-1]) + self.b[l]

			# Use activation function for hidden nodes.
			# For output layer, use chosen activation.
			if l<self.n_layers-1:
				self.h[l] = self.Activation(z)
			else:
				# For a regression network, use linear activation
				# for output layer.
				self.h[l] = self.OutputActivation(z)
		# Return activity of output layer
		return self.h[-1]

	def FF_BP(self, X, T):
		'''
			net.FF_BP(X, T)

			Runs both the feedforward and backprop phases for a batch of
			input/output samples, each sample stored in the columns of
			X and T.

			Inputs
			  X is a AxP matrix, where the input dim is A, and P samples
			  T is a BxP matrix, where the output dim is B, and P samples

			Output
			  None
		'''
		H = []
		HH = X
		P = np.shape(X)[1]
		for W, b in zip(self.W[:-1], self.b[1:-1]):
			HH = np.dot(W,HH) + np.outer(b, np.ones(shape=[1,P]))
			print(np.shape(HH))
			HH = Logistic(HH)
		HH = np.dot(self.W[-1],HH) + np.outer(self.b[-1], np.ones(shape=[1,P]))
		HH = self.OutputActivation(HH)
		return HH


	def FeedForward_dropout(self, x):
		'''
			y = net.FeedForward_dropout(x)

			Runs the network forward, starting with x as input.
			However, a random half of the hidden nodes are set to 0 activity.
			Returns the activity of the output layer.

			All node use 
			Note: The activation function used for the output layer
			depends on what self.Loss is set to.
		'''
		# Take care of first layer
		self.h[0] = x # set input layer
		z = np.dot(self.W[0], self.h[0]) + self.b[1]

		# For each layer...
		for l in range(1, self.n_layers-1):
			self.h[l] = self.Activation(z)
			# Set some nodes to 0
			not_dropped = int(np.ceil(self.N[l]/2))
			idx = list(range(self.N[l]))
			np.random.shuffle(idx)
			for k in idx[:int(not_dropped)]:
				self.h[l][k] = 0.

			# Calc. input current to next layer
			z = 2.*np.dot(self.W[l], self.h[l]) + self.b[l+1]

		self.h[-1] = self.OutputActivation(z)
		# Return activity of output layer
		return self.h[-1]


	def Evaluate(self, data):
		'''
		E = net.Evaluate(data)

		Computes the average loss over the supplied dataset.

		Inputs
		  data is a list of 2 arrays containing inputs and targets

		Outputs
		  E is a scalar, the average loss
		'''
		total_loss = 0.
		for x, t in zip(data[0], data[1]):
			self.FeedForward(x)
			total_loss += self.Loss(t)
		return total_loss / len(data[0])

	
	def ClassificationAccuracy(self, data):
		n_correct = 0
		for x, t in zip(data[0], data[1]):
			y = self.FeedForward(x)
			yb = OneHot(y)
			if np.argmax(yb)==np.argmax(t):
				n_correct += 1
		return float(n_correct) / len(data[0])

	
	def CrossEntropy(self, t):
	    y = self.h[-1]
	    E = -sum(t*np.log(y) + (1.-t)*np.log(1.-y))
	    return E

	# def CrossEntropy_Primed(self, t):
	# 	return (self.h[-1] - t) / Logistic_Primed_y(self.h[-1])
	
	def MSE(self, t):
		'''
			E = net.MSE(t)

			Evaluates the MSE loss function using t and the activity of the top layer.
			To evaluate the network's performance on an input/output pair (x,t), use
			  net.FeedForward(x)
			  E = net.Loss(t)

			Inputs
			  t is an array holding the target output

			Outputs
			  E is the loss function for the given case
		'''
		y = self.h[-1]
		E = 0.5 * sum((y-t)**2)
		return E
	    
	# def MSE_Primed(self, t):
	# 	return (self.h[-1] - t)

	def TopGradient_Logistic_CE(self, t):
		return self.h[-1] - t

	def TopGradient_Identity_MSE(self, t):
		return self.h[-1] - t

	def BackProp(self, t, lrate=0.05, decay=0.):
		'''
		    Given a target t, updates the connection weights and biases using the
		    backpropagation algorithm.
		'''
		dEdz = self.TopGradient(t)

		for l in range(self.n_layers-2, -1, -1):
			sigma_primed = self.Activation_Primed(self.h[l])
			dEdW = np.outer(self.h[l], dEdz).T
			dEdb = dEdz
			dEdz = sigma_primed * np.dot(self.W[l].T, dEdz)
			self.W[l] -= lrate*dEdW + decay*self.W[l]
			self.b[l+1] -= lrate*dEdb + decay*self.b[l+1]
		return

	def BackProp_no_increment(self, t, lrate=0.05, decay=0.):
		'''
		    Given a target t, updates the connection weights and biases using the
		    backpropagation algorithm.
		'''
		dEdz = self.TopGradient(t)
		for l in range(self.n_layers-2, -1, -1):
			#sigma_primed = self.h[l] * (1.-self.h[l])
			sigma_primed = self.Activation_Primed(self.h[l])
			#self.dEdW[l] = momentum*self.dEdW[l] + (1.-momentum)*np.outer(self.h[l], dEdz).T
			#self.dEdb[l+1] = momentum*self.dEdb[l+1] + (1.-momentum)*dEdz
			self.dEdW[l] += np.outer(self.h[l], dEdz).T
			self.dEdb[l+1] += dEdz
			dEdz = sigma_primed * np.dot(self.W[l].T, dEdz)
		return


	def IncrementParameters(self, lrate=0.05):
	    '''
	    Use the gradients stored in dEdW and dEdb to update the weights, W and b.
	    
	    Also resets the gradients.
	    '''
	    for l in range(self.n_layers-1):
	    	# Momentum
	        # Use the gradients to update our weights
	        self.W[l] -= lrate*self.dEdW[l]
	        self.b[l+1] -= lrate*self.dEdb[l+1]
	        # Reset the gradient accumulators
	        self.dEdW[l] = np.zeros(np.shape(self.W[l]))
	        self.dEdb[l+1] = np.zeros(np.shape(self.b[l+1]))
	    return
	        
	
	def Learn(self, data, lrate=0.05, epochs=1, progress=True, decay=0., dropout=False):
		'''
			Network.Learn(data, lrate=0.05, epochs=1, progress=True)

			Run through the dataset 'epochs' number of times, incrementing the
			network weights for each training sample. For each epoch, it
			shuffles the order of the samples.

			Inputs
			  data is a list of 2 arrays, one for inputs, and one for targets
			  lrate is the learning rate (try 0.001 to 0.5)
			  epochs is the number of times to go through the training data
			  progress (Boolean) indicates whether to show a progress bar
		'''
		data_shuffled = list(zip(data[0],data[1]))
		if progress:
		    fp = FloatProgress(min=0,max=epochs)  
		    display(fp)
		loss_history = []
		for k in range(epochs):
			np.random.shuffle(data_shuffled)
			for x, t in data_shuffled:
				if dropout:
					self.FeedForward_dropout(x)
				else:
					self.FeedForward(x)
				self.BackProp(t, decay=decay)
				#self.IncrementParameters(lrate=lrate)
			#if progress and np.mod(k,100)==0:
			if progress:
				fp.value += 1

			loss_history.append([k, self.Evaluate(data)])

		if progress:
			return np.array(loss_history)
		else:
			return


	def SGD(self, data, lrate=0.05, epochs=1, batch_size=10, dropout=False):
		data_shuffled = list(zip(data[0], data[1]))
		fp = FloatProgress(min=0,max=epochs)  
		display(fp)
		loss_history = []
		for k in range(epochs):
			# Split data into batches
			np.random.shuffle(data_shuffled)
			n_batches = int(np.floor( len(data[0]) / batch_size ))
			idx = [n*batch_size for n in range(n_batches+1)]
			for batch_number in range(n_batches):
				if batch_number<=n_batches-1:
					mini_batch = data_shuffled[idx[batch_number]:idx[batch_number+1]]
				else:
					mini_batch = data_shuffled[idx[batch_number]:]
				for x, t in mini_batch:
					if not(dropout):
						self.FeedForward(x)
					else:
						self.FeedForward(x, dropout=True)
					self.BackProp_no_increment(t)
				self.IncrementParameters(lrate=lrate)
			#if progress and np.mod(k,100)==0:
			fp.value += 1

			loss_history.append([k, self.Evaluate(data)])

		return np.array(loss_history)

	def __SGD__(self, data, lrate=0.05, epochs=1, decay=0., progress=True, batch_size=10, dropout=False):
		data_shuffled = list(zip(data[0], data[1]))
		print('SGD')
		if progress:
		    fp = FloatProgress(min=0,max=epochs)  
		    display(fp)
		loss_history = []
		for k in range(epochs):
			# Split data into batches
			np.random.shuffle(data_shuffled)
			n_batches = int(np.floor( len(data[0]) / batch_size ))
			idx = [n*batch_size for n in range(n_batches+1)]
			for batch_number in range(n_batches):
				if batch_number<=n_batches-1:
					mini_batch = data_shuffled[idx[batch_number]:idx[batch_number+1]]
				else:
					mini_batch = data_shuffled[idx[batch_number]:]
				for x, t in mini_batch:
					if not(dropout):
						self.FeedForward(x)
					else:
						self.FeedForward_dropout(x)
					self.BackProp_no_increment(t, decay=decay)
				self.IncrementParameters(lrate=lrate)
			#if progress and np.mod(k,100)==0:
			if progress:
				fp.value += 1

			loss_history.append([k, self.Evaluate(data)])

		if progress:
			return np.array(loss_history)
		else:
			return

	def __SGDv__(self, data, lrate=0.05, epochs=1, progress=True, batch_size=10, momentum=0.):
		data_shuffled = list(zip(data[0], data[1]))
		if progress:
		    fp = FloatProgress(min=0,max=epochs)  
		    display(fp)
		loss_history = []
		for k in range(epochs):
			# Split data into batches
			np.random.shuffle(data_shuffled)
			n_batches = int(np.floor( len(data[0]) / batch_size ))
			idx = [n*batch_size for n in range(n_batches+1)]
			for batch_number in range(n_batches):
				if batch_number<=n_batches-1:
					mini_batch = data_shuffled[idx[batch_number]:idx[batch_number+1]]
				else:
					mini_batch = data_shuffled[idx[batch_number]:]
				for x, t in mini_batch:
					self.FeedForward(x)
					self.BackProp_no_increment(t, decay=0., momentum=momentum)
				self.IncrementParameters(lrate=lrate)
			#if progress and np.mod(k,100)==0:
			if progress:
				fp.value += 1

			loss_history.append([k, self.Evaluate(data)])

		if progress:
			return np.array(loss_history)
		else:
			return





# end