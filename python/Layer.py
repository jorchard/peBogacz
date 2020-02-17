# -*- coding: utf-8 -*-
"""
Layer base class

@author: jorchard
"""

import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda:0") # Uncomment this to run on GPU
else:
    device = torch.device("cpu")

class PELayer:
    '''
    Later structure is
       ==> [(v)<-->(e)] <==> [(v)<-->(e)] <==
    '''

    def __init__(self, n=0):
        self.layer_type = 'PE'

        self.n = n  # Number of nodes in layer
        self.idx = []

        # Node activities
        # Allocated dynamically to accommodate different batch sizes
        self.v = []
        self.e = []
        self.dvdt = []
        self.dedt = []

        # Node biases
        if n>0:
            self.b = torch.zeros(self.n, device=device)
            self.dbdt = torch.FloatTensor(self.n).zero_().to(device)
        else:
            self.b = []
            self.dbdt = []

        # Error node variance for feedback
        self.variance = 1.

        # Misc. parameters
        self.is_input = False
        self.is_top = False
        self.is_rf = False
        self.layer_above = []
        self.layer_below = []

        self.alpha = torch.ones(self.n, device=device)  # FF weight (-> v from e below)
        self.beta = torch.ones(self.n, device=device)   # FB influence (v<- from corresponding e)
        self.Sigma = 1.
        self.v_decay = 0.

        self.tau = 0.1
        self.probe_on = False
        self.v_history = []
        self.e_history = []
        self.batch_size = 0


    def Save(self, fp):
        np.save(fp, self.n)
        np.save(fp, self.is_input)
        np.save(fp, self.is_top)
        np.save(fp, self.is_rf)
        np.save(fp, self.variance)
        np.save(fp, self.alpha.cpu())
        np.save(fp, self.beta.cpu())
        np.save(fp, self.tau.cpu())
        np.save(fp, self.b.cpu())


    def Load(self, fp):
        self.n = np.asscalar( np.load(fp) )
        self.is_input = np.asscalar( np.load(fp) )
        self.is_top = np.asscalar( np.load(fp) )
        self.is_rf = np.asscalar( np.load(fp))
        self.variance = np.asscalar( np.load (fp) )
        self.alpha = torch.tensor( np.load(fp) ).float().to(device)
        self.beta = torch.tensor( np.load(fp) ).float().to(device)
        self.tau = torch.tensor( np.load(fp) ).float().to(device)
        self.b = torch.tensor( np.load(fp) ).float().to(device)
        self.dbdt = torch.zeros_like( self.b ).float().to(device)

    def Allocate(self, batch_size=1):
        if batch_size!=self.batch_size or self.batch_size==0:
            self.batch_size = batch_size
            del self.v, self.e, self.dvdt, self.dedt, self.v_history, self.e_history
            self.v_history = []
            self.e_history = []

            self.v = torch.zeros([batch_size, self.n], device=device)
            self.e = torch.zeros([batch_size, self.n], device=device)
            self.dvdt = torch.zeros([batch_size, self.n], device=device)
            self.dedt = torch.zeros([batch_size, self.n], device=device)
            self.dbdt = torch.zeros([batch_size, self.n], device=device)

    def Release(self):
        del self.v, self.e, self.dvdt, self.dedt, self.b, self.variance

    def SetBias(self, b):
        for k in range(len(self.b)):
            self.b[k] = b[k]

    def ShowState(self):
        print(str(self.idx)+':  v = '+str(np.array(self.v.cpu())))

    def ShowError(self):
        print(str(self.idx)+':  e = '+str(np.array(self.e.cpu())))

    def ShowBias(self):
        print('  b = '+str(np.array(self.b.cpu())))

    def PEError(self):
        return torch.sum(self.e**2) / self.batch_size / self.n

    def Reset(self):
        del self.v_history, self.e_history
        self.v_history = []
        self.e_history = []
        if isinstance(self.v, (torch.Tensor)):
            self.v.zero_()
        if isinstance(self.e, (torch.Tensor)):
            self.e.zero_()

    def Clamp(self):
        self.alpha = torch.zeros(self.n).float().to(device)
        self.beta  = torch.zeros(self.n).float().to(device)

    def SetFF(self):
        self.alpha = torch.ones(self.n).float().to(device)
        self.beta  = torch.zeros(self.n).float().to(device)

    def SetFB(self):
        self.alpha = torch.zeros(self.n).float().to(device)
        self.beta  = torch.ones(self.n).float().to(device)

    def SetBidirectional(self):
        self.alpha = torch.ones(self.n).float().to(device)
        self.beta  = torch.ones(self.n).float().to(device)

    def SetFixed(self):
        self.alpha = torch.zeros(self.n).float().to(device)
        self.beta  = torch.zeros(self.n).float().to(device)

    def SetVariance(self, variance=1.0):
        self.variance = variance

    def Step(self, dt=0.01, set_error=False):
        k = dt/self.tau
        self.v = self.v + k*self.dvdt
        if set_error:
            self.e = self.dedt
        else:
            self.e = self.e + k*self.dedt
        self.dvdt.zero_()
        self.dedt.zero_()

    def IncrementValue(self):
        self.v = self.v + self.dvdt
        self.dvdt.zero_()

    def Probe(self, bool):
        self.probe_on = bool

    def Record(self):
        if self.probe_on:
            self.v_history.append(np.array(self.v.cpu()))
            self.e_history.append(np.array(self.e.cpu()))


#***************************************************
#
#  InputPELayer
#
#***************************************************
class InputPELayer(PELayer):

    def __init__(self, n=10):
        PELayer.__init__(self, n=n)
        self.layer_type = 'Input'

        self.is_input = True
        self.is_top = False
        self.is_rf = False

    def Allocate(self, batch_size=1):
        old_batch_size = self.batch_size
        PELayer.Allocate(self, batch_size=batch_size)

    def SetInput(self, x):
        self.v = x.clone().detach()

    def Record(self):
        self.v_history.append(np.array(self.v.cpu()))
        self.e_history.append(np.array(self.e.cpu()))

    def FeedForwardFromError(self):
        self.dvdt -= self.beta*self.e + self.v_decay*self.beta*self.v

#***************************************************
#
#  TopPELayer
#
#***************************************************
class TopPELayer(PELayer):

    def __init__(self, n=10):
        PELayer.__init__(self, n=n)
        self.layer_type = 'Top'

        self.is_top = True
        self.is_input = False
        self.is_rf = False
        self.expectation = [] # container for constant input
        self.beta = torch.zeros(self.n, device=device)   # FB influence (v<- from corresponding e)

    def Allocate(self, batch_size=1):
        old_batch_size = self.batch_size
        PELayer.Allocate(self, batch_size=batch_size)
        if batch_size!=old_batch_size:
            if old_batch_size!=0:
                del self.expectation
            if batch_size==1:
                self.expectation = torch.zeros([self.n], dtype=torch.float, device=device)
            else:
                self.expectation = torch.zeros([batch_size, self.n], dtype=torch.float, device=device)

    def Reset(self):
        PELayer.Reset(self)
        if isinstance(self.expectation, (torch.Tensor)):
            self.expectation.zero_()

    def SetExpectation(self, t):
        self.expectation = t.clone().detach()

    def Record(self):
        self.v_history.append(np.array(self.v.cpu()))

    def PropagateExpectationToError(self):
        self.dedt = self.v - self.expectation - self.Sigma*self.e

#***************************************************
#
#  RetinotopicLayer
#
#***************************************************
class RetinotopicPELayer(PELayer):

    def __init__(self, imsize=(1,1), channels=1, receptive_field=1, receptive_field_spacing=1, padding=0):
        '''
        Initializes a retinotopic (receptive fields) layer based on the size of the input to this layer,
        the number of channels, the receptive_field size, and receptive_field_spacing size.

        imsize: tuple containing (height, width) of layer that inputs to this layer.
        channels: number of channels
        receptive_field: tuple containing (height, width) of receptive_field or an integer
        receptive_field_spacing: distance between each receptive field size
        '''
        self.channels = channels

        if (type(receptive_field) is int):
            self.receptive_field = (receptive_field, receptive_field)
        else:
            self.receptive_field = receptive_field

        self.receptive_field_spacing = receptive_field_spacing

        # Calculate the output shape of the image using the receptive_field_spacing and receptive_field size
        # along with image dimensions
        self.height = imsize[0]
        self.width = imsize[1]

        #Calculate number of neurons in this layer
        self.n = self.height*self.width*self.channels
        PELayer.__init__(self, n=self.n)

        self.layer_type = 'RF'

        #Other parameters
        self.is_top = False
        self.is_input = False
        self.is_rf = True

        self.b = torch.randn(self.channels, imsize[0]*imsize[1], dtype=torch.float32, device=device) / np.sqrt(self.channels*self.height)
        self.dbdt = torch.zeros_like( self.b ).float().to(device)

        self.alpha = torch.ones(self.channels, self.height, self.width, dtype=torch.float32, device=device)
        self.beta = torch.ones(self.channels, self.height, self.width, dtype=torch.float32, device=device)

        self.expectation = []

    def Allocate(self, batch_size=1):
        old_batch_size = self.batch_size

        if batch_size != old_batch_size:
            self.batch_size = batch_size
            del self.v, self.e, self.dvdt, self.dedt, self.v_history, self.e_history, self.expectation
            self.v_history = []
            self.e_history = []

            self.v = torch.zeros([batch_size, self.channels, self.height, self.width], dtype=torch.float32, device=device)
            self.e = torch.zeros([batch_size, self.channels, self.height, self.width], dtype=torch.float32, device=device)
            self.dvdt = torch.zeros([batch_size, self.channels, self.height, self.width], dtype=torch.float32, device=device)
            self.dedt = torch.zeros([batch_size, self.channels, self.height, self.width], dtype=torch.float32, device=device)
            self.expectation = torch.zeros([batch_size, self.channels, self.height, self.width], dtype=torch.float32, device=device)

    def Reset(self):
        PELayer.Reset(self)

        if isinstance(self.e, (torch.Tensor)):
            self.e.zero_()
        if isinstance(self.v, (torch.Tensor)):
            self.v.zero_()
        if isinstance(self.expectation, (torch.Tensor)):
            self.expectation.zero_()

    def Record(self):
        self.v_history.append(np.array(self.v.cpu()))

    def Save(self, fp):
        np.save(fp, self.channels)
        np.save(fp, self.receptive_field)
        np.save(fp, self.receptive_field_spacing)
        np.save(fp, self.height)
        np.save(fp, self.width)
        np.save(fp, self.n)
        np.save(fp, self.is_input)
        np.save(fp, self.is_top)
        np.save(fp, self.is_rf)
        np.save(fp, self.variance)
        np.save(fp, self.alpha.cpu())
        np.save(fp, self.beta.cpu())
        np.save(fp, self.tau.cpu())
        np.save(fp, self.b.cpu())


    def Load(self, fp):
        self.channels = np.asscalar( np.load(fp) )
        self.receptive_field = np.load(fp)
        self.receptive_field = (int(self.receptive_field[0]), int(self.receptive_field[1]))
        self.receptive_field_spacing = np.asscalar( np.load(fp) )
        self.height = np.asscalar( np.load(fp) )
        self.width = np.asscalar( np.load(fp) )
        self.n = np.asscalar( np.load(fp) )
        self.is_input = np.asscalar( np.load(fp) )
        self.is_top = np.asscalar( np.load(fp) )
        self.is_rf = np.asscalar( np.load(fp))
        self.variance = np.asscalar( np.load (fp) )
        self.alpha = torch.tensor( np.load(fp) ).float().to(device)
        self.beta = torch.tensor( np.load(fp) ).float().to(device)
        self.tau = torch.tensor( np.load(fp) ).float().to(device)
        self.b = torch.tensor( np.load(fp) ).float().to(device)
        self.dbdt = torch.zeros_like( self.b ).float().to(device)

    def SetExpectation(self, t):
        self.expectation = t.clone().detach()

    def Clamp(self):
        self.alpha = torch.zeros(self.channels, self.height, self.width).float().to(device)
        self.beta  = torch.zeros(self.channels, self.height, self.width).float().to(device)

    def SetFF(self):
        self.alpha = torch.ones(self.channels, self.height, self.width).float().to(device)
        self.beta  = torch.zeros(self.channels, self.height, self.width).float().to(device)

    def SetFB(self):
        self.alpha = torch.zeros(self.channels, self.height, self.width).float().to(device)
        self.beta  = torch.ones(self.channels, self.height, self.width).float().to(device)

    def SetBidirectional(self):
        self.alpha = torch.ones(self.channels, self.height, self.width).float().to(device)
        self.beta  = torch.ones(self.channels, self.height, self.width).float().to(device)

    def SetFixed(self):
        self.alpha = torch.zeros(self.channels, self.height, self.width).float().to(device)
        self.beta  = torch.zeros(self.channels, self.height, self.width).float().to(device)

    def SetIsTopRetinotopic(self):
        self.is_top = True

    def PropagateExpectationToError(self):
        self.dedt = self.v - self.expectation - self.Sigma*self.e
