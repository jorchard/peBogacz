# NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt
import Layer
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from IPython.display import display
from ipywidgets import FloatProgress
from abc import ABC, abstractmethod

dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda:0") # Uncomment this to run on GPU
else:
    device = torch.device("cpu")

#============================================================
#
# Abstract Connection class
#
#============================================================
class Connection(object):

    def __init__(self, b=None, a=None, act=None):
        '''
        Connection(b, a, act=None, W=None, M=None, symmetric=False)

        Create an abstract connection between two layers.

        Inputs:
          b is the layer (object) below
          a is the layer (object) above
          act is the activation function, 'tanh', 'logistic', 'identity', 'softmax'
        '''
        ltypes = (Layer.PELayer, Layer.InputPELayer, Layer.TopPELayer, Layer.RetinotopicPELayer)
        if isinstance(b, ltypes) or isinstance(a, ltypes):
            self.below = b
            self.above = a

            self.below_idx = self.below.idx
            self.above_idx = self.above.idx

            af_given = hasattr(act, '__len__')

            self.W = None
            self.M = None
            self.dWdt = None
            self.dMdt = None
            self.lam = 0.

            if af_given:
                self.SetActivationFunction(act)
                print(act)
            else:
                self.activation_function = 'tanh'
                self.sigma = tanh
                self.sigma_p = tanh_p
        else:
            self.below = []
            self.above = []
            self.below_idx = -99
            self.above_idx = -99
            self.activation_function = 'tanh'
            self.sigma = tanh
            self.sigma_p = tanh_p

        self.learn = True


    def SetM(self, M):
        self.M = torch.tensor(M).float().to(device)

    def SetW(self, W):
        self.W = torch.tensor(W).float().to(device)

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

    def Load(self, fp):
        W_decay = np.asscalar( np.load(fp) )
        M_decay = np.asscalar( np.load(fp) )
        self.SetDecay(W_decay=W_decay, M_decay=M_decay)

        learn = np.asscalar( np.load(fp) )
        self.learn = learn

        activation_function = str( np.load(fp) )
        self.SetActivationFunction(activation_function)

    def UpdateWeights(self, k, batch_size, update_M=True, update_W=True):
        '''
        Updates self.W and self.M.

        Inputs:
            k: learning rate
            batch_size: size of batch in network
        '''
        if update_M:
            self.M = self.M + k*self.dMdt/batch_size - k*self.lam*self.M

        if update_W:
            self.W = self.W + k*self.dWdt/batch_size - k*self.lam*self.W

    def UpdateBias(self, k, batch_size):
        self.below.b = self.below.b + k*self.below.dbdt/batch_size

    def ResetGradients(self):
        self.dMdt.zero_()
        self.dWdt.zero_()
        self.below.dbdt.zero_()

    def SetValueNode(self):
        '''
        Sets value node of layer below in the style of direct feed forward.
        '''

        self.below.v = self.FeedForward()

    def SetErrorNode(self, below_value):
        '''
        Sets error node of layer below via below.v - feedforward(above.v)
        '''

        FF_result = self.FeedForward()

        self.below.e = (below_value - FF_result) #/ self.below.variance

    @abstractmethod
    def FeedForward(self):
        '''
        Performs a feedforward operation, layer.v @ self.M + below.b

        '''
        pass

    @abstractmethod
    def FeedBack(self, error):
        '''
        Performs a feedback operation, layer.e @ self.W

        '''
        pass

    @abstractmethod
    def UpdateIncrementToValue(self, above_error, beta_time=0.2):
        '''
        Updates dvdt of layer above.
        '''
        pass

    @abstractmethod
    def UpdateIncrementToError(self, below_value):
        '''
        Updates dedt of layer below.
        '''
        pass

    @abstractmethod
    def UpdateIncrementToWeights(self, update_feedback=True):
        '''
        Hebbian update to dMdt and dWdt.

        Inputs:
            Updates dWdt if update_feedback is True.
        '''
        pass

    @abstractmethod
    def UpdateIncrementToBiases(self):
        '''
        Hebbian update to below.dbdt
        '''
        pass

#============================================================
#
# RetinotopicConnection class
#
#============================================================
class RetinotopicConnection(Connection):

    def __init__(self, b=None, a=None, act=None, W=None, M=None, symmetric=False, shared=False):
        '''
        RetinotopicConnection(b, a, act=None, W=None, M=None, symmetric=False)

        Create a retinotopic (receptive fields) connection with unshared weights.

        Inputs:
          b is the layer (object) below
          a is the later (object) above
          act is the activation function, 'tanh', 'logistic', 'identity', 'softmax'
          W and M can be supplied (optional)
          symmetric is True if you want W = M^T
          shared is True for receptive fields with shared weights, in which case, use RetinotopicSharedWeightsConnection
        '''
        super().__init__(b=b, a=a, act=act)

        ltypes = (Layer.PELayer, Layer.InputPELayer, Layer.TopPELayer, Layer.RetinotopicPELayer)
        if isinstance(b, ltypes) or isinstance(a, ltypes):
            self.in_channels = self.above.channels
            self.out_channels = self.below.channels

            self.receptive_field = self.below.receptive_field

            self.receptive_field_spacing = self.below.receptive_field_spacing

            self.shared=False

            # Calculate the output shape of the image using the receptive_field_spacing and receptive_field size
            # along with image dimensions
            hp = self.below.height
            wp = self.below.width

            # This weight axis will have to transform from in_channels*K[0]*K[1]
            # to out_channels*1*1
            D1 = self.in_channels*self.receptive_field[0]*self.receptive_field[1]

            # Parameters required for feedback overlapping receptive field windows' normalization
            self.above_output_size = (self.above.height, self.above.width)
            self.above_receptive_field_area = self.above.receptive_field[0]*self.above.receptive_field[1]
            self.num_of_sliding_blocks = (self.above_output_size[0] - self.above.receptive_field[0] + 1) * (self.above_output_size[1] - self.above.receptive_field[1] + 1)

            self.normalize_feedback=True

            ones = torch.ones((1, self.above_receptive_field_area, self.num_of_sliding_blocks))
            self.feedback_normalization = F.fold(ones, self.above_output_size, self.above.receptive_field).float().to(device)

            Wgiven = hasattr(W, '__len__')
            Mgiven = hasattr(M, '__len__')

            # Weight Matrix M will have 3 axes
            # [D1, out_channels, out_image_size]
            # D1: number of channels in input image * height of receptive_field * width of receptive_field
            # out_channels: number of channels in output image
            # out_image_size: the size of the output image, calculated from receptive_field size and receptive_field_spacing across input image

            if Mgiven:
                self.M = torch.tensor(M).float().to(device)
            else:
                self.M = torch.randn(D1, self.out_channels, hp*wp, dtype=torch.float32, device=device) / np.sqrt(hp*wp)# / 10

            # Weight Matrix W will have 3 axes
            # [out_channels, D1, out_image_size]

            if Wgiven:
                self.W = torch.tensor(W).float().to(device)
            elif symmetric:
                self.W = deepcopy(self.M.transpose(1,0))
            else:
                self.W = torch.randn(self.out_channels, D1, hp*wp, dtype=torch.float32, device=device) / np.sqrt(hp*wp)# / 10

            # The input is an image of shape BxCxHxW
            # the output is of shape B x C*K[0]*K[1] x L
            # L is number of blocks extracted; in this case L = hp*wp

            self.FF_unfold = nn.Unfold(self.receptive_field, stride=self.receptive_field_spacing)
            self.FF_fold = nn.Fold((hp,wp), (1,1), stride=1)

            self.FB_unfold = nn.Unfold((1,1), stride=1)
            self.FB_fold = nn.Fold((self.above.height, self.above.width), self.receptive_field, stride=self.receptive_field_spacing)

            self.dMdt = torch.zeros( D1, self.out_channels, hp*wp, dtype=torch.float32, device=device )
            self.dWdt = torch.zeros( self.out_channels, D1, hp*wp, dtype=torch.float32, device=device )

            self.M_decay = 0.0
            self.W_decay = 0.0

        else:
            self.W = []
            self.M = []
            self.dWdt = []
            self.dMdt = []
            self.W_decay = -99
            self.M_decay = -99
            self.lam = -99

    def Save(self, fp):
        np.save(fp, self.below_idx)
        np.save(fp, self.above_idx)
        np.save(fp, self.W.cpu())
        np.save(fp, self.M.cpu())

        #NeuralNetwork.Load needs this for Connect
        np.save(fp, self.shared)

        np.save(fp, self.W_decay)
        np.save(fp, self.M_decay)
        np.save(fp, self.learn)
        np.save(fp, self.activation_function)

    def FeedForward(self):
        '''
        Performs a feedforward operation, above.v @ self.M + below.bias
        '''
        # Ex. Input image: [10, 3, 32, 32] --> [batch_size, input_channels, height, width]
        v_unfolded = self.FF_unfold(self.sigma(self.above.v))
        # After unfolding with receptive_field=(5,5), receptive_field_spacing=1, input image: [10, 75, 784]

        # Image: [10, 75, 784]
        # Weight Matrix: [75, 4, 784]
        # Einsum: multiply the second axis of Image with the first axis of Weight Matrix and
        #         the third axis of Image with the third axis of Weight Matrix, summing the unused axes,
        #         and output a tensor with dimensions
        #         [the first axis of Image, the second axis of Weight Matrix, the third axis of Image/Weight Matrix]
        FF_result = torch.einsum('ijk,jlk->ilk', [v_unfolded, self.M]) + self.below.b
        # Output: [10, 4, 784] --> [batch_size, output_channels, height*width]

        # Ex. Input to FF_fold: [10, 4, 784]
        result_folded = self.FF_fold(FF_result)
        # After folding with receptive_field=(1,1), receptive_field_spacing=1
        # output image: [10, 4, 28, 28] --> [batch_size, output_channels, height, width]

        return result_folded

    def FeedBack(self):
        '''
        Performs a feedback operation, below.e @ self.W
        '''
        # Ex. Input image: [10, 4, 28, 28] --> [batch_size, output_channels, height, width]
        e_unfolded = self.FB_unfold(self.below.e)
        # After unfolding with receptive_field=(1,1), receptive_field_spacing=1, input image: [10, 4, 784]

        # Image: [10, 4, 784]
        # Weight Matrix: [4, 75, 784]
        # Einsum: multiply the second axis of Image with the first axis of Weight Matrix and
        #         the third axis of Image with the third axis of Weight Matrix, summing the unused axes,
        #         and output a tensor with dimensions:
        #         [the first axis of Image, the second axis of Weight Matrix, the third axis of Image/Weight Matrix]
        FB_result = torch.einsum('ijk,jlk->ilk',[e_unfolded, self.W])
        # Output: [10, 75, 784] --> [batch_size, input_channels, height*width]

        # Ex. Input to FB_fold: [10, 75, 784]
        result_folded = self.FB_fold(FB_result) * self.sigma_p(self.above.v)
        # After folding with receptive_field=(5,5), receptive_field_spacing=1
        # output image: [10, 3, 32, 32] --> [batch_size, input_channels, height, width]

        # Normalize the overlapping windows effect of feedback
        if self.normalize_feedback:
            result_folded = result_folded / self.feedback_normalization

        return result_folded

    def UpdateIncrementToValue(self, above_error, beta_time=0.2, FB_weight=1.0):
        '''
        Updates dvdt of layer above.

        Inputs:
            above_error: self.above.e
        '''

        FB_result = self.FeedBack()

        self.above.dvdt = beta_time*( self.above.alpha*FB_weight*FB_result - self.above.beta*above_error )

        if self.above.is_top:
            self.above.dvdt = self.above.dvdt - beta_time*self.above.v_decay*self.above.alpha*self.above.v
        else:
            self.above.dvdt = self.above.dvdt - beta_time*self.above.v_decay*self.above.v

    def UpdateIncrementToError(self, below_value):
        '''
        Updates dedt of layer below.

        Inputs:
            below_value: self.below.v
        '''
        FF_result = self.FeedForward()

        self.below.dedt = (below_value - FF_result - self.below.Sigma*self.below.e) #/ self.below.variance

    def UpdateIncrementToWeights(self, update_feedback=True):
        '''
        Hebbian update to dMdt and dWdt.

        Inputs:
            update_feedback: Updates dWdt if True
        '''
        # Ex. above.v: [10, 3, 32, 32] --> [batch_size, input_channels, height, width]
        above_value_nodes = self.FF_unfold(self.sigma(self.above.v))
        # After unfolding with receptive_field=(5,5), receptive_field_spacing=1: [10, 75, 784]

        # Ex. below.e: [10, 4, 28, 28] --> [batch_size, output_channels, height, width]
        below_error_nodes = self.FB_unfold(self.below.e)
        # After unfolding with receptive_field=(1,1), receptive_field_spacing=1: [10, 4, 784]

        # Ex. Given above.v: [10, 75, 784] and below.e: [10, 4, 784]
        # Einsum: Multiply the first axes and the third axes of above.v and below.e, summing the unused axes, and
        #         output a tensor with dimensions:
        #         [second axis of above.v, second axis of below.e, and third axis of above.v/below.e]
        self.dMdt = torch.einsum('ijk,ilk->jlk',[above_value_nodes, below_error_nodes])
        # Ex. self.dMdt.shape = [75, 4, 784]

        if update_feedback:
            # Ex. Given above.v: [10, 75, 784] and below.e: [10, 4, 784]
            # Einsum: Multiply the first axes and the third axes of above.v and below.e, summing the unused axes, and
            #         output a tensor with dimensions:
            #         [second axis of below.e, second axis of above.v, and third axis of above.v/below.e]
            self.dWdt = torch.einsum('ijk,ilk->ljk',[above_value_nodes, below_error_nodes])
            # Ex. self.dWdt.shape = [4, 75, 784]


    def UpdateIncrementToBiases(self):
        '''
        Hebbian update to below.dbdt
        '''
        # Ex. below.e: [10, 4, 28, 28]
        below_error_nodes = self.FB_unfold(self.below.e)
        # After unfolding with receptive_field=(1,1), receptive_field_spacing=1: [10, 4, 784]

        # Sum along axis 0
        self.below.dbdt = torch.sum(below_error_nodes, dim=0)
        # Given below_error_nodes dimensions of [10, 4, 784], summing along axis 0 gives:
        # below.dbdt: [4, 784]

#============================================================
#
# RetinotopicSharedWeightsConnection class
#
#============================================================
class RetinotopicSharedWeightsConnection(RetinotopicConnection):

    def __init__(self, b=None, a=None, act=None, W=None, M=None, symmetric=False, shared=True):
        '''
        RetinotopicSharedWeightsConnection(b, a, act=None, W=None, M=None, symmetric=False)

        Create a retinotopic (receptive field) connection with shared weights.

        Inputs:
          b is the layer (object) below
          a is the later (object) above
          act is the activation function, 'tanh', 'logistic', 'identity', 'softmax'
          W and M can be supplied (optional)
          symmetric is True if you want W = M^T
          shared is True for convolutions with shared weights
        '''
        super().__init__(b=b, a=a, act=act, W=W, M=M, symmetric=symmetric)

        self.shared=True

        Wgiven = hasattr(W, '__len__')
        Mgiven = hasattr(M, '__len__')

        D1 = self.in_channels*self.receptive_field[0]*self.receptive_field[1]
        hp = b.height

        if Mgiven:
            self.M = torch.tensor(M).float().to(device)
        else:
            self.M = torch.randn(D1, self.out_channels, dtype=torch.float32, device=device) / np.sqrt(hp) / 10

        if Wgiven:
            self.W = torch.tensor(W).float().to(device)
        elif symmetric:
            self.W = deepcopy(self.M.transpose(1,0))
        else:
            self.W = torch.randn(self.out_channels, D1, dtype=torch.float32, device=device) / np.sqrt(hp) / 10

        self.dMdt = torch.zeros( D1, self.out_channels, dtype=torch.float32, device=device )
        self.dWdt = torch.zeros( self.out_channels, D1, dtype=torch.float32, device=device )

    def FeedForward(self):
        '''
        Performs a feedforward operation, above.v @ self.M + below.bias
        '''
        v_unfolded = self.FF_unfold(self.sigma(self.above.v))

        FF_result = torch.einsum('ijk,jl->ilk', [v_unfolded, self.M]) + self.below.b

        result_folded = self.FF_fold(FF_result)

        return result_folded

    def FeedBack(self):
        '''
        Performs a feedback operation, below.e @ self.W
        '''
        e_unfolded = self.FB_unfold(self.below.e)

        FB_result = torch.einsum('ijk,jl->ilk',[e_unfolded, self.W])

        result_folded = self.FB_fold(FB_result) * self.sigma_p(self.above.v)

        return result_folded

    def UpdateIncrementToWeights(self, update_feedback=True):
        '''
        Hebbian update to dMdt and dWdt.

        Inputs:
            update_feedback: Updates dWdt if True
        '''
        above_value_nodes = self.FF_unfold(self.sigma(self.above.v))
        below_error_nodes = self.FB_unfold(self.below.e)

        self.dMdt = torch.einsum('ijk,ilk->jl',[above_value_nodes, below_error_nodes])

        if update_feedback:
            self.dWdt = torch.einsum('ijk,ilk->lj',[above_value_nodes, below_error_nodes])

#============================================================
#
# DenseConnection class
#
#============================================================
class DenseConnection(Connection):

    def __init__(self, b=None, a=None, act=None, W=None, M=None, symmetric=False):
        '''
        DenseConnection(b, a, act=None, W=None, M=None, symmetric=False)

        Create a dense connection between two layers.

        Inputs:
          b is the layer (object) below
          a is the later (object) abover
          act is the activation function, 'tanh', 'logistic', 'identity', 'softmax'
          W and M can be supplied (optional)
          symmetric is True if you want W = M^T
        '''
        super().__init__(b=b, a=a, act=act)

        ltypes = (Layer.PELayer, Layer.InputPELayer, Layer.TopPELayer, Layer.RetinotopicPELayer)
        if isinstance(b, ltypes) or isinstance(a, ltypes):
            Wgiven = hasattr(W, '__len__')
            Mgiven = hasattr(M, '__len__')

            if Mgiven:
                self.M = torch.tensor(M).float().to(device)
            else:
                self.M = torch.randn( a.n, b.n, dtype=torch.float32, device=device) / np.sqrt(b.n) / 2.

            if Wgiven:
                self.W = torch.tensor(W).float().to(device)
            elif symmetric:
                self.W = deepcopy(self.M.transpose(1,0))
            else:
                self.W = torch.randn( b.n, a.n, dtype=torch.float32, device=device) / np.sqrt(b.n) / 2.
                # This distribution is taken from the caption of Fig. 6 in Wittington and Bogacz
                #self.W = ( torch.rand( b.n, a.n, dtype=torch.float32, device=device) - 0.5 ) *8*np.sqrt(6)/np.sqrt(a.n+b.n)

            self.dMdt = torch.zeros( a.n, b.n, dtype=torch.float32, device=device)
            self.dWdt = torch.zeros( b.n, a.n, dtype=torch.float32, device=device)

            self.M_decay = 0.0
            self.W_decay = 0.0

        else:
            self.W = []
            self.M = []
            self.dWdt = []
            self.dMdt = []
            self.W_decay = -99
            self.M_decay = -99
            self.lam = -99

    def FeedForward(self):
        '''
        Performs a feedforward operation, above.v @ self.M + below.b

        '''

        #Regular feedforward operation between dense layers
        return self.sigma(self.above.v) @ self.M + self.below.b

    def FeedBack(self):
        '''
        Performs a feedback operation, below.e @ self.W

        '''

        return self.below.e @ self.W * self.sigma_p(self.above.v)

    def UpdateIncrementToValue(self, above_error, beta_time=0.2, FB_weight=1.0):
        '''
        Updates dvdt of layer above.

        Inputs:
            above_error: self.above.e
        '''

        FB_result = self.FeedBack()

        self.above.dvdt = beta_time*( self.above.alpha*FB_weight*FB_result - self.above.beta*above_error )

        if self.above.is_top:
            self.above.dvdt = self.above.dvdt - beta_time*self.above.v_decay*self.above.alpha*self.above.v
        else:
            self.above.dvdt = self.above.dvdt - beta_time*self.above.v_decay*self.above.v

    def UpdateIncrementToError(self, below_value):
        '''
        Updates dedt of layer below.

        Inputs:
            below_value: self.below.v
        '''
        FF_result = self.FeedForward()

        self.below.dedt = (below_value - FF_result - self.below.Sigma*self.below.e)

    def UpdateIncrementToWeights(self, update_feedback=True):
        '''
        Hebbian update to dMdt and dWdt.
        '''
        self.dMdt = self.sigma(self.above.v).transpose(1,0) @ self.below.e

        if update_feedback:
            self.dWdt = self.below.e.transpose(1,0) @ self.sigma(self.above.v)

    def UpdateIncrementToBiases(self):
        '''
        Hebbian update to below.dbdt

        Inputs:
            Updates dWdt if update_feedback is True.
        '''
        self.below.dbdt = torch.sum(self.below.e, dim=0)

#============================================================
#
# RetinotopicToDenseConnection class
#
#============================================================
class RetinotopicToDenseConnection(DenseConnection):

    def __init__(self, b=None, a=None, act=None, W=None, M=None, symmetric=False):
        '''
        DenseConnection(b, a, act=None, W=None, M=None, symmetric=False)

        Create a connection from an RetinotopicPELayer to a dense (PE or Input) layer.

        Inputs:
          b is the layer (object) below
          a is the later (object) abover
          act is the activation function, 'tanh', 'logistic', 'identity', 'softmax'
          W and M can be supplied (optional)
          symmetric is True if you want W = M^T
        '''
        super().__init__(b=b, a=a, act=act, W=W, M=M, symmetric=symmetric)

        self.FF_unfold = nn.Unfold((1,1), stride=1)
        self.FB_fold = nn.Fold((a.height, a.width), (1,1), stride=1)

    def FeedForward(self):
        '''
        Performs a feedforward operation, above.v @ self.M + below.b

        '''

        #Convert receptive field into 1D layer
        RF_unfolded = self.FF_unfold(self.sigma(self.above.v))
        above_value_node = torch.reshape(RF_unfolded, [self.above.batch_size, self.above.n])

        return above_value_node @ self.M + self.below.b

    def FeedBack(self):
        '''
        Performs a feedback operation, below.e @ self.W

        '''
        FB_result = self.below.e @ self.W

        FB_result = torch.reshape(FB_result, (self.above.batch_size, self.above.channels, self.above.height*self.above.width))
        FB_result = self.FB_fold(FB_result) * self.sigma_p(self.above.v)

        return FB_result

    def UpdateIncrementToWeights(self, update_feedback=True):
        '''
        Hebbian update to dMdt and dWdt.
        '''

        RF_unfolded = self.FF_unfold(self.sigma(self.above.v))
        above_value_nodes = torch.reshape(RF_unfolded, [self.above.batch_size, self.above.n])

        self.dMdt = torch.einsum('ij,ik->jk',[above_value_nodes, self.below.e])

        if update_feedback:
            self.dWdt = torch.einsum('ij,ik->kj',[above_value_nodes, self.below.e])


    '''
    These functions perform their respective operations for the network scheme in which the last retinotopic layer
    has a receptive field the size of the previous layer and high number of channels, with each channel behaving as
    a single neuron for the next, fully connected layer
    '''

    def FeedForwardQ(self):
        '''
        Performs a feedforward operation, above.v @ self.M + below.b

        '''

        #Convert receptive field into 1D layer

        # [10, 600, 1, 1] ->  [10, 600]
        retinotopic_layer_flattened = torch.reshape(self.sigma(self.above.v), (self.above.batch_size, self.above.n))

        return retinotopic_layer_flattened @ self.M + self.below.b

    def FeedBackQ(self):
        '''
        Performs a feedback operation, below.e @ self.W

        '''
        FB_result = self.below.e @ self.W
        # [10, 600] -> [10, 600, 1, 1]
        FB_result = torch.reshape(FB_result, [self.above.batch_size, self.above.n, 1, 1])

        FB_result = FB_result #* self.sigma_p(self.above.v)

        return FB_result

    def UpdateIncrementToWeightsQ(self, update_feedback=True):
        '''
        Hebbian update to dMdt and dWdt.
        '''

        above_value_nodes = torch.reshape(self.sigma(self.above.v), (self.above.batch_size, self.above.n))

        self.dMdt = torch.einsum('ij,ik->jk',[above_value_nodes, self.below.e])

        if update_feedback:
            self.dWdt = torch.einsum('ij,ik->kj',[above_value_nodes, self.below.e])

#=============================================================
#
# NeuralNetwork class
#
#============================================================
class NeuralNetwork(object):

    '''
    Initialization and allocation
    '''
    def __init__(self):
        self.layers = []
        self.n_layers = 0
        self.connections = []
        self.cross_entropy = False
        self.weight_decay = 0.
        self.t = 0.
        self.t_runstart = 0.
        self.t_last_weight_update = 0.
        self.learning_blackout = 1.0 # how many seconds to wait before turning learning on
        self.blackout_interval = 0.0 #how many seconds to wait between weight updates after learning blackout has passed
        self.t_history = []
        self.learning_tau = 2.
        self.l_rate = 0.2
        self.learn = False
        self.learn_weights = True
        self.learn_biases = True
        self.batch_size = 0
        self.probe_on = False
        self.update_top_layer_in_overwrite_states = False
        self.rms_history = []
        self.pe_error_history = []
        self.test_accuracy_history = []
        self.train_error_history = []

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
            for idx,l in enumerate(self.layers):
                l.Allocate(batch_size=proposed_batch_size)

    '''
    Saving and Loading
    '''

    def Save(self, fp):
        if type(fp)==str:
            fp = open(fp, 'wb')
        np.save(fp, len(self.layers))
        for l in self.layers:
            np.save(fp, l.layer_type)
            l.Save(fp)
        np.save(fp, len(self.connections))
        for c in self.connections:
            c.Save(fp)
        np.save(fp, self.test_accuracy_history)
        fp.close()

    def Load(self, fp):
        if type(fp)==str:
            fp = open(fp, 'rb')
        n_layers = np.asscalar( np.load(fp) )
        self.Release()
        for k in range(n_layers):
            layer_type = np.load(fp)

            if layer_type == 'Input':
                L = Layer.InputPELayer()
            elif layer_type == 'Top':
                L = Layer.TopPELayer()
            elif layer_type == 'RF':
                L = Layer.RetinotopicPELayer()
            else:
                L = Layer.PELayer()
            L.Load(fp)
            self.AddLayer(L)

        n_connections = np.asscalar( np.load(fp) )
        for k in range(n_connections):
            bi = np.asscalar( np.load(fp) ) # below_idx
            ai = np.asscalar( np.load(fp) ) # above_idx
            W = np.load(fp)
            M = np.load(fp)

            if (self.layers[bi].is_rf):
                shared_weights = np.load(fp)
            else:
                shared_weights = False

            self.Connect(bi, ai, W=W, M=M, symmetric=False, shared=shared_weights)
            self.connections[-1].Load(fp)
        self.test_accuracy_history = np.load(fp)
        fp.close()

    '''
    Functions for creating network connections and layers
    '''

    def Connect(self, prei, posti, act=None, W=None, M=None, symmetric=False, shared=False):
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

        if preL.is_rf:
            if shared:
                self.connections.append(RetinotopicSharedWeightsConnection(preL, postL, act=act, W=W, M=M, symmetric=symmetric))
            else:
                self.connections.append(RetinotopicConnection(preL, postL, act=act, W=W, M=M, symmetric=symmetric))
        elif postL.is_rf:
            self.connections.append(RetinotopicToDenseConnection(preL, postL, act=act, W=W, M=M, symmetric=symmetric))
        else:
            self.connections.append(DenseConnection(preL, postL, act=act, W=W, M=M, symmetric=symmetric))

    def AddLayer(self, L):
        self.layers.append(L)
        self.n_layers += 1
        L.idx = self.n_layers-1

        if L.is_rf:
            self.layers[-1].SetIsTopRetinotopic()
            if self.n_layers >= 2:
                self.layers[-2].is_top = False

    def ConnectNextLayer(self, post):
        self.AddLayer(post)
        self.Connect(self.n_layers-2, self.n_layers-1)

    def Release(self):
        for l in self.layers:
            l.Release()
            del l
        for c in self.connections:
            c.Release()
            del c
        self.layers = []
        self.connections = []

    '''
    Utiltiy functions for setting fields
    '''

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

    def SetInput(self, x):
        self.layers[0].SetInput(x)

    def SetExpectation(self, x):
        self.layers[-1].SetExpectation(x)

    def SetExpectationState(self, v):
        self.Allocate(v)
        self.layers[-1].v = torch.tensor(v).float().to(device)

    def SetBidirectional(self):
        for l in self.layers:
            l.SetBidirectional()

    def SetFF(self):
        for l in self.layers:
            l.SetFF()

    '''
    Utility functions for retrieving fields of interest
    '''

    def ShowState(self):
        for idx, layer in enumerate(self.layers):
            if layer.is_input:
                if torch.max(layer.beta)<1.e-10:
                    print('Layer '+str(idx)+' (input): *CLAMPED*')
                else:
                    print('Layer '+str(idx)+' (input):')
                layer.ShowState()
                layer.ShowError()
            elif layer.is_top:
                if torch.max(layer.alpha)<1.e-10:
                    print('Layer '+str(idx)+' (expectation): *CLAMPED*')
                else:
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

    '''
    Utility functions for resetting the network
    '''

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

    '''
    Utility functions for recording the network's training or test error
    '''

    def Cost(self, target):
        return self.layer[-1].Cost(target)

    def Record(self):
        '''
        Records the state of the network.
        '''
        if self.batch_size==1:
            self.t_history.append(self.t)
            for layer in self.layers:
                layer.Record()

    def rms_error(self, x, y):
        self.BackprojectExpectation(y.clone().detach())
        mu0 = self.GenerateSamples()
        rms = torch.sqrt(torch.mean(torch.sum(torch.pow(x-mu0, 2), 1)/np.shape(x)[1]))
        return rms

    def PEError(self):
        '''
        pe_error = NN.PEError()

        Returns the sum of the squares of all the error nodes.
        '''
        total_pe_error = 0.
        for l in self.layers:
            total_pe_error += l.PEError()
        return total_pe_error

    def dataset_accuracy(self, dataset_in, dataset_out, dataset_length):
        self.Allocate(dataset_in)
        z = self.BackprojectExpectation(dataset_in.clone().detach())

        y_classes = np.argmax(z.cpu(),1)
        t_classes = np.argmax(dataset_out.cpu(), 1)

        correct = np.count_nonzero((y_classes - t_classes)==0)
        return correct / dataset_length

    '''
    Functions that handle the network's predictive and generative (feedforward and feedback) passes.
    '''

    def Predict(self, T, x, dt=0.01):
        self.learn = False

        #Place class vector at layer 0
        self.Allocate(x)
        self.SetInput(x)

        self.SetBidirectional()

        self.layers[0].SetFF()
        self.layers[-1].SetFF()

        self.Run(T, dt=dt)

        return self.layers[-1].v

    def FastPredict(self, x, T=100, beta_time=0.2):
        '''
        y = NN.FastPredict(x, T=100)

        Run network to equilibrium with input clamped to x. This method uses the
        fast convergence method of Whittington & Bogacz [2017].
        '''
        # Set the input layer
        self.learn = False
        self.layers[0].SetInput(x)

        self.SetBidirectional()

        self.layers[0].SetFF()
        self.layers[-1].SetFF()

        self.update_top_layer_in_overwrite_states = True
        self.SetExpectation(torch.zeros_like( self.layers[-1].e ).float().to(device))

        # 3. Run infer_ps
        self.OverwriteErrors()
        for k in range(0, T):
            self.OverwriteStates(beta_time=beta_time)
            self.OverwriteErrors()

        self.update_top_layer_in_overwrite_states = False

        g = self.connections[-1].FeedBack()
        return g

    def Generate(self, T, y, dt=0.01):
        #No learning allowed
        self.learn = False

        #Set Feedback
        self.layers[0].SetBidirectional()
        self.layers[-1].SetFB()

        #Set input images at expectation container
        self.Allocate(y)
        self.SetExpectation(y.clone().detach())

        #Faster than self.Walk
        self.Run(T, dt=dt)

        return self.layers[0].v

    def GenerateSamples(self):
        mu0 = self.connections[0].FeedForward()
        return mu0

    '''
    Functions that handle the training of the network.
    '''

    def Learn(self, x, t, test=None, observe_generative=False, T=2., epochs=5, dt=0.01, batch_size=10, shuffle=True, turn_down_lam=1.0, learning_delay=1.0):
        self.Allocate(batch_size)

        self.SetBidirectional()
        self.layers[0].SetFF()

        fp = FloatProgress(min=0,max=epochs*len(x))
        display(fp)

        if test:
            test_dataset_length = len(test[0])

        for k in range(epochs):
            if turn_down_lam<1.0:
                #Turn down weight-decay and v-decay
                lam = self.connections[0].lam
                new_lam = turn_down_lam * lam

                self.SetWeightDecay(new_lam)

                vdecay = self.layers[0].v_decay
                new_vdecay = turn_down_lam * vdecay

                self.SetvDecay(new_vdecay)
                print('turning down weight-decay and v-decay by a factor of '+str(turn_down_lam))

            batches = MakeBatches(x, t, batch_size=batch_size, shuffle=shuffle)

            it = 0

            for samp in batches:
                self.Infer(T, samp[0], samp[1], dt=dt, learn=True, learning_delay=learning_delay)
                fp.value += batch_size

    def Infer(self, T, x, y, dt=0.01, learn=False, learning_delay=1.0):
        self.learn = learn

        # Clamp both ends
        self.layers[0].SetFF()
        self.layers[-1].SetFB()

        self.Allocate(x)
        self.SetInput(x)
        if (self.layers[-1].is_rf):
            self.SetExpectation(torch.unsqueeze(y.clone().detach(), dim=1))
        else:
            self.SetExpectation(y.clone().detach())

        self.Run(T, dt=dt, learning_delay=learning_delay)

    # update_W, update_M are parameters to Run. No really needed
    def Run(self, T, dt, learning_delay=1.0):
        self.probe_on = False
        for l in self.layers:
            if l.probe_on:
                self.probe_on = True

        self.t_runstart = self.t
        steps = torch.arange(self.t, self.t+T, dt)

        for t in steps:
            # if dampen_v_decay and t % dampen_every_T == 0:
            #     v_decay = self.layers[0].v_decay
            #
            #     new_vdecay = v_decay_dampener * v_decay
            #     self.SetvDecay(new_vdecay)

            update_M = update_W = t >= learning_delay

            self.t = t
            self.ResetGradients()
            self.Integrate()
            self.Step(dt, update_M, update_W)
            if self.probe_on:
                self.Record()

    def Integrate(self):
        #Conditionally (beta=1) propagate expectation, or input image, to top layer
        self.layers[-1].PropagateExpectationToError()

        #Conditionally (beta=1) propagate classification from error node to value node at input layer
        self.layers[0].FeedForwardFromError()

        # First, address only the connections between layers
        for c in self.connections:
            blw = c.below
            abv = c.above

            # e <-- v
            c.UpdateIncrementToError(blw.v)

            # e --> v
            c.UpdateIncrementToValue(abv.e)

            # Hebbian synaptic weight increments
            if c.learn==True:
                c.UpdateIncrementToWeights(update_feedback=True)
                c.UpdateIncrementToBiases()

    def Step(self, dt=0.01, update_M=True, update_W=True):
        for l in self.layers:
            l.Step(dt=dt, set_error=False)

        # Only update weights if learning is one, and we are past
        # the "blackout" transient period.
        if self.learn and self.t-self.t_runstart >= self.learning_blackout and self.t-self.t_last_weight_update >= self.blackout_interval:
            k = dt/self.learning_tau
            for c in self.connections:
                if self.learn_weights:
                    c.UpdateWeights(k, self.batch_size, update_M, update_W)
                if self.learn_biases:
                    c.UpdateBias(k, self.batch_size)
            self.t_last_weight_update = self.t

    '''
    Functions for the network's fast training method.
    '''

    def FastLearn(self, x, t, test=False, T=20, beta_time=0.2, epochs=5, Beta_one=0.9, Beta_two=0.999, ep=0.00000001, batch_size=10, noise=False, freeze=10, shuffle=True):
        '''
        Implementation of Whittington & Bogacz 2017

        '''
        if test:
            test_length = len(test[0])
            print(self.dataset_accuracy(test[0], test[1], test_length))
            self.Reset()
        train_length = len(x)

        fp = FloatProgress(min=0,max=epochs*train_length/batch_size)
        display(fp)

        #Initialize Adam parameters
        alpha = self.l_rate

        self.Allocate(batch_size)
        self.ResetGradients()

        self.SetBidirectional()
        self.layers[0].SetFF()

        freeze_W = False

        m = [] #1st momentum vector containing each layer's dMdt, dWdt, and dbdt in that order
        v = [] #2nd momentum vector with elements identical to above
        g = [] #vector of all gradients with elements identical to above

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

        time=0

        for k in range(0, epochs):
            #Remove multiplicative noise after 13 epochs
            if k > 12:
                noise = False

            if k > freeze:
                freeze_W = True

            epoch_pe_error = 0.
            batches = MakeBatches(x, t, batch_size=batch_size, shuffle=shuffle)

            for j in range(0, len(batches)):
                mb = batches[j]

                #Perform Adam while loop
                time += 1

                #1. Feedforward Pass
                if (self.layers[-1].is_rf):
                    self.BackprojectExpectation(torch.unsqueeze(mb[1], dim=1))
                else:
                    self.BackprojectExpectation(mb[1])

                # 2. Set the desired output
                # self.SetInput(mb[0])
                self.layers[0].SetInput(mb[0])

                # 3. Run infer_ps
                # This involves fixing the input in the bottom and top layers.
                self.layers[0].SetFF()  # Don't update state of bottom layer
                self.layers[-1].SetFixed() # Don't update state of top layer

                self.OverwriteErrors()
                for k in range(0, T):
                    self.OverwriteStates(beta_time=beta_time)
                    self.OverwriteErrors()

                epoch_pe_error += self.PEError()

                # 4. Calculate gradients from the error nodes
                idx = 0 #keeps track of idx in g

                for c in self.connections:
                    blw = c.below
                    abv = c.above

                    c.UpdateIncrementToWeights(update_feedback=True)
                    c.UpdateIncrementToBiases()

                    c.UpdateWeights(alpha, batch_size, update_M=True, update_W=True)
                    c.UpdateBias(alpha, batch_size)

                    #TODO: Implement Adam
                    '''
                    g[idx] = c.dMdt
                    idx += 1
                    g[idx] = c.dWdt
                    idx += 1
                    g[idx] = c.below.dbdt
                    idx += 1
                    '''
                '''
                for i in range (0, len(m), 3):
                    c = self.connections[i // 3]

                    m[i] = Beta_one*m[i] + (1 - Beta_one)*g[i]
                    v[i] = Beta_two*v[i] + (1 - Beta_two)*(g[i]*g[i])

                    m_hat = m[i] / (1 - (Beta_one**time))
                    v_hat = v[i] / (1 - (Beta_two**time))
                    c.M = c.M + alpha * (m_hat*(torch.reciprocal(torch.sqrt(v_hat) + ep)) - c.lam*c.M)

                    m[i+1] = Beta_one*m[i+1] + (1 - Beta_one)*g[i+1]
                    v[i+1] = Beta_two*v[i+1] + (1 - Beta_two)*(g[i+1]*g[i+1])

                    m_hat = m[i+1] / (1 - (Beta_one**time))
                    v_hat = v[i+1] / (1 - (Beta_two**time))

                    c.W = c.W + alpha * (m_hat*(torch.reciprocal(torch.sqrt(v_hat) + ep)) - c.lam*c.W)

                    m[i+2] = Beta_one*m[i+2] + (1 - Beta_one)*g[i+2]
                    v[i+2] = Beta_two*v[i+2] + (1 - Beta_two)*(g[i+2]*g[i+2])

                    m_hat = m[i+2] / (1 - (Beta_one**time))
                    v_hat = v[i+2] / (1 - (Beta_two**time))
                    if self.learn_biases:
                        c.below.b = c.below.b + alpha*m_hat*(torch.reciprocal(torch.sqrt(v_hat) + ep))
                '''

                self.ResetGradients()
                fp.value += 1

                if test:
                    print(self.dataset_accuracy(test[0], test[1], test_length))
                    self.Allocate(batch_size)

    def BackprojectExpectation(self, y):
        '''
        Initialize all the state nodes from the top-layer expection.

        This does not overwrite the state of layer[0].
        '''
        self.layers[-1].v = y.clone().detach()

        for idx in range(self.n_layers-2, 0, -1):
            self.connections[idx].SetValueNode()

        mu0 = self.connections[0].FeedForward()
        return mu0

    def OverwriteErrors(self):
        '''
        OverwriteErrors()

        Uses current states to overwrite the error nodes; it sets them to their equilibria, assuming
        the states are held constant.

        This method does NOT update the error nodes in the top layer.
        '''
        for idx in range(0, self.n_layers-1):
            self.connections[idx].SetErrorNode(self.layers[idx].v)

    def OverwriteStates(self, beta_time=0.01, FB_weight=1.0):
        '''
        NN.OverwriteStates()

        Updates the states, incrementing them by the incoming connections.
        Note that the layer-wise alpha and beta are used to weigh the forward and
        backward inputs, respectively.

        This method potentially updates the state nodes in ALL layers, including the bottom and top.
        '''

        self.layers[0].FeedForwardFromError()
        self.layers[0].IncrementValue()

        for idx in range(1, self.n_layers-1):
            self.connections[idx-1].UpdateIncrementToValue(self.layers[idx].e, beta_time=beta_time, FB_weight=FB_weight)
            self.layers[idx].IncrementValue()

        if self.update_top_layer_in_overwrite_states:
            self.connections[-1].UpdateIncrementToValue(self.layers[-1].e, beta_time=beta_time, FB_weight=FB_weight)
            self.layers[-1].IncrementValue()

    def UpdateConnectionWeights(self, lrate, batch_size):
        for c in self.connections:
            c.UpdateIncrementToWeights(update_feedback=True)
            c.UpdateIncrementToBiases()

            c.UpdateWeights(lrate, batch_size)
            c.UpdateBias(lrate, batch_size)

            c.ResetGradients()


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
        r = torch.randperm(N)
    batches = []
    for k in range(0, N, batch_size):
        if k+batch_size<=N:
            din = data_in[r[k:k+batch_size]]
            dout = data_out[r[k:k+batch_size]]
        else:
            din = data_in[r[k:]]
            dout = data_out[r[k:]]
        if isinstance(din, (list, tuple)):
            batches.append( [torch.stack(din, dim=0).float().to(device) , torch.stack(dout, dim=0).float().to(device)] )
        else:
            batches.append( [din.float().to(device) , dout.float().to(device)] )
    return batches

def logistic(v):
    return torch.reciprocal( torch.add( torch.exp(-v), 1.) )

def logistic_p(v):
    lv = logistic(v)
    return lv*(1.-lv)

def ReLU(v):
    return torch.clamp(v, min=0.)

def ReLU_p(v):
    return torch.clamp(torch.sign(v), min=0.)

def tanh(v):
    return torch.tanh(v)

def tanh_p(v):
    return 1. - torch.pow(torch.tanh(v),2)

def identity(v):
    return v

def identity_p(v):
    return torch.ones_like(v)

def softmax(v):
    sftmax = torch.nn.Softmax()
    return sftmax(v)

def softmax_p(v):
    z = softmax(v)
    return z*(1.-z)

def OneHot(z):
    y = np.zeros(np.shape(z))
    y[np.argmax(z)] = 1.
    return y

def DrawDigit(x, dim):
    plt.imshow(np.reshape(x.cpu(), (dim,dim)), cmap='gray')
