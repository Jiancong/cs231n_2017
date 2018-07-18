from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        (C,H,W) = input_dim
        conv_dim=(num_filters, C, filter_size, filter_size)
        
        # conv parameters
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


        conv_pad = conv_param['pad']
        conv_stride = conv_param['stride']

        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        pool_stride = pool_param['stride']

        self.params['W1'] = np.random.normal(scale=weight_scale, size=conv_dim)
        self.params['b1'] = np.zeros(num_filters)

        # conv layer
        conv_after_height = (H - filter_size + 2*conv_pad) // conv_stride + 1
        conv_after_width = (W - filter_size + 2*conv_pad) // conv_stride + 1
        conv_after_depth = num_filters

        # max pooling layer
        mp_h= (conv_after_height - pool_height) // pool_stride + 1
        mp_w= (conv_after_width -pool_width) // pool_stride + 1
        mp_d= num_filters
        print("mp_h,w,d=>", mp_h, mp_w, mp_d)

        # hidden affine layer
        W2_row_size = num_filters * input_dim[1]//2 * input_dim[2]//2
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(W2_row_size, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)

        # last affine layer
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        (N, C, W, H) = X.shape
        conv_height = (H - filter_size) + 1 #Whether stride should be account?
        conv_width = (W - filter_size) + 1

        #print("W1.shape=>", W1.shape)
        #print("b1.shape=>", b1.shape)
        #print("X.shape=>", X.shape)
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

        #print("W2.shape=>", W2.shape)
        #print("b2.shape=>", b2.shape)
        #print("out1.shape=>", out1.shape)
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        out3, cache3 = affine_forward(out2, W3, b3)
        scores = out3

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5*self.reg*np.sum(W1**2) + 0.5*self.reg*np.sum(W2**2) + 0.5*self.reg*np.sum(W3**2)
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        dx3, grads['W3'], grads['b3'] = affine_backward(dscores, cache3)
        dx2, grads['W2'], grads['b2'] = affine_relu_backward(dx3, cache2)
        dx1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx2, cache1)

        grads['W3'] += self.reg*W3
        grads['W2'] += self.reg*W2
        grads['W1'] += self.reg*W1
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
