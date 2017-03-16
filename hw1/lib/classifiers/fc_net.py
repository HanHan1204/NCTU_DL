import numpy as np

from lib.layers import *
from lib.layer_utils import *


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - relu} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=1*28*28, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    #  Initialize the parameters of the network, storing all values in
    dims = [input_dim] + hidden_dims + [num_classes]
    for i in xrange(self.num_layers):
      self.params['b%d' % (i+1)] = np.zeros(dims[i + 1])
      self.params['W%d' % (i+1)] = np.random.randn(dims[i], dims[i + 1]) * weight_scale

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    layer = {}
    layer[0] = X
    cache_layer = {}

    for i in xrange(1, self.num_layers):
      layer[i], cache_layer[i] = affine_relu_forward(layer[i - 1],
                                                     self.params['W%d' % i],
                                                     self.params['b%d' % i])
    # Forward into last layer
    WLast = 'W%d' % self.num_layers
    bLast = 'b%d' % self.num_layers
    scores, cache_scores = affine_forward(layer[self.num_layers - 1],
                                          self.params[WLast],
                                          self.params[bLast])

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    loss, dscores = softmax_loss(scores, y)

    # add regularization loss:
    for i in xrange(1, self.num_layers + 1):
      loss += 0.5 * self.reg * np.sum(self.params['W%d' % i]**2)

    # Backprop into last layer
    dx = {}
    dx[self.num_layers], grads[WLast], grads[bLast] = affine_backward(dscores, cache_scores)
    grads[WLast] += self.reg * self.params[WLast]

    # Backprop into remaining layers
    for i in reversed(xrange(1, self.num_layers)):
      # r = cache_layer[i + 1]
      dx[i], grads['W%d' % i], grads['b%d' % i] = affine_relu_backward(dx[i + 1], cache_layer[i])
      grads['W%d' % i] += self.reg * self.params['W%d' % i]

    return loss, grads
