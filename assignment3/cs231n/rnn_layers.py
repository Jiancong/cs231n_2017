from __future__ import print_function, division
from builtins import range
import numpy as np

"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(np.dot(prev_h,Wh)+ np.dot(x, Wx) + b)
    cache = (next_h, prev_h, x, Wx, Wh, b)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache

def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    (next_h, prev_h, x, Wx, Wh, b) = cache

    dWx = np.dot(x.T, (1 - next_h**2) * dnext_h) # (D,N) * (N,H) =(D,H)
    dWh = np.dot(prev_h.T, (1-next_h**2) * dnext_h) # (H, N) * (N,H)= (H,H)

    dx = np.dot((1-next_h**2) * dnext_h, Wx.T) #  (N,H)* (H,D) = (N,D)

    dprev_h = np.dot((1-next_h**2)* dnext_h, Wh.T)  # (N,H)* (H,H) = (N,H)

    db = np.sum((1-next_h**2)*dnext_h, axis=0)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    (N, T, D) = x.shape
    (N, H) = h0.shape
    #rnn_step_forward(x, prev_h, Wx, Wh, b)
    cache = []
    h = np.zeros((N, T, H))
    hnew = np.transpose(h, (1,0,2)) # (T, N, H)

    h_prev = h0
    for t in range(T):
        #cache = (next_h, prev_h, x, Wx, Wh, b)
        next_h, c = rnn_step_forward(x[:,t,:], h_prev, Wx, Wh, b)
        h_prev = next_h
        hnew[t,:,:]= next_h
        cache.append(c)
        
    h = np.transpose(hnew, (1,0,2)) # (N, T, H)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)
    - NOTE: this dh[:, t, :] should be considered as the derivative dh to dy
    - cache: the cache got from rnn_forward

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    (N,T,H) = dh.shape
    dhtrans = np.transpose(dh, (1,0,2)) # (T, N, H)
    (next_h, prev_h, x, Wx, Wh, b) = cache[-1]
    (N,D) = x.shape

    dh0 = np.zeros((N,H)) 
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    dprev_h = np.zeros((N, H))
    db = np.zeros((H,))
    
    dx = np.zeros((T,N, D)) # (T, N, D)
    
    for i in reversed(range(T)):
        c = cache[i]
        
        dtotal_dh = dhtrans[i,:,:]  + dprev_h   #(N,H)
        #return dx, dprev_h, dWx, dWh, db
        #cache = (next_h, prev_h, x, Wx, Wh, b)
        dxi, dprev_h, dWxi, dWhi, dbi = rnn_step_backward(dtotal_dh, c)
        #print("dx:", dxi, ",dprev_hi:", dprev_hi, ",dWxi:",dWxi, ",dWhi:", dWhi, ",dbi:", dbi)
        dWx += dWxi
        dWh += dWhi
        db += dbi           
        dx[i,:,:] = dxi
        
    dx = np.transpose(dx, (1, 0, 2))
    dh0 = dprev_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    (N, T) = x.shape
    (V, D) = W.shape

    x_reshape = x.reshape((N*T))
    x_onehot = np.zeros((N*T, V))
    x_onehot[np.arange(N*T), x_reshape] = 1 # [N*T, V]

    out_tmp = np.dot(x_onehot, W) # [N*T, D]
    out = out_tmp.reshape((N, T, D))
    cache = (x, W , x_onehot) 
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    (x, W, x_onehot) = cache
    (N, T) = x.shape
    (V, D) = W.shape

    dout_reshape = dout.reshape(N*T, D)
    dW = np.dot(x_onehot.T, dout_reshape)

    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    (N,H)= prev_c.shape
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b.T  #(N, 4H)+ (N,4H) + (4H,)
    ai = a[:, :H]
    af = a[:, H:2*H]
    ao = a[:, 2*H:3*H]
    ag = a[:, 3*H:]
    i = sigmoid(ai) #(N, H)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c) 
    cache = [x, Wx, Wh, b, prev_c, prev_h, next_c, next_h, i, f, o, g]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    (x, Wx, Wh, b, prev_c, prev_h, next_c, next_h, i, f, o, g) = cache

    # many of this gradients contribute in 2 ways, once via next_c, once
    # via next_h. here we call those paths 1 and 2, just for clarity.
    alt_dnext_h = dnext_h*o*(1-np.power(np.tanh(next_c), 2)) 
    dS1 = dnext_c
    dS1 += alt_dnext_h
    dprev_c = f*dS1

    di = dS1*g
    dg = dS1*i

    do = dnext_h * np.tanh(next_c)
    df = dS1 * prev_c

    # backprop dsigmoid(x) = sigmoid(x)*(1-sigmoid(x))
    # backprop dtanh(x) = 1-np.power(tanh(x), 2)
    da_i = di * i * (1-i)
    da_g = dg * (1-np.power(g, 2))
    da_f = df * f * (1-f)
    da_o = do * o * (1-o)

    # remember a is just the concatenation of a_i, a_f, a_o, a_g
    da = np.hstack((da_i, da_f, da_o, da_g))
    #print("da.shape=>", da.shape)

    dWx = np.dot(x.T, da)  #(D,N) * (N,4H) = (D, 4H)
    dWh = np.dot(prev_h.T, da) #(H,N) * (N,4H) = (H, 4H)
    #print("da.T.shape=>", da.T.shape)
    db = np.sum(da.T, axis=1) #(4H,) = (N,4H).T
    dx = np.dot(da, Wx.T) #(N,D) = (N,4H) (D,4H)

    dprev_h = np.dot(da , Wh.T) #(N,4H) * (H,4H)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    (N, T, D) = x.shape
    (N, H) = h0.shape

    cache = []
    c0 = np.zeros((N, H))
    h = np.zeros((N, T, H))
    htranspose = np.transpose(h, (1, 0, 2)) #(T, N, H)

    prev_h = h0 
    prev_c = c0
    for t in range(T):
        next_h, next_c, c = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c 
        htranspose[t,:, :] = next_h
        cache.append(c)

    h = np.transpose(htranspose, (1,0,2)) # (N, T, H)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    (N, T, H) = dh.shape
    dhtrans = np.transpose(dh, (1, 0,2)) # (T, N, H)
    (x, Wx, Wh, b, prev_c, prev_h, next_c, next_h, i, f, o, g) = cache[-1]
    (N,D) =x.shape 

    dh0 = np.zeros((N,H))
    dWx = np.zeros((D,4*H))
    dWh = np.zeros((H,4*H))
    db = np.zeros((4*H,))
    dprev_c = np.zeros((N, H))
    dprev_h = np.zeros((N, H))

    dx = np.zeros((T, N , D ))

    dnext_c = dprev_c.copy()
    
    for i in reversed(range(T)):
        c = cache[i]
        dnext_h = (dhtrans[i,:,:]  + dprev_h).copy()
        dxi, dprev_h, dprev_c, dWxi, dWhi, dbi = lstm_step_backward(dnext_h, dnext_c, c)
        dx[i,:,:] = dxi
        dnext_c = dprev_c
        
        dWx += dWxi
        dWh += dWhi
        db += dbi
    pass
    dx = np.transpose(dx, (1, 0, 2))
    dh0 = dprev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
