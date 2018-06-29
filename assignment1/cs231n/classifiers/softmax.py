import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # (D, C)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Please refer *CAREFULLY* here to know the loss and gradient
  # https://deepnotes.io/softmax-crossentropy
  # when implementing, refer class note too. 
  # the website is http://cs231n.github.io/linear-classify/#softmax
  num_classes = W.shape[1] # C : 10
  num_train = X.shape[0] # N:500

  linear_results = np.dot(X, W) #(N, C) = (N, D) * (D, C)

  for i in range(num_train):
      f_i = linear_results[i, :] # (1, C)
      f_i -= np.max(f_i) 
      # (, C)
      p = np.exp(f_i) / np.sum(np.exp(f_i)) # safe to do, gives the correct answer

      log_likelihood_i = -np.log(p[y[i]])
      loss += log_likelihood_i

      # Compute gradient
      # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
      # Here we are computing the contribution to the inner sum for a given i.
      # refer here to find the details https://stackoverflow.com/questions/41663874/cs231n-how-to-calculate-gradient-for-softmax-loss-function
      for j in range(num_classes):
          softmax_score = p[j]
          # Gradient calculation.
          if j == y[i]:
            dW[:, j] += (-1 + softmax_score) * X[i]
          else:
            dW[:, j] += softmax_score * X[i]

  # Compute average
  loss /= num_train
  dW /= num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W) #(N, C) = (N, D) * (D, C)
  shift_scores = scores - np.max(scores, axis=1)[: , np.newaxis]
  softmax_scores = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1)[:, np.newaxis] # (N,C)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  # (num_train, num_classes): N, C

  loss = np.sum(-np.log(softmax_scores[range(num_train),y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dScore = softmax_scores  # (N, C)
  dScore[range(num_train),y] = dScore[range(num_train),y] - 1 # This is a magic 

  dW = np.dot(X.T, dScore)
  dW /= num_train
  dW += 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

