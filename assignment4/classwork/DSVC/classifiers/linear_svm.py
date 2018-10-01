import numpy as np
from random import shuffle
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)#在这里 W是含有 3073行，10列的矩阵，这句代码的意思就是用每一个训练样本去和0到 9 的权重相乘，就是 theta * X[i]
    correct_class_score = scores[y[i]]#这句代码的意思就是：y[i]是训练集中的labels 就是标志是第几类，我们通过上面的计算可以得出对应的当前的X乘以第几类的
    for j in range(num_classes):#y[i]对应的是训练集中的第几类的意思
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] = dW[:, j] + X[i]
        dW[:, y[i]] = dW[:, y[i]] - X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
 
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  scores = X.dot(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores_correct = scores[np.arange(num_train),y]
  scores_correct = np.reshape(scores_correct,(num_train,-1))
  margins = scores-scores_correct + 1
  margins = np.maximum(0,margins)
  margins[np.arange(num_train),y] = 0
  loss += np.sum(margins) / num_train
  loss += 0.5*reg*np.sum(W*W)
  #############################################################################
 

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  margins[margins>0]=1
  row_num = np.sum(margins,axis=1)
  margins[np.arange(num_train),y] = -row_num
  dW += np.dot(X.T,margins)/num_train+reg * W
  #############################################################################

  return loss, dW
