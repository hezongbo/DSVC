import numpy as np
import random
import math

class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.ww =None
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        pass
    #先计算出样本数是多少
        m = X_batch.shape[0]
        #计算出theta*x,就是sigmod函数的z
        z = X_batch.dot(self.w)
        #下面就是计算代价函数了
        J=(-1.0) * np.sum( y_batch.dot(np.log(self.sigmoid(z)))+(1-y_batch).dot(np.log(1-self.sigmoid(z))))/m
        grand=X_batch.T.dot(self.sigmoid(z)-y_batch)/m
        #########################################################################
        return J,grand
    
    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape[0],X.shape[1]

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            pass
            #########################################################################
            index = np.random.choice(num_train, size=batch_size, replace=verbose) 
            X_batch = X[index,:]
            y_batch = y[index]
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            pass
            #########################################################################
            self.w=self.w-learning_rate * grad
            #########################################################################
    
            if verbose and it % 20 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        print(len(loss_history))
        return loss_history
    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        pass
        ###########################################################################
        z = X.dot(self.w)
        y_prevalue = self.sigmoid(z)
        for i in range(len(y_pred)):
            if y_prevalue[i] > 0.5:
                y_pred[i]= 1       
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose = True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        """
        #得出数据的行数和列数
        num_train,dim=X.shape[0],X.shape[1]
        #这块中的ww之所以跟上面定义w不一样,是因为，在这里要得出所有的从0到9的每一个数子的函数，也就是Z
        self.ww=np.zeros((dim,10))
        for i in range(10):
            y_train=[]
            for lable in y:
                if lable==i:
                    y_train.append(1)
                else:
                    y_train.append(0)
            y_train=np.array(y_train)
            self.w=None
            print('当前预测是数字是:',i)
            self.train(X,y_train,learning_rate, num_iters ,batch_size)
            self.ww[:,i]=self.w
        print(self.ww)
        
    #这个地方的预测我是这样理解的，跟上面的二分类不一样的是这是基于多分类的，    
    def one_vs_all_predict(self, X):        
        lables=self.sigmoid(X.dot(self.ww))#这个地方就是用不同数字对应的函数去计算
        mul_y_pred = np.zeros(X.shape[0])
        for i in range(len(mul_y_pred)):           
            mul_y_pred[i]=np.argmax(lables[i,:])#选出一个概率最大的出来，例如一组数据（x1,x2,x3,x4,...x10）会与数字0到9的函数计算得出来那哪个的概率最大，哪个概率最大，就能预测他应该是哪一个数字               
        return mul_y_pred