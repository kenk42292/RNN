import numpy as np
import operator, sys, pickle
from utils import *
from datetime import datetime

class RNN:

    def __init__(self, word_dim, hidden_dim):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        #Randomly initialize weight matrices
        init_range = np.sqrt(1./word_dim)
        self.Wxh = np.random.uniform(-init_range, init_range, 
                    (hidden_dim, word_dim))
        self.Whh = np.random.uniform(-init_range, init_range,
                    (hidden_dim, hidden_dim))
        self.Why = np.random.uniform(-init_range, init_range,
                    (word_dim, hidden_dim))

    def forward_propagate(self, input):
        T = len(input)

        outputs = np.zeros((T, self.word_dim))
        hidden_states = np.zeros((T+1, self.hidden_dim))

        for t in np.arange(T):
            hidden_states[t] = np.tanh(self.Wxh[:,input[t]] \
                                + self.Whh.dot(hidden_states[t-1]))
            outputs[t] = softmax(self.Why.dot(hidden_states[t]))
        return [outputs, hidden_states]


    def train(self, X_train, Y_train, nepoch, learning_rate, batch_size):
        epoch_to_loss = []
        for epoch in range(nepoch):
            if not epoch%1000:
                time = datetime.today().strftime("%Y-%m-%d %H%M%S")
                loss = self.calc_loss(X_train, Y_train)
                print "%s : epoch %d : loss %f" % (time, epoch, loss)
                if (len(epoch_to_loss) > 0) and (epoch_to_loss[-1][1]<loss):
                    learning_rate*=0.5
                    print "setting learning rate to: %f"%learning_rate
                epoch_to_loss.append((epoch, loss))
                
            self.sgd_step(X_train, Y_train, learning_rate, batch_size)
            

    def sgd_step(self, X_train, Y_train, learning_rate, batch_size):
        indeces = np.random.choice(np.arange(len(X_train)), size=batch_size, replace=False);
        dL_dWxh_total, dL_dWhh_total, dL_dWhy_total = 0, 0, 0
        for i in indeces:
            dL_dWxh, dL_dWhh, dL_dWhy = self.bptt(X_train[i], Y_train[i])
            dL_dWxh_total += dL_dWxh
            dL_dWhh_total += dL_dWhh
            dL_dWhy_total += dL_dWhy
        dL_dWxh, dL_dWhh, dL_dWhy \
            = dL_dWxh_total/batch_size, dL_dWhh_total/batch_size, dL_dWhy_total/batch_size
        self.Wxh -= learning_rate*dL_dWxh
        self.Whh -= learning_rate*dL_dWhh
        self.Why -= learning_rate*dL_dWhy

    def bptt(self, x, y):
        T = len(x)
        o, h = self.forward_propagate(x)
        dL_dWxh, dL_dWhh, dL_dWhy = 0, 0, 0
        delta_sum = np.zeros_like(h[0])
        error = o
        error[np.arange(len(y)), y] -= 1
        for t in range(T)[::-1]:
            dL_dWhy += np.outer(error[t], h[t])
            delta_t = error[t].dot(self.Why)*(1-h[t]**2).T
            #delta_sum is the previous delta_sum updated, plus the new delta_t
            delta_sum = delta_sum.dot(self.Whh)*(1-h[t]**2).T
            delta_sum += delta_t
            dL_dWhh += delta_sum*h[t-1]
            dL_dWhy += delta_sum*x[t]
        return [dL_dWxh, dL_dWhh, dL_dWhy]


    def calc_loss(self, X_train, Y_train):
        L = 0
        total_words = 0
        for i in range(len(X_train)):
            outputs = self.forward_propagate(X_train[i])[0]
            correct_word_probs = outputs[np.arange(len(X_train[i])), Y_train[i]] 
            L -= np.sum(np.log(correct_word_probs))
            total_words += len(Y_train[i])
        return L/total_words






