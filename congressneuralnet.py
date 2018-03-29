import tensorflow as tf
import numpy as np
import random
#constructing data
def Schwartzberg(district):
    return score
def Polsby Popper(district):
    return score
def Reock(district):
    return score
def Convex Hull(district):
    return score
def Harris(district):
    return score


hyper_param_one = .5
X = tf.placeholder(dtype = tf.float32, shape = [None, 5])
W = tf.get_variable(name = 'W', dtype = tf.float32, initializer=tf.ones([5, 1]))
b = tf.get_variable(name = 'b', dtype = tf.float32, initializer= tf.zeros([1]))
unnormalized_prob = tf.add(tf.matmul(X, W), b)
y = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = unnormalized_prob)
train_step = tf.train.GradientDescentPptimizer(learning_rate = hyper_param_one).minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initiliazier().run()
sess.run(train_step, dataset)
