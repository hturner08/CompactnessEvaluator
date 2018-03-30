import tensorflow as tf
import numpy as np
import random
# class District:
#     def __init__(self, perimeter, area):
#         self.perimeter =
#constructing data

#Gathering Data
district_data = json.load(open('stats.json'))
district_dataset= []
for district in data
    stats = []
    stats.append(district["State"])
    stats.append(district["NAMELSAD"])
    stats.append(district["Polsby-Popper"]/100)
    stats.append(district["Schwartzberg"]/100)
    stats.append(district["Area/Convex Hull"]/100)
    stats.append(district["Reock"]/100)
    district_dataset.append(stats)

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
