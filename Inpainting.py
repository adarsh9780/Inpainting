
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from skimage.data import astronaut
from scipy.misc import imresize
import os
get_ipython().magic('matplotlib inline')

img = plt.imread('a.jpg')
img = imresize(img, (256, 256))
print(img.shape)
plt.imshow(img)


# In[2]:


#Store coordinates of each pixel in
position = []

#And corresponding color intensity in
color = []


for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        position.append([row, col])
        color.append(img[row][col])
    
#convert list object in to array
position = np.array(position)
color = np.array(color)

#print the shapes
position.shape, color.shape


# In[3]:


normalized_position = (position-np.mean(position))/np.std(position)


# In[4]:


#print the shapes again
normalized_position.shape, color.shape


# In[5]:


plt.imshow(color.reshape(img.shape))


# In[6]:


def linear(X, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())
        h = tf.matmul(X, W) + b
        if activation is not None:
            h = activation(h)
        return h


# In[7]:


X = tf.placeholder(tf.float32, [None, 2], name='X')
Y = tf.placeholder(tf.float32, [None, 3], name='Y')
n_neurons = [2, 64, 64, 64, 64, 64, 64, 3]

current_input = X
for layer_i in range(1, len(n_neurons)):
    with tf.variable_scope('layer_' + str(layer_i)) as scope:
        current_input = linear(
            X=current_input,
            n_input=n_neurons[layer_i - 1],
            n_output=n_neurons[layer_i],
            activation=tf.nn.relu if (layer_i+1) < len(n_neurons) else None,)
        scope.reuse_variables()
Y_pred = current_input


# In[8]:


def distance(p1, p2):
    return tf.abs(p1 - p2)


# In[9]:


# cost = tf.reduce_mean(
#     tf.reduce_sum(tf.abs(tf.subtract(Y_pred, Y)), 1))

cost = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(Y_pred, Y), 2)))

# cost = tf.reduce_mean(
#     tf.reduce_sum(distance(Y_pred, Y), 1))


# In[10]:


optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# In[11]:


n_iterations = 500
batch_size = 50
with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    # This will set W and b to their initial random normal value.
    sess.run(tf.global_variables_initializer())

    # We now run a loop over epochs
    prev_training_cost = 0.0
    for it_i in range(n_iterations):
        idxs = np.random.permutation(range(len(position)))
        n_batches = len(idxs) // batch_size
        for batch_i in range(n_batches):
            idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
            sess.run(optimizer, feed_dict={X: position[idxs_i], Y: color[idxs_i]})

        training_cost = sess.run(cost, feed_dict={X: position, Y: color})
        print(it_i, training_cost)

        if (it_i + 1) % 20 == 0:
            ys_pred = Y_pred.eval(feed_dict={X: position}, session=sess)
            fig, ax = plt.subplots(1, 1)
            img = np.clip(ys_pred.reshape(img.shape), 0, 255).astype(np.uint8)
            plt.imshow(img)
            plt.show()


# In[ ]:




