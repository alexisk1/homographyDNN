"""
Coding tests
"""

import tensorflow as tf


OUTPUT_SIZE = 1
    
#Hyper parameters
EPOCHS = 3000
BATCH_SIZE = 64
rate = 0.0001
beta = 0.001

def flatten(layer):
    shape = layer.get_shape()
    num_elements_ = shape[1:4].num_elements()
    flattened_layer = tf.reshape(layer, [-1, num_elements_])
    return flattened_layer, num_elements_

def conv2d_custom(input, filter_size, num_of_channels, num_of_filters,name, activation=tf.nn.relu, dropout=None,
                  padding='SAME', max_pool=True, strides=(1, 1)):  
    shape = [filter_size, filter_size, num_of_channels, num_of_filters]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05),name=name+"w")
    bias =  tf.Variable(tf.constant(0.01, shape=[num_of_filters]),name=name+"b")

    
    layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides[0], strides[1], 1], padding=padding) + bias
    
    if activation != None:
        layer = activation(layer)
    if max_pool:
        layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
    return layer

def dense_custom(input, input_size, output_size, name, activation=tf.nn.relu, dropout=None):
    weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.05),name=name+"w")
    bias =  tf.Variable(tf.constant(0.01 , shape=[output_size]),name=name+"b")
    
    layer = tf.matmul(input, weights) + bias
    
    if activation != None:
        layer = activation(layer)
    
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
        
    return layer

def homography_net(inp):
    conv1 = conv2d_custom(inp, 3, 2, 32,"conv1", max_pool=False)
    conv2 = conv2d_custom(conv1, 3, 32, 32,"conv2", max_pool=False)
    max1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3 = conv2d_custom(max1, 3, 32, 32, "conv3", max_pool=False)
    conv4 = conv2d_custom(conv3, 3, 32, 32, "conv4", max_pool=False)
    max2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv5 = conv2d_custom(max2, 3, 32, 64, "conv5", max_pool=False)
    conv6 = conv2d_custom(conv5, 3, 64, 64, "conv6", max_pool=False)
    max3 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv7 = conv2d_custom(max3, 3, 64, 128, "conv7", max_pool=False)
    conv8 = conv2d_custom(conv7, 3, 128, 128, "conv8", max_pool=False)

    flat, features = flatten(conv8)
    
    output = dense_custom(flat, features, OUTPUT_SIZE,"dense", activation=None)
    
    return output



IMG_WIDTH = 128
IMG_HEIGHT= 128
#Input/ prediction / cost function / optimizer
X = tf.placeholder(tf.float32, (None, IMG_HEIGHT, IMG_WIDTH , 2), name="input")
y = tf.placeholder(tf.float32, (None), name="output")
regression_node = homography_net(X)

loss_op =tf.losses.mean_squared_error(y, regression_node)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_op = optimizer.minimize(loss_op)


