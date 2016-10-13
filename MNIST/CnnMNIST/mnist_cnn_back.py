# @Author: chenpingjun1990
# @Date:   2016-10-11T22:45:16-04:00
# @Last modified by:   chenpingjun1990
# @Last modified time: 2016-10-12T10:03:40-04:00

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def main():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = conv2d(x_image, W_conv1) + b_conv1
    h_relu1 = tf.nn.relu(h_conv1)
    h_pool1 = max_pool_2x2(h_relu1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_relu2 = tf.nn.relu(h_conv2)
    h_pool2 = max_pool_2x2(h_relu2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Loss
    cross_entropy_vec = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
    cross_entropy = tf.reduce_mean(cross_entropy_vec)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(cross_entropy)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Start training
    BATCH_SIZE = 50
    NUM_ITER = 20000
    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction = 0.6)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(NUM_ITER):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            if epoch % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_xs, y_: batch_ys, keep_prob: 1.0
                    })
                print("[INFO] Step: {:5d}, training accuracy {:.3f}".format(
                    epoch, train_accuracy
                ))
            train_op.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

        result = sess.run(fetches=[accuracy], feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
            })
    print("[RES] Test accuracy {:.3f}".format(result[0]))

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', type=str, default='/tmp/data',
                       help='Directory for storing data')
    FLAGS = parse.parse_args()

    main()
