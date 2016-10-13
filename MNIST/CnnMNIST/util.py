from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def inference(images, keep_prob):
    x_image = tf.reshape(images, [-1, 28, 28, 1])

    # First Convolutional Layer
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = conv2d(x_image, W_conv1) + b_conv1
        h_relu1 = tf.nn.relu(h_conv1)
        h_pool1 = max_pool_2x2(h_relu1)

    # Second Convolutional Layer
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
        h_relu2 = tf.nn.relu(h_conv2)
        h_pool2 = max_pool_2x2(h_relu2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely Connected Layer
    with tf.name_scope('dconn'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='dropout')

    # Readout Layer
    with tf.name_scope('softmax_linear'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_logits

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels)
    loss = tf.reduce_mean(cross_entropy, name='entropy_mean')
    return loss

def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss
    tf.scalar_summary(loss.op.name, loss)
    # Create AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step
    global_step = tf.Variable(0, trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    labels  = tf.to_int64(labels)
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def fill_feed_dict(data_set, img_pl, label_pl, keep_prob,
                   prob_val=0.5, batch_size=50):
    img_feed, label_feed = data_set.next_batch(batch_size)
    feed_dict = {
        img_pl: img_feed,
        label_pl: label_feed,
        keep_prob: prob_val
    }

    return feed_dict

def do_eval(sess, eval_op, data_set, img_pl, label_pl, keep_prob,
            prob_val=1.0, batch_size=50):
    true_count = 0
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, img_pl, label_pl,
                                   keep_prob, prob_val, batch_size)
        true_count += sess.run(eval_op, feed_dict=feed_dict)
        precision = true_count / num_examples
    print("[RES] Num examples: {:5d} Num correct: {:5d} Precision is: {:.3f}\
          ".format(num_examples, true_count, true_count / num_examples))
