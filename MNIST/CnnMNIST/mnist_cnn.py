from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import util

FLAGS = None

def run_training():
    data_sets = input_data.read_data_sets(FLAGS.data_dir)

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 784))
        y_ = tf.placeholder(tf.float32, shape=(FLAGS.batch_size))
        keep_prob = tf.placeholder(tf.float32)
        y_logits = util.inference(x, keep_prob)
        loss_op = util.loss(y_logits, y_)
        train_op = util.training(loss_op, FLAGS.learning_rate)
        eval_op = util.evaluation(y_logits, y_)

        # Build summary Tensor based on the TF collection of Summaries
        summary = tf.merge_all_summaries()
        # Create a saver for writing training checkpoints
        saver = tf.train.Saver()

        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction = 0.6)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter(FLAGS.data_dir, sess.graph)
            for step in range(FLAGS.max_steps):
                feed_dict = util.fill_feed_dict(data_sets.train, x, y_,
                                                keep_prob, prob_val=0.5,
                                                batch_size=FLAGS.batch_size)
                sess.run(train_op, feed_dict=feed_dict)
                # _, loss_value = sess.run([train_op, loss_op],
                #                          feed_dict=feed_dict)
                if step % 100 == 0:
                    # print("[INFO] Step {:4d}: loss = {:.3f}\
                    #       ".format(step, loss_value))
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    summary_str = sess.run(summary, feed_dict=feed_dict,
                                           options=run_options,
                                           run_metadata=run_metadata)
                    summary_writer.add_run_metadata(run_metadata,
                                                    "step%d" % step)
                    # summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    # summary_writer.flush()

                if (step+1) % 1000 == 0:
                    checkpoint_file = os.path.join(FLAGS.data_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=step)
                    print("[INFO] Accuracy, step: {:5d}".format(step+1))
                    # Accuracy on training set
                    print("[RES] Training Accuracy:")
                    util.do_eval(sess, eval_op, data_sets.train, x, y_,
                            keep_prob, 1.0, batch_size=FLAGS.batch_size)
                    # Accuracy on validation set
                    print("[ES] Validation Accuracy:")
                    util.do_eval(sess, eval_op, data_sets.validation, x, y_,
                            keep_prob, 1.0, batch_size=FLAGS.batch_size)
                    # Accuracy on test set
                    print("[RES] Test Accuracy:")
                    util.do_eval(sess, eval_op, data_sets.test, x, y_,
                            keep_prob, 1.0, batch_size=FLAGS.batch_size)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', type=str, default='/tmp/data',
                       help='Directory for storing data')
    parse.add_argument('--batch_size', type=int, default=50,
                       help='Number of samples for each mini batch')
    parse.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for training')
    parse.add_argument('--max_steps', type=int, default=20000,
                       help='Number of steps to run trainer')
    FLAGS = parse.parse_args()

    run_training()
