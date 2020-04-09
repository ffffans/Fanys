import tensorflow as tf
from layers import *
import numpy as np

import cv2
import os
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def creat_conv_net(x, img_width, img_height, img_channel, drop_out, n_class=2):
    input = tf.reshape(x, [-1, img_width, img_height, img_channel])
    weights = []
    bias = []

    variables = []
    # layer1
    w1_1 = weight_variable([3, 3, img_channel, 32])
    b1_1 = bias_variable([32])
    conv1_1 = conv2d(input, w1_1, b1_1, drop_out)
    conv1_1 = tf.nn.relu(conv1_1)

    w1_2 = weight_variable([3, 3, 32, 32])
    b1_2 = bias_variable([32])
    conv1_2 = conv2d(conv1_1, w1_2, b1_2, drop_out)
    conv1_2 = tf.nn.relu(conv1_2)

    pool1 = max_pool(conv1_2, 2)
    weights.append((w1_1, w1_2))
    bias.append((b1_1, b1_2))

    # layer2
    w2_1 = weight_variable([3, 3, 32, 64])
    b2_1 = bias_variable([64])
    conv2_1 = conv2d(conv1_2, w2_1, b2_1, drop_out)
    conv2_1 = tf.nn.relu(conv2_1)

    w2_2 = weight_variable([3, 3, 64, 64])
    b2_2 = bias_variable([64])
    conv2_2 = conv2d(conv2_1, w2_2, b2_2, drop_out)
    conv2_2 = tf.nn.relu(conv2_2)
    weights.append((w2_1, w2_2))
    bias.append((b2_1, b2_2))

    # layer3 deconv
    w3 = weight_variable([3, 3, 32, 64])
    b3 = bias_variable([32])
    deconv1 = deconv2d(conv2_2, w3) + b3
    deconv1 = tf.nn.relu(deconv1)
    deconv_concat1 = crop_and_concat(deconv1, conv1_2)

    # layer4
    w4_1 = weight_variable([3, 3, 64, 32])
    b4_1 = bias_variable([32])
    conv4_1 = conv2d(deconv_concat1, w4_1, b4_1, drop_out)
    conv4_1 = tf.nn.relu(conv4_1)

    w4_2 = weight_variable([3, 3, 32, 32])
    b4_2 = bias_variable([32])
    conv4_2 = conv2d(conv4_1, w4_2, b4_2, drop_out)
    conv4_2 = tf.nn.relu(conv4_2)
    weights.append((w4_1, w4_2))
    bias.append((b4_1, b4_2))

    # layer5 output
    w5 = weight_variable([1, 1, 32, n_class])
    b5 = bias_variable([n_class])
    output_map = tf.nn.sigmoid(conv2d(conv4_2, w5, b5, drop_out))

    return output_map, variables


class UnetModel(object):
    def __init__(self, img_width, img_height, img_channel, n_class=2, cost="cross_entropy", **kwargs):
        tf.reset_default_graph()

        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.n_class = n_class

        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[None, None, None, self.img_channel], name="train_x")
        self.y = tf.placeholder("float", shape=[None, None, None, self.n_class], name="ground_truth")
        self.drop_out = tf.placeholder("float", name="drop_out")
        # self.lr = tf.placeholder("float", name="learning_rate")

        self.y_pred, self.variables = creat_conv_net(self.x, img_width, img_height, img_channel, self.drop_out)
        self.cost = self.get_cost(cost)
        self.gradients_node = tf.gradients(self.cost, self.variables)

        with tf.name_scope("cross_entropy"):
            self.cross_entropy = cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                               tf.reshape(pixel_wise_softmax(self.y_pred), [-1, n_class]))

        with tf.name_scope("results"):
            self.predicter = pixel_wise_softmax(self.y_pred)
            self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
            self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def get_cost(self, cost_name):
        H, W, C = self.y_pred.get_shape().as_list()[1:]
        with tf.name_scope("cost"):
            flat_logits = tf.reshape(self.y_pred, [-1, H * W * C])
            flat_labels = tf.reshape(self.y, [-1, H * W * C])
            # flat_logits = tf.reshape(self.y_pred, [-1, self.img_channel])
            # flat_labels = tf.reshape(self.y, [-1, self.n_class])
            if cost_name == "cross_entropy":
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits,
                                                                              labels=flat_labels))
            elif cost_name == "dice_coefficient":
                eps = 1e-5
                prediction = pixel_wise_softmax(self.y_pred)
                intersection = tf.reduce_sum(prediction * self.y)
                union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
                loss = -(2 * intersection / (union))

            else:
                raise ValueError("Unknown cost function: {:}".format(cost_name))

            return loss

    def predict(self, test_imgs, model_path):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_path)
            test_imgs = np.reshape(test_imgs, (1, test_imgs.shape[0], test_imgs.shape[1], self.img_channel))
            pred = sess.run(self.y_pred, feed_dict={self.x: test_imgs,
                                                    self.drop_out: 1})
            # result = np.reshape(pred, (test_imgs[1], test_imgs[2]))
            # result = result.astype(np.float32) * 255.
            # result = np.clip(pred, 0, 255).astype("uint8")

        return pred

    def restore(self, sess, model_path):
        saver = tf.train.Saver()
        saver.retore(sess, model_path)
        logging.info("model saved in file: %s" % model_path)

    def save(self, sess, model_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path


class Trainer(object):
    def __init__(self, net, batch_size=10, vertification_batch_size=4, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.vertification_batch_size = vertification_batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self, iteration, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.01)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=iteration,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost,
                                                                               global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                           global_step=global_step)

        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name="global_step")

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]), name="norm_gradients")

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.acc)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, iteration=10, epochs=100, drop_out=0.75, display_step=1,
              prediction_path="prediction", retore=False, write_graph=False):
        save_path = os.path.join(output_path, "model.ckpt")
        if epochs == 0:
            return save_path

        init = self._initialize(iteration, output_path, retore, prediction_path)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if retore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.retore(sess, ckpt.model_checkpoint_path)

            test_x, test_y = data_provider(self.vertification_batch_size)
            self.output_train_state(sess, test_x, test_y, "_init")

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")

            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch * iteration), ((epoch + 1) * iteration)):
                    batch_x, batch_y = data_provider(self.batch_size)

                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                        feed_dict={self.net.x: batch_x,
                                   self.net.y: batch_y,
                                   self.net.drop_out: drop_out})

                    if step % display_step == 0:
                        self.output_minibatch_state(sess, summary_writer, step, batch_x, batch_y)

                    total_loss += loss

                self.output_epoch_state(epoch, total_loss, iteration, lr)
                self.output_train_state(sess, test_x, test_y, "epoch_%s" % epoch)
                save_path = self.net.save(sess, save_path)

            logging.info("Optimization Finish!")
            return save_path
        
    def output_train_state(self, sess, batch_x, batch_y, name):
        prediction, loss = sess.run((self.net.predicter, self.net.cost), feed_dict={self.net.x: batch_x,
                                                                                    self.net.y: batch_y,
                                                                                    self.net.drop_out: 1.})
        logging.info("Vertification error = {:.1f}%, loss = {:.4f}".format(self.error_rate(prediction, batch_y), loss))
        
    def output_epoch_state(self, epoch, total_loss, iteration, lr):
        logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / iteration), lr))
        
    def output_minibatch_state(self, sess, summary_writer, step, batch_x, batch_y):
        summary_str, loss, acc, prediction = sess.run((self.summary_op, self.net.cost, self.net.acc, self.net.predicter),
                                                       feed_dict={self.net.x: batch_x,
                                                                  self.net.y: batch_y,
                                                                  self.net.drop_out: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Miinibatch loss = {:.4f}, Training accuary = {:.4f}, Minibatch error = {:.1f}%".format(
            step, loss, acc, self.error_rate(prediction, batch_y)
        ))
        
    def error_rate(self, prediction, batch_y):
        return 100.0 - ( 100.0 * np.sum(np.argmax(prediction, 3) == np.argmax(batch_y, 3)) /
                (prediction.shape[0] * prediction.shape[1] * prediction.shape[2]))

# if __name__:
#     x = cv2.imread("TCGA_CS_4941_19960909_1.tif")
#     x = tf.cast(x, tf.float32)
#
#     creat_conv_net(x, 64, 64, 1, 0.8)
