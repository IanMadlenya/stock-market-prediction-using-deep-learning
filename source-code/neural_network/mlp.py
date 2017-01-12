import tensorflow as tf
import os
import numpy as np


class MLPClassifier:
    def __init__(self, n_in, hidden_units, n_out, learning_rate=1e-3):
        # set random seed to fix the weights and biases		
        tf.set_random_seed(21)

        # neural network parameters
        self.n_in = n_in
        self.hid = hidden_units
        self.n_out = n_out
        self.keep_prob = tf.placeholder(tf.float32)

        # tf graph inputs
        self.X = tf.placeholder(tf.float32, [None, n_in])
        self.y = tf.placeholder(tf.float32, [None, n_out])

        # define prediction, cost function and optimizer
        self.pred = self.mlp(self.X, self.keep_prob)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        # evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # initialization
        self.init = tf.global_variables_initializer()

        # save and restore all the variables
        self.saver = tf.train.Saver()
        self.model_path = os.path.join(os.path.dirname(os.getcwd()), 'neural_network', 'save_model', 'mlp.ckpt')


    def mlp(self, X, dropout):
        new_layer = self.get_equ(X, self.get_W(self.n_in, self.hid[0]), self.get_b(self.hid[0]))
        new_layer = tf.nn.dropout(tf.nn.relu(new_layer), dropout)
        if len(self.hid) != 1:
            idx = 0
            while idx != len(self.hid) - 1:
                new_layer = self.get_equ(new_layer, self.get_W(self.hid[idx], self.hid[idx+1]), self.get_b(self.hid[idx+1]))
                new_layer = tf.nn.dropout(tf.nn.relu(new_layer), dropout)
                idx += 1
        out_layer = self.get_equ(new_layer, self.get_W(self.hid[-1], self.n_out), self.get_b(self.n_out))
        return out_layer


    def get_W(self, row, col):
        return tf.Variable(tf.random_normal([row, col]))


    def get_b(self, length):
        return tf.Variable(tf.random_normal([length]))


    def get_equ(self, X, W, b):
        return tf.add(tf.matmul(X, W), b)


    def fit(self, X_train, y_train, n_epoch=1, batch_size=32, train_dropout=1.0):
        with tf.Session() as sess:
            sess.run(self.init)
            step = 1
            n_batch = int(len(X_train) / batch_size)
            X_train_batch_list, y_train_batch_list = np.array_split(X_train, n_batch), np.array_split(y_train, n_batch)
            for _ in range(n_epoch):
                for X_train_batch, y_train_batch in zip(X_train_batch_list, y_train_batch_list):
                    sess.run(self.train, feed_dict={self.X: X_train_batch, self.y: y_train_batch, self.keep_prob: train_dropout})
                    acc = sess.run(self.accuracy, feed_dict={self.X: X_train_batch, self.y: y_train_batch, self.keep_prob: 1.0})
                    print("Iter %d, Training Accuracy = %.3f" % (step, acc))
                    step += 1
            save_path = self.saver.save(sess, self.model_path) # save model weights to disk
            print("Model saved in: %s" % save_path)


    def predict(self, X_test):
        with tf.Session() as sess:
            sess.run(self.init)
            self.saver.restore(sess, self.model_path) # restore model weights from disk
            y_test_pred = sess.run(self.pred, feed_dict={self.X: X_test, self.keep_prob: 1.0})
        return y_test_pred


    def get_accuracy(self, pred, target):
        pred = np.argmax(pred, axis=1)
        target = np.argmax(target, axis=1)
        return np.equal(pred, target).astype(float).mean()
