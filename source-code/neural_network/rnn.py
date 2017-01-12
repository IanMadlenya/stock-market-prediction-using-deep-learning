import tensorflow as tf
import os
import numpy as np


class RNNClassifier:
    def __init__(self, n_step, n_in, n_hidden_units, n_out, learning_rate=1e-3):
        # set random seed to fix the weights and biases		
        tf.set_random_seed(21)

        # neural network parameters
        self.n_step = n_step
        self.n_in = n_in
        self.n_hidden_units = n_hidden_units
        self.n_out = n_out
        self.keep_prob = tf.placeholder(tf.float32)

        # tf graph inputs
        self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_in])
        self.y = tf.placeholder(tf.float32, [None, self.n_out])

        # define weights and biases
        self.W = tf.Variable(tf.random_normal([self.n_hidden_units, self.n_out]))
        self.b = tf.Variable(tf.random_normal([self.n_out]))

        # define prediction, cost function and optimizer
        self.pred = self.rnn_lstm(self.X, self.W, self.b, self.keep_prob)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        # evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # initialization
        self.init = tf.global_variables_initializer()

        # save and restore all the variables
        self.saver = tf.train.Saver()
        self.model_path = os.path.join(os.path.dirname(os.getcwd()), 'neural_network', 'save_model', 'rnn.ckpt')


    def rnn_lstm(self, X, W, b, dropout):
        X = tf.transpose(X, [1, 0, 2])
        X = tf.reshape(X, [-1, self.n_in])
        X = tf.split(0, self.n_step, X)

        # define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden_units, state_is_tuple=True)
        # get lstm cell output
        outputs, states = tf.nn.rnn(cell=lstm_cell, inputs=X, dtype=tf.float32)
        # linear activation, using rnn inner loop last output
        return tf.add(tf.matmul(outputs[-1], tf.nn.dropout(W, dropout)), tf.nn.dropout(b, dropout))


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
