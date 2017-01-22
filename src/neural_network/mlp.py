import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


class MLPClassifier:
    def __init__(self, n_in, hidden_units, n_out):
        # set random seed to fix the weights and biases		
        tf.set_random_seed(21)

        # neural network parameters
        self.n_in = n_in
        self.hid = hidden_units
        self.n_out = n_out
        self.keep_prob = tf.placeholder(tf.float32)

        # tf graph inputs
        self.X = tf.placeholder(tf.float32, [None, n_in])
        self.y = tf.placeholder(tf.int64)

        # define prediction, cost function and optimizer
        self.pred = self.mlp(self.X, self.keep_prob)
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.pred, self.y))
        self.learning_rate = tf.placeholder(tf.float32)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

        # evaluate model
        self.is_correct = tf.equal(tf.argmax(self.pred, 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))

        # initialization
        self.init_op = tf.global_variables_initializer()

        # save and restore all the variables
        self.saver = tf.train.Saver()
        self.model_path = os.path.join(os.path.dirname(os.getcwd()), 'neural_network', 'save_model', 'mlp.ckpt')


    def mlp(self, X, dropout):
        # [n_samples, n_feature] dot [n_feature, n_hidden[0]] -> [n_samples, n_hidden[0]]
        new_layer = self.get_equ(X, self.get_W(self.n_in, self.hid[0]), self.get_b(self.hid[0]))
        new_layer = tf.nn.dropout(tf.nn.relu(new_layer), dropout)

        """
        if there are three hidden layers: [1, 2, 3], we need two iterations:
        [n_samples, 1] dot [1, 2] -> [n_samples, 2]
        [n_samples, 2] dot [2, 3] -> [n_samples, 3]
        finally we have [n_samples, n_hidden[-1]]
        """
        if len(self.hid) != 1:
            for idx in range(len(self.hid) - 1):
                new_layer = self.get_equ(new_layer, self.get_W(self.hid[idx], self.hid[idx+1]),
                                         self.get_b(self.hid[idx+1]))
                new_layer = tf.nn.dropout(tf.nn.relu(new_layer), dropout)

        # [n_samples, n_hidden[-1]] dot [n_hidden[-1], n_out] -> [n_samples, n_out]
        out_layer = self.get_equ(new_layer, self.get_W(self.hid[-1], self.n_out), self.get_b(self.n_out))
        return out_layer


    def get_W(self, row, col):
        return tf.Variable(tf.random_normal([row, col]))


    def get_b(self, length):
        return tf.Variable(tf.random_normal([length]))


    def get_equ(self, X, W, b):
        return tf.add(tf.matmul(X, W), b)


    def fit(self, X_train, y_train, n_epoch=1000, n_batch=1, dropout=1.0):
        with tf.Session() as sess:
            sess.run(self.init_op)
            step = 1
            X_train_batch_list = np.array_split(X_train, n_batch)
            y_train_batch_list = np.array_split(y_train, n_batch)
            for _ in range(n_epoch):
                for X_train_batch, y_train_batch in zip(X_train_batch_list, y_train_batch_list):
                    sess.run(self.train_step, feed_dict={self.X: X_train_batch, self.y: y_train_batch,
                                                         self.keep_prob: dropout, self.learning_rate: 1e-3})
                    acc = sess.run(self.accuracy, feed_dict={self.X: X_train_batch, self.y: y_train_batch,
                                                             self.keep_prob: 1.0})
                    print("Iteration {}, Training Accuracy = {:.3f}".format(step, acc))
                    step += 1
            save_path = self.saver.save(sess, self.model_path) # save model weights to disk
            print("Model saved in: {}" .format(save_path))


    def fit_plot(self, X_train, y_train, n_epoch=1000, n_batch=1, dropout=1.0):
        iplot_xs, iplot_ys = [], []
        plt.ion()

        with tf.Session() as sess:
            sess.run(self.init_op)
            step = 1
            X_train_batch_list = np.array_split(X_train, n_batch)
            y_train_batch_list = np.array_split(y_train, n_batch)
            for _ in range(n_epoch):
                for X_train_batch, y_train_batch in zip(X_train_batch_list, y_train_batch_list):
                    sess.run(self.train_step, feed_dict={self.X: X_train_batch, self.y: y_train_batch,
                                                         self.keep_prob: dropout, self.learning_rate: 1e-3})
                    acc = sess.run(self.accuracy, feed_dict={self.X: X_train_batch, self.y: y_train_batch,
                                                             self.keep_prob: 1.0})
                    
                    if step % 10 == 0:
                        # update plot
                        iplot_xs.append(step)
                        iplot_ys.append(acc)
                        plt.plot(iplot_xs, iplot_ys, color='cornflowerblue')
                        plt.draw()
                        plt.pause(0.1)

                    print("Iteration {}, Training Accuracy = {:.3f}".format(step, acc))
                    step += 1
            save_path = self.saver.save(sess, self.model_path) # save model weights to disk
            print("Model saved in: {}" .format(save_path))
        
        # hold the graph of final state
        plt.ioff()
        plt.show()


    def predict(self, X_test):
        with tf.Session() as sess:
            sess.run(self.init_op)
            self.saver.restore(sess, self.model_path) # restore model weights from disk
            y_test_pred = sess.run(self.pred, feed_dict={self.X: X_test, self.keep_prob: 1.0})
        return np.argmax(y_test_pred, axis=1)
