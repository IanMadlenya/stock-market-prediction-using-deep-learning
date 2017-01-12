import tensorflow as tf
import os
import numpy as np


class CNNClassifier:
    def __init__(self, n_in, n_out, learning_rate=1e-3):
        # set random seed to fix the weights and biases		
        tf.set_random_seed(21)

        # tf graph inputs
        self.X = tf.placeholder(tf.float32, [None, n_in])
        self.y = tf.placeholder(tf.float32, [None, n_out])
        self.keep_prob = tf.placeholder(tf.float32)

        # define weights and biases
        self.W = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 2 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, n_out]))
        }

        self.b = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([n_out]))
        }

        # define prediction, cost function and optimizer
        self.pred = self.conv_net(self.X, self.W, self.b, self.keep_prob)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        
        # evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # initialization
        self.init = tf.global_variables_initializer()

        # save and restore all the variables
        self.saver = tf.train.Saver()
        self.model_path = os.path.join(os.path.dirname(os.getcwd()), 'neural_network', 'save_model', 'cnn.ckpt')


    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


    # Create model
    def conv_net(self, x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 56, 56, 1])

        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=4)

        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out


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
