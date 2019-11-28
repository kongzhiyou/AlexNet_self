import tensorflow as tf
import numpy as np
from cfg import config


def fcLayer(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[num_in, num_out], trainable=True)
        b = tf.get_variable('b', [num_out], trainable=True)
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
    return tf.nn.relu(out, name=scope.name)

def convLayer(x, k_h, k_w, stride_x, stride_y, feature_num, name, padding='SAME', groups=1):
    channel = x.shape[2]
    conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, stride_x, stride_y, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[k_h, k_w, channel / groups, feature_num])  # 从外面获取
        b = tf.get_variable('b', shape=[feature_num])

        # tf.split: 切分一个张量，num_or_size_splits切分后的个数，
        # axis:在第几维上切分
        x_new = tf.split(value=x, num_or_size_splits=groups, axis=3)
        w_new = tf.split(value=w, num_or_size_splits=groups, axis=3)
        # a = [1,2,3] b=[4,5,6]  zip(a,b) = [(1,4),(2,5),(3,6)]
        feature_map = [conv(t1, t2) for t1, t2 in zip(x_new, w_new)]  # 计算卷积，两个featureMap
        merge_featureMap = tf.concat(axis=3, values=feature_map)  # 合并两个featureMap
        out = tf.nn.bias_add(merge_featureMap, b)
    return tf.nn.relu(out, name=name)

def max_pool(x, f_h, f_w, s_y, s_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, f_h, f_w, 1], strides=[1, s_y, s_x, 1],
                          padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob=keep_prob)

class AlexModel(object):

    def _init_(self, x,keep_prop, num_classes, skip_layer, weights_path):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prop
        self.SKIP_LAYER = skip_layer
        self.WEIGHTS_PATH = weights_path
        self.LEARNING_RATW = config.LEARNING_RATE
        self.create()

    def create(self):
        # conv layer1
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, padding='VALID', name='conv1')
        lrn1 = lrn(conv1, 2, 2e-05, 0.75, name='lrn1')
        pool1 = max_pool(lrn1, 3, 3, 2, 2, name='pool1', padding='VALID')

        # conv layer2
        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, groups=2, padding='VALID', name='conv2')
        lrn2 = lrn(conv2, 2, 2e-05, 0.75, name='lrn2')
        pool2 = max_pool(lrn2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # conv layer3
        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, name='conv3')

        # conv layer4
        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, groups=2, name='conv4')

        # conv layer5
        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, name='pool5', padding='VALID')

        # layer6  Flatten->FC ->Dropout
        flattend = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fcLayer(flattend, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # FC layer7
        fc7 = fcLayer(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # FC layer-> output 4096Xclass_num unscaled activations
        self.fc8 = fcLayer(dropout7, 4096, self.NUM_CLASSES, name='fc8')
        return self.fc8

    def load_initial_weights(self, session):
        '''
        load weights from file into network
        '''
        # load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()
        # loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            # check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):
                    # Assin weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:
                        # biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        # weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

    # softmax_cross_entropy_with_logits 先执行softmax,再执行cross_entropy_with_logits
    def loss(self,y_pred=None,y_true = None):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


    def optimize(self,loss=None):
        return tf.train.AdamOptimizer().minimize(loss=loss)