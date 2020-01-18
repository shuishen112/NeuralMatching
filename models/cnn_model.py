'''
@Description: cnn model of question answering
@Author: your name
@Date: 2019-07-08 11:18:09
@LastEditTime: 2019-07-10 22:13:41
@LastEditors: Please set LastEditors
'''

import tensorflow as tf
from models.model import Model


class CNN(Model):

    def cnn_encode(self, q_emb, a_emb, a_neg_emb):
        """
        basis cnn encoding of the model
            :param self: 
            :param q_emb: 
            :param a_emb: 
        """
        with tf.name_scope("cnn_encoding"):
            q_emb = tf.expand_dims(q_emb, -1)
            a_emb = tf.expand_dims(a_emb, -1)
            a_neg_emb = tf.expand_dims(a_neg_emb, -1)

            outputs_q = []
            outputs_a = []
            outputs_a_neg = []
            for filter_size in self.filter_sizes:
                with tf.name_scope('conv-pool-%s' % filter_size):
                    kernel_size = [filter_size, self.embedding_size]
                    # encode the question
                    conv1 = tf.layers.conv2d(q_emb, self.num_filters, strides=[1, 1],
                                             kernel_size = kernel_size,
                                             padding='VALID',
                                             reuse=None,
                                             activation=tf.nn.relu,
                                             name='conv_{}'.format(
                                                 str(filter_size))
                                             )

                    pool1 = tf.reduce_max(conv1,axis = 1,keep_dims = True)

                    outputs_q.append(pool1)
                    # encode the app_name notice that the reuse will make the query
                    # and app share the parameters
                    conv2 = tf.layers.conv2d(a_emb, self.num_filters, strides=[1, 1],
                                             kernel_size = kernel_size,
                                             padding='VALID',
                                             reuse=True,
                                             activation=tf.nn.relu,
                                             name='conv_{}'.format(
                                                 str(filter_size))
                                             )
                    pool2 = tf.reduce_max(conv2,axis = 1,keep_dims = True)

                    outputs_a.append(pool2)

                    # encode the name

                    conv3 = tf.layers.conv2d(a_neg_emb, self.num_filters, strides=[1, 1],
                                             kernel_size = kernel_size,
                                             padding='VALID',
                                             reuse=True,
                                             activation=tf.nn.relu,
                                             name='conv_{}'.format(str(filter_size)))

                    pool3 = tf.reduce_max(conv3,axis = 1,keep_dims = True)
                    outputs_a_neg.append(pool3)

            # output concat

            num_filters_total = self.num_filters * len(self.filter_sizes)

            h_pool_q = tf.concat(outputs_q, 3)
            h_pool_a = tf.concat(outputs_a, 3)
            h_pool_a_neg = tf.concat(outputs_a_neg, 3)
            q_encode = tf.reshape(h_pool_q, [-1, num_filters_total])
            a_encode = tf.reshape(h_pool_a, [-1, num_filters_total])
            a_neg_encode = tf.reshape(h_pool_a_neg, [-1, num_filters_total])

            return q_encode, a_encode, q_encode, a_neg_encode

    def encode_sentence(self):
        """
        encode the sentence with cnn model
            :param self: 
        """
        self.encode_q_pos, self.encode_a_pos, self.encode_q_neg, self.encode_a_neg = self.cnn_encode(
            self.q_embedding, self.a_embedding, self.a_neg_embedding
        )
