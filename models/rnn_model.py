'''
@Description: In User Settings Edit
@Author: zhansu
@Date: 2019-07-02 22:32:58
@LastEditTime: 2019-07-04 17:05:28
@LastEditors: Please set LastEditors
'''

import tensorflow as tf
from .model import Model
from tensorflow.contrib import rnn


class RNN_model(Model):
    """
    this class is implement the RNN model
        :param Model: 
    """

    def lstm_cell(self, name):
        with tf.variable_scope('forward' + name):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, 1.0)
        with tf.variable_scope('backward' + name):
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)

        if self.dropout_keep_prob_holder is not None:
            self.lstm_fw_cell = rnn.DropoutWrapper(
                lstm_fw_cell, output_keep_prob=self.dropout_keep_prob_holder)
            self.lstm_bw_cell = rnn.DropoutWrapper(
                lstm_bw_cell, output_keep_prob=self.dropout_keep_prob_holder)
        return self.lstm_fw_cell, self.lstm_bw_cell

    def lstm_model(self, fw_cell, bw_cell, embedding_sentence, seq_len):
        with tf.variable_scope('scope', reuse=tf.AUTO_REUSE):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, embedding_sentence, sequence_length=seq_len, dtype=tf.float32)
        print("outputs:===>", outputs)
        # 3. concat output
        # [batch_size,sequence_length,hidden_size*2]
        output_rnn = tf.concat(outputs, axis=2)
        # self.output_rnn_last = tf.reduce_mean(output_rnn,axis = 1) #[batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2] #TODO
        return output_rnn

    def encode_sentence(self):
        fw_cell, bw_cell = self.lstm_cell('rnn')
        self.U = tf.Variable(tf.truncated_normal(
            shape=[self.hidden_size * 2, self.hidden_size * 2], stddev=0.01, name='U'))

        self.q_rnn = self.lstm_model(
            fw_cell, bw_cell, self.q_embedding, self.q_len)
        self.a_pos_rnn = self.lstm_model(
            fw_cell, bw_cell, self.a_embedding, self.a_len)
        self.a_neg_rnn = self.lstm_model(
            fw_cell, bw_cell, self.a_neg_embedding, self.a_neg_len)

        self.encode_q_pos, self.encode_a_pos = self.rnn_attention(
            self.q_rnn, self.a_pos_rnn, self.q_mask, self.a_mask)
        self.encode_q_neg, self.encode_a_neg = self.rnn_attention(
            self.q_rnn, self.a_neg_rnn, self.q_mask, self.a_neg_mask)

    def traditional_attention(self, q, a, q_mask, a_mask):

        self.max_input_left = tf.shape(q)[1]
        print("max_input_left==>{}".format(self.max_input_left))
        self.max_input_right = tf.shape(a)[1]
        print("max_input_right==>{}".format(self.max_input_right))
        # [batch ,max_input_left,vector_size] -> [batch,1,vector_size]
        q = tf.reduce_mean(q, axis=1, keep_dims=True)
        first = tf.matmul(tf.reshape(
            a, [-1, self.hidden_size * 2]), self.U)  # [batch * max_input_right, vector_size]  mat [vector_size * vector_size]
        second = tf.reshape(
            first, [-1, self.max_input_right, self.hidden_size * 2])  # [batch, max_input_right,vector_size]
        alpha = tf.nn.softmax(
            tf.matmul(second, tf.transpose(q, perm=[0, 2, 1])), 1)  # [batch, max_input_right,vector_size] mat [batch,vector_size,1] -> [batch,max_input_right,1]
        # [batch,max_input_right,vector_size] * [batch,max_input_right,1] -> [batch,max_input_right]
        a_attention = tf.reduce_sum(tf.multiply(a, alpha), axis=1)
        return tf.squeeze(q, axis=1), a_attention

    def rnn_attention(self, q_rnn, a_rnn, q_mask, a_mask):
        if self.attention == 'tradition':
            return self.traditional_attention(q_rnn, a_rnn, q_mask, a_mask)
