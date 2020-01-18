'''
@Description: this is the unigram implementation of question and answering
@Author: zhansu
@Date: 2019-07-07 22:16:23
@LastEditTime: 2019-07-07 23:06:00
@LastEditors: Please set LastEditors
'''

import tensorflow as tf 
from models.model import Model

class Unigram(Model):
    """
    unigram model 
        :param Model: 
    """
    def encode_sentence(self):

        # self.q_w = tf.Variable(tf.random.normal([1,self.q_len]),name = 'q_w')
        # self.a_w = tf.Variable(tf.random.normal([1,self.a_len]),name = 'a_w')
        self.encode_q_pos = tf.reduce_sum(self.q_embedding,axis = 1,name = 'encode_q_pos')#[batch,seq_len,embedding]->[batch,embedding]
        self.encode_a_pos = tf.reduce_sum(self.a_embedding,axis = 1,name = 'encode_a_pos')
        self.encode_q_neg = tf.reduce_sum(self.q_embedding,axis = 1,name = 'encode_q_neg')
        self.encode_a_neg = tf.reduce_sum(self.a_neg_embedding,axis = 1,name = 'encode_a_neg')


