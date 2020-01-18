'''
@Description: 
@Author: zhansu
@Date: 2019-06-28 20:14:28
@LastEditTime : 2020-01-18 16:23:01
@LastEditors  : Please set LastEditors
'''
from tensorflow import flags
import tensorflow as tf
from config import Singleton
import data_helper
import time
import datetime
import os
import models
import numpy as np
import evaluation
import sys
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


now = int(time.time())
timeArray = time.localtime(now)
log_filename = "log/" + time.strftime("%Y%m%d", timeArray)
if not os.path.exists(log_filename):
    os.makedirs(log_filename)

program = os.path.basename('QA')
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_filename+'/{}_qa.log'.format(time.strftime("%H%M", timeArray)), filemode='w')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


args = Singleton().get_rnn_flag()
# args = Singleton().get_cnn_flag()

opts = args.flag_values_dict()
for item in opts:
    logger.info('{} : {}'.format(item, opts[item]))

logger.info('load data ...........')
train, test, dev = data_helper.load_train_file(
    opts['data_dir'], filter=args.clean)

q_max_sent_length = max(map(lambda x: len(x), train['question'].str.split()))
a_max_sent_length = max(map(lambda x: len(x), train['answer'].str.split()))

alphabet = data_helper.get_alphabet([train, test, dev])
logger.info('the number of words :%d ' % len(alphabet))

embedding = data_helper.get_embedding(
    alphabet, opts['embedding_file'], embedding_size=opts['embedding_size'])

opts["embeddings"] = embedding
opts["vocab_size"] = len(alphabet)
opts["max_input_right"] = a_max_sent_length
opts["max_input_left"] = q_max_sent_length
opts["filter_sizes"] = list(map(int, args.filter_sizes.split(",")))

with tf.Graph().as_default():

    model = models.setup(opts)
    model._model_stats()
    for i in range(args.num_epoches):
        data_gen = data_helper.get_mini_batch(train, alphabet, args.batch_size)
        model.train(data_gen,i)

        test_datas = data_helper.get_mini_batch_test(
            test, alphabet, args.batch_size)

        predicted_test = model.predict(test_datas)
        map_, mrr_, p_1 = evaluation.evaluationBypandas(test, predicted_test)

        logger.info('map:{}--mrr:{}--p@1--{}'.format(map_, mrr_, p_1))
        print('map:{}--mrr:{}--p@1--{}'.format(map_, mrr_, p_1))
