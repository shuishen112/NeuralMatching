'''
@Description: the datahelper of the model
@Author: zhansu
@Date: 2019-06-28 20:14:28
@LastEditTime: 2019-07-10 21:22:18
@LastEditors: Please set LastEditors
'''
# -*- coding:utf-8 -*-


import os
import numpy as np
import tensorflow as tf
import string
from collections import Counter
import pandas as pd
import tqdm
import logging

logging.getLogger("QA")


def load_train_file(data_dir, filter=False):
    """
    load the dataset
            :param data_dir: the data_dir
            :param filter=False: whether clean the dataset
    """
    train_df = pd.read_csv(os.path.join(data_dir, 'train.txt'), header=None, sep='\t', names=[
        'question', 'answer', 'flag'], quoting=3).fillna('')
    if filter:
        train_df = remove_the_unanswered_sample(train_df)
    dev_df = pd.read_csv(os.path.join(data_dir, 'dev.txt'), header=None, sep='\t', names=[
        'question', 'answer', 'flag'], quoting=3).fillna('')
    if filter:
        dev_df = remove_the_unanswered_sample(dev_df)
    test_df = pd.read_csv(os.path.join(data_dir, 'test.txt'), header=None, sep='\t', names=[
        'question', 'answer', 'flag'], quoting=3).fillna('')
    if filter:
        test_df = remove_the_unanswered_sample(test_df)
    return train_df, test_df, test_df


def cut(sentence):
    """
    split the sentence to tokens
            :param sentence: raw sentence
    """
    tokens = sentence.lower().split()

    return tokens


def remove_the_unanswered_sample(df):
    """
    clean the dataset
            :param df: dataframe
    """
    counter = df.groupby("question").apply(lambda group: sum(group["flag"]))
    questions_have_correct = counter[counter > 0].index
    counter = df.groupby("question").apply(
        lambda group: sum(group["flag"] == 0))
    questions_have_uncorrect = counter[counter > 0].index
    counter = df.groupby("question").apply(lambda group: len(group["flag"]))
    questions_multi = counter[counter > 1].index

    return df[df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()


def get_alphabet(corpuses):
    """
    obtain the dict
            :param corpuses: 
    """
    word_counter = Counter()

    for corpus in corpuses:
        for texts in [corpus["question"].unique(), corpus["answer"]]:
            for sentence in texts:
                tokens = cut(sentence)
                for token in tokens:
                    word_counter[token] += 1
    print("there are {} words in dict".format(len(word_counter)))
    logging.info("there are {} words in dict".format(len(word_counter)))
    word_dict = {word: e + 2 for e, word in enumerate(list(word_counter))}
    word_dict['UNK'] = 1
    word_dict['<PAD>'] = 0

    return word_dict


def get_embedding(alphabet, filename="", embedding_size=100):
    embedding = np.random.rand(len(alphabet), embedding_size)
    with open(filename, encoding='utf-8') as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print('epch %d' % i)
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size = items[0], items[1]
                print((vocab_size, embedding_size))
            else:
                word = items[0]
                if word in alphabet:
                    embedding[alphabet[word]] = items[1:]

    print('done')

    return embedding


def encode_to_split(sentence, alphabet):
    """
    convert the sentence to ids
            :param sentence: raw sentence
            :param alphabet: word_dict of the dataset
    """
    tokens = cut(sentence)
    seq = [alphabet[w] if w in alphabet else alphabet['[UNK]'] for w in tokens]
    return seq


def get_mini_batch_test(df, alphabet, batch_size):
    """
    get the batch_data for the test
            :param df: DataFrame
            :param alphabet: word_dict
            :param batch_size: batch_size
    """
    q = []
    a = []

#   get the q list and a list
    for index, row in df.iterrows():
        question = encode_to_split(row["question"], alphabet)
        answer = encode_to_split(row["answer"], alphabet)
        q.append(question)
        a.append(answer)
    m = 0
    n = len(q)
    idx_list = np.arange(m, n, batch_size)
    mini_batches = []
#   batch the dataset
    for idx in idx_list:
        mini_batches.append(np.arange(idx, min(idx + batch_size, n)))
    for mini_batch in mini_batches:
        mb_q = [q[t] for t in mini_batch]
        mb_a = [a[t] for t in mini_batch]

        mb_q, mb_q_mask = prepare_data(mb_q)
        mb_a, mb_a_mask = prepare_data(mb_a)

        yield(mb_q, mb_a, mb_q_mask, mb_a_mask)


def get_mini_batch(df, alphabet, batch_size, sort_by_len=True, shuffle=False):
    """
    ge the dataset of the train batch
            :param df: DataFrame
            :param alphabet: word_dict
            :param batch_size: batch_size
            :param sort_by_len=True: 
            :param shuffle=False: whether shuffle the dataset
    """
    q = []
    a = []
    neg_a = []
    for question in df['question'].unique():
        group = df[df["question"] == question]
        pos_answers = group[group["flag"] == 1]["answer"]
        neg_answers = group[group["flag"] == 0]["answer"].reset_index()
#       random sample the negtive sample
        for pos in pos_answers:
            if len(neg_answers.index) > 0:
                neg_index = np.random.choice(neg_answers.index)
                neg = neg_answers.loc[neg_index, ]["answer"]
            seq_q = encode_to_split(question, alphabet)
            seq_a = encode_to_split(pos, alphabet)
            seq_neg_a = encode_to_split(neg, alphabet)
            q.append(seq_q)
            a.append(seq_a)
            neg_a.append(seq_neg_a)
    # sorted by the question

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]), reverse=True)

    if sort_by_len:
        sorted_index = len_argsort(q)
        q = [q[i] for i in sorted_index]
        a = [a[i] for i in sorted_index]
        neg_a = [neg_a[i] for i in sorted_index]

    # get batch
    m = 0
    n = len(q)

    idx_list = np.arange(m, n, batch_size)
    # shuffle the dataset

    if shuffle:
        np.random.shuffle(idx_list)

    mini_batches = []
    for idx in idx_list:
        mini_batches.append(np.arange(idx, min(idx + batch_size, n)))

    for mini_batch in mini_batches:
        mb_q = [q[t] for t in mini_batch]
        mb_a = [a[t] for t in mini_batch]
        mb_neg_a = [neg_a[t] for t in mini_batch]

        mb_q, mb_q_mask = prepare_data(mb_q)
        mb_a, mb_a_mask = prepare_data(mb_a)

        mb_neg_a, mb_a_neg_mask = prepare_data(mb_neg_a)

        yield(mb_q, mb_a, mb_neg_a, mb_q_mask, mb_a_mask, mb_a_neg_mask)


def prepare_data(seqs):
    """
    prepare the dataset by the batch
            :param seqs: 
    """
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    max_len = max(max_len,5) #noting that the max len is larger than the filter_size of the convolution window size
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype('float')
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    # print( x, x_mask)
    return x, x_mask
