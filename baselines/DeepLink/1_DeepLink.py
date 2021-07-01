# libraries
# -*- coding: UTF-8 -*-
from gensim.models import word2vec, KeyedVectors, Word2Vec
import preprocessor

import tensorflow as tf
import numpy as np
import logging
import os
import json
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
tf.compat.v1.disable_eager_execution()
from decimal import *
getcontext().prec = 28


logging.basicConfig(filename='Your file name to save the logs', filemode='a', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

VECTOR_SIZE = 100
TRAIN_ITERS = 300
BATCH_SIZE = 16

HIDDEN_SIZE = 100
N_INPUTS = 100
LEARNING_RATE = 0.01

LSTM_KEEP_PROB = 0.9

MAX_RECORD = {'step': -1, 'acc': 0.0}
MAX_PRECISION = {'step': -1, 'acc': 0.0}
MAX_RECALL = {'step': -1, 'acc': 0.0}
MAX_F_MEASURE = {'step': -1, 'acc': 0.0}

wordModel = KeyedVectors.load_word2vec_format('.bin file of word2vec of your data', binary=True)


def text2vec(text, isHtml):
    if isHtml:
        seqs = preprocessor.processHTMLNoCamel(text)
    else:
        seqs = preprocessor.preprocessNoCamel(text)
    res = []
    for seq in seqs:
        for word in seq:
            try:
                res.append(wordModel[word])
            except KeyError:
                res.append(np.zeros(VECTOR_SIZE))
    return res

# shape = [None, seq len, Vec size]
def read_data(df):
    X1 = []
    X2 = []
    T = []
    L1 = []
    L2 = []
    LT = []
    Y = []
    logging.info("Loaded the file")
    for index, row in df.iterrows():
        commit = text2vec(row['message_processed'], False)
        issue = text2vec(row['description_processed'], True)
        title = text2vec(row['summary_processed'], False)
        if len(commit) < 5:
            continue
        if len(issue)+len(title) < 5:
            continue
        L1.append(len(commit))
        X1.append(commit)
        L2.append(len(issue))
        X2.append(issue)
        LT.append(len(title))
        T.append(title)
        Y.append(float(row['label']))
        gc.collect()
    logging.info("reading data is done")
    return X1, X2, T, L1, L2, LT, Y


# shape=[batch_size, None]
def make_batches(data, batch_size):
    logging.info("batching is starting")
    X1, X2, T, L1, L2, LT, Y = data
    num_batches = len(Y) // batch_size
    data1 = np.array(X1[: batch_size*num_batches])
    data1 = np.reshape(data1, [batch_size, num_batches])
    data_batches1 = np.split(data1, num_batches, axis=1)  #  list
    data_batches1_rs = []
    for d1 in data_batches1:
        sub_batch = []
        maxD = 0
        for d in d1:
            for dt in d:
                maxD = max(maxD, len(dt))
        for d in d1:
            for dt in d:
                todo = maxD - len(dt)
                for index in range(todo):
                    dt.append(np.zeros(VECTOR_SIZE))
                sub_batch.append(np.array(dt))
        data_batches1_rs.append(np.array(sub_batch))

    data2 = np.array(X2[: batch_size*num_batches])
    data2 = np.reshape(data2, [batch_size, num_batches])
    data_batches2 = np.split(data2, num_batches, axis=1)
    data_batches2_rs = []
    for d2 in data_batches2:
        sub_batch = []
        maxD = 0
        for d in d2:
            for dt in d:
                maxD = max(maxD, len(dt))
        for d in d2:
            for dt in d:
                todo = maxD - len(dt)
                for index in range(todo):
                    dt.append(np.zeros(VECTOR_SIZE))
                sub_batch.append(np.array(dt))
        data_batches2_rs.append(np.array(sub_batch))

    dataT = np.array(T[: batch_size*num_batches])
    dataT = np.reshape(dataT, [batch_size, num_batches])
    data_batchesT = np.split(dataT, num_batches, axis=1)  #  list
    data_batchesT_rs = []
    for d3t in data_batchesT:
        sub_batch = []
        maxD = 0
        for d in d3t:
            for dt in d:
                maxD = max(maxD, len(dt))
        for d in d3t:
            for dt in d:
                todo = maxD - len(dt)
                for index in range(todo):
                    dt.append(np.zeros(VECTOR_SIZE))
                sub_batch.append(np.array(dt))
        data_batchesT_rs.append(np.array(sub_batch))

    len1 = np.array(L1[: batch_size*num_batches])
    len1 = np.reshape(len1, [batch_size, num_batches])
    len_batches1 = np.split(len1, num_batches, axis=1)
    len_batches1 = np.reshape(np.array(len_batches1), [num_batches, BATCH_SIZE])

    len2 = np.array(L2[: batch_size * num_batches])
    len2 = np.reshape(len2, [batch_size, num_batches])
    len_batches2 = np.split(len2, num_batches, axis=1)
    len_batches2 = np.reshape(np.array(len_batches2), [num_batches, BATCH_SIZE])

    lenT = np.array(LT[: batch_size * num_batches])
    lenT = np.reshape(lenT, [batch_size, num_batches])
    len_batchesT = np.split(lenT, num_batches, axis=1)
    len_batchesT = np.reshape(np.array(len_batchesT), [num_batches, BATCH_SIZE])

    label = np.array(Y[: batch_size*num_batches])
    label = np.reshape(label, [batch_size, num_batches])
    label_batches = np.split(label, num_batches, axis=1)
    logging.info("batching is done!!!")
    return list(zip(data_batches1_rs, data_batches2_rs, data_batchesT_rs, len_batches1, len_batches2, len_batchesT, label_batches))


class MyModel(object):
    def __init__(self, is_training, batch_size):
        self.batch_size = batch_size

        self.input1 = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, None, VECTOR_SIZE])
        self.input2 = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, None, VECTOR_SIZE])
        self.inputT = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, None, VECTOR_SIZE])
        self.len1 = tf.compat.v1.placeholder(tf.int32, [BATCH_SIZE, ])
        self.len2 = tf.compat.v1.placeholder(tf.int32, [BATCH_SIZE, ])
        self.lent = tf.compat.v1.placeholder(tf.int32, [BATCH_SIZE, ])
        self.target = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, 1])

        with tf.compat.v1.variable_scope("message_processed"):
            outputs1, states1 = self.RNN(self.input1, self.len1, is_training)
        with tf.compat.v1.variable_scope("description_processed"):
            outputs2, states2 = self.RNN(self.input2, self.len2, is_training)
        with tf.compat.v1.variable_scope("summary_processed"):
            outputs3, states3 = self.RNN(self.inputT, self.lent, is_training)

        newoutput1 = states1[-1].h
        newoutput2 = states2[-1].h
        newoutput3 = states3[-1].h

        # Define loss and optimizer
        self.cos_score = self.getScore(newoutput1, newoutput2, newoutput3)
        self.loss_op = self.getLoss(self.cos_score, self.target)

        if not is_training:
            return

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.train_op = optimizer.minimize(self.loss_op)

    def getScore(self, state1, state2, state3):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(state1 * state1, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(state2 * state2, 1))
        pooled_mul_12 = tf.reduce_sum(state1 * state2, 1)
        score1 = tf.compat.v1.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores1")  # +1e-8 avoid 'len_1/len_2 == 0'
        score1 = tf.compat.v1.reshape(score1, [BATCH_SIZE, 1])

        pooled_len_3 = tf.sqrt(tf.reduce_sum(state3 * state3, 1))
        pooled_mul_13 = tf.reduce_sum(state1 * state3, 1)
        score2 = tf.compat.v1.div(pooled_mul_13, pooled_len_1 * pooled_len_3 + 1e-8, name="scores2")  # +1e-8 avoid 'len_1/len_2 == 0'
        score2 = tf.compat.v1.reshape(score2, [BATCH_SIZE, 1])

        score = tf.compat.v1.concat([score1, score2], 1)
        score = tf.compat.v1.reduce_max(score, 1)
        return tf.compat.v1.reshape(score, [BATCH_SIZE, 1])

    #  |t - cossimilar(state1, state2)|
    def getLoss(self, score, t):
        rs = t - score
        rs = tf.compat.v1.abs(rs)
        return tf.compat.v1.reduce_sum(rs)

    def RNN(self, input_data, seq_len, is_training):
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [
            tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=dropout_keep_prob)
            for _ in range(1)
        ]
        rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_cells)
        outputs, state = tf.compat.v1.nn.dynamic_rnn(rnn_cell, input_data, sequence_length=seq_len, dtype=tf.float32)
        return outputs, state


def run_epoch(session, model, batches, step):
    # session.run(model.init_state)
    for x1, x2, t, l1, l2, lt, y in batches:
        loss, _ = session.run([model.loss_op, model.train_op],
                           feed_dict={model.input1: x1, model.input2: x2, model.inputT: t, model.len1: l1, model.len2: l2, model.lent: lt, model.target: y})
        logging.info("At the step %d, the loss is %f" % (step, loss))


def test_epoch(session, model, batches, step):
    temp = []
    total_correct = 0
    total_tests = len(batches) * BATCH_SIZE
    index = 0
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    for x11, x21, t1, l11, l21, lt1, y1 in batches:
        score, loss = session.run([model.cos_score, model.loss_op],
                                feed_dict={model.input1: x11, model.input2: x21, model.inputT: t1, model.len1: l11, model.len2: l21, model.lent: lt1,
                                           model.target: y1})
        temp.append(loss)
        total_correct = total_correct + get_correct(score, y1, index, len(batches))
        index = index + 1
        measure = get_measure(score, y1)
        total_TP = total_TP + measure[0]
        total_TN = total_TN + measure[1]
        total_FP = total_FP + measure[2]
        total_FN = total_FN + measure[3]

    precision = float(total_TP) / (total_TP + total_FP+1e-8)
    recall = float(total_TP) / (total_TP + total_FN+1e-8)
    if precision==0 and recall==0:
        f_measure=0
    else:
        f_measure = (2 * precision * recall) / (precision + recall)

    logging.info("At the test %d, the avg loss is %f, the accuracy is %f" % (step, np.mean(np.array(temp)), float(total_correct) / total_tests))
    logging.info("At the test %d, TP:%d TN:%d FP:%d FN:%d" % (step, total_TP, total_TN, total_FP, total_FN))
    logging.info("At the test %d, precision:%f recall:%f f_measure:%f" % (step, precision, recall, f_measure))
    if (float(total_correct) / total_tests) > MAX_RECORD['acc']:
        MAX_RECORD['step'] = step
        MAX_RECORD['acc'] = float(total_correct) / total_tests
    if precision > MAX_PRECISION['acc']:
        MAX_PRECISION['step'] = step
        MAX_PRECISION['acc'] = precision
    if recall > MAX_RECALL['acc']:
        MAX_RECALL['step'] = step
        MAX_RECALL['acc'] = recall
    if f_measure > MAX_F_MEASURE['acc']:
        MAX_F_MEASURE['step'] = step
        MAX_F_MEASURE['acc'] = f_measure
    logging.info("MAX is at step %d: %f" % (MAX_RECORD['step'], MAX_RECORD['acc']))
    logging.info("MAX precision is at step %d: %f" % (MAX_PRECISION['step'], MAX_PRECISION['acc']))
    logging.info("MAX recall is at step %d: %f" % (MAX_RECALL['step'], MAX_RECALL['acc']))
    logging.info("MAX f_measure is at step %d: %f" % (MAX_F_MEASURE['step'], MAX_F_MEASURE['acc']))


def get_correct(score, target, index, NUM):
    result = 0
    for i in range(len(target)):
        if target[i][0] == 1 and score[i][0] > 0.5:
            result = result + 1
        elif target[i][0] == 0 and score[i][0] < 0.5:
            result = result + 1
    return result


def get_measure(score, target):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(target)):
        if target[i][0] == 1:
            if score[i][0] > 0.5:
                TP = TP+1
            else:
                FN = FN+1
        elif target[i][0] == 0:
            if score[i][0] < 0.5:
                TN = TN+1
            else:
                FP = FP+1
    return TP, TN, FP, FN


def main():
    df = pd.read_parquet('data/Your Project file')
    train_df = df.loc[df['train_flag'] == 1]
    test_df = df.loc[df['train_flag'] == 0]
    
#     batching
    train_batches = make_batches(read_data(df=df_train), BATCH_SIZE)
    test_batches = make_batches(read_data(df=df_test), BATCH_SIZE)
#     creating model
    with tf.compat.v1.variable_scope("rnn_model", reuse=None):
        train_model = MyModel(True, BATCH_SIZE)
    with tf.compat.v1.variable_scope("rnn_model", reuse=True):
        test_model = MyModel(False, BATCH_SIZE)
#     Training the model
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        sess.run(init)
        logging.info("Test Set: %d" % (len(test_batches) * BATCH_SIZE))

        for step in range(TRAIN_ITERS):
            logging.info("Step: " + str(step))
            run_epoch(session=sess, model=train_model, batches=train_batches, step=step)
            test_epoch(session=sess, model=test_model, batches=test_batches, step=step)
        saver.save(sess, 'Path to save the model chack point', global_step=TRAIN_ITERS)
        logging.info("Optimization Finished!")

        
if __name__ == "__main__":
    main()