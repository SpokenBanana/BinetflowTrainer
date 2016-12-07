import numpy as np
import pickle
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score
from datetime import datetime
from utils import *
from joblib import Parallel, delayed
import tensorflow as tf


"""
Notes:
    look a step ahead.
    Look at all 13 files (with dt and rf only)
    look at all files together and run analysis with bot type (instead of file)
    tensorflow, layers
    count support, test for each label
    to ipython notebook
"""

class Summarizer:
    def __init__(self):
        self.data = {
            'n_conn': 0,
            'avg_duration': 0,
            'n_udp': 0,
            'n_tcp': 0,
            'n_icmp': 0,
            'n_sports>1024': 0,
            'n_sports<1024': 0,
            'n_dports>1024': 0,
            'n_dports<1024': 0,
            'n_s_a_p_address': 0,
            'n_s_b_p_address': 0,
            'n_s_c_p_address': 0,
            'n_s_na_p_address': 0,
            'n_d_a_p_address': 0,
            'n_d_b_p_address': 0,
            'n_d_c_p_address': 0,
            'n_d_na_p_address': 0,
            'normal_flow_count': 0,
            'background_flow_count': 0
        }
        self.is_attack = 0  # would be 1 if it is an attack, set 0 by default
        self._duration = 0
        self.used = False

    def add(self, item):
        self.used = True
        self.data['n_conn'] += 1

        proto = 'n_%s' % item['proto']
        if proto in self.data:
            self.data[proto] += 1

        self._duration += float(item['dur'])
        self.data['avg_duration'] = self._duration / self.data['n_conn']

        # sometimes ports are in a weird format so exclude them for now
        try:
            if int(item['sport']) < 1024:
                self.data['n_sports<1024'] += 1
            else:
                self.data['n_sports>1024'] += 1
        except Exception:
            pass

        try:
            if int(item['dport']) < 1024:
                self.data['n_dports<1024'] += 1
            else:
                self.data['n_dports>1024'] += 1
        except Exception:
            pass

        if 'Botnet' in item['label']:
            self.is_attack = 1
        elif 'Normal' in item['label']:
            self.data['normal_flow_count'] += 1
        elif 'Background' in item['label']:
            self.data['background_flow_count'] += 1

        self.data['n_s_%s_p_address' % classify(item['srcaddr'])] += 1
        self.data['n_d_%s_p_address' % classify(item['dstaddr'])] += 1

def train_and_test_with(features, labels, classifier):
    """
        classifier: the str rep machine learning algorithm being used
    """
    clf = get_classifier(classifier)

    feat_train, feat_test, label_train, label_test = train_test_split(
        features, labels, test_size=0.5, random_state=42)

    clf.fit(feat_train, label_train)

    predicted_labels = clf.predict(feat_test)
    attack_train = sum(label_train)
    attack_test = sum(label_test)
    result = {}
    result['recall'] = recall_score(label_test, predicted_labels)
    result['accuracy'] = accuracy_score(label_test, predicted_labels)
    result['precision'] = precision_score(label_test, predicted_labels)
    result['f1 score'] = f1_score(label_test, predicted_labels)
    result['attacks'] = sum(labels)
    result['normal count'] = len(labels) - result['attacks']
    result['training size'] = len(feat_train)
    result['1'] = '%s, %s' % (attack_train,  attack_test)
    result['0'] = '%s, %s' % (len(label_train)-attack_train,
                              len(label_test) - attack_test)
    return result


def train_and_test_step(features, labels, classifier, step):
    correct = 0
    clf = get_classifier(classifier)

    for i in range(len(features) - step):
        clf.fit([features[i]], [labels[i]])
        if labels[i+step] == clf.predict([features[i+step]]):
            correct += 1

    return correct / (len(features) - step)


def aggregate_file(interval, file_name, start=None):
    """ Aggregate the data within the windows of time

        interval:       time in seconds to aggregate data
        file_name:      which file to record
        start:          start time to record data, if none given then it starts
                        from te beginning.

        returns: array of the aggregated data in each interval
    """
    if start is None:
        start = get_start_time_for(file_name)

    start = datetime.strptime(start, TIME_FORMAT)
    summaries = [Summarizer() for _ in range(10)]
    with open(file_name, 'r+') as data:
        headers = data.readline().strip().lower().split(',')
        for line in data:
            args = line.strip().split(',')
            time = datetime.strptime(args[0], TIME_FORMAT)
            window = int((time - start).total_seconds() / interval)
            if window < 0:
                continue
            if window >= len(summaries):
                for i in range(window + 1):
                    summaries.append(Summarizer())
            item = dict(zip(headers, args))
            summaries[window].add(item)
    return [s for s in summaries if s.used]

def aggregate_and_pickle(interval, file_name, start=None):
    if start is None:
        start = get_start_time_for(file_name)

    summary = aggregate_file(interval, file_name, start)
    pickle_summarized_data(interval, start, file_name, summary)
    return summary


def train_with_tensorflow(features, labels):
    correctness = 0
    with tf.Session() as sess:
        # sess = tf.InteractiveSession()

        x = tf.placeholder(tf.float32, shape=[None, 19])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        W = tf.Variable(tf.zeros([19, 2]))
        b = tf.Variable(tf.zeros([2]))

        sess.run(tf.global_variables_initializer())
        y = tf.matmul(x, W) + b
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step = tf.train.GradientDescentOptimizer(
                0.5).minimize(cross_entropy)

        labels = to_tf_label(labels) 
        train_step.run(feed_dict={x: features, y_: labels})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        correctness = accuracy.eval(feed_dict={x: features, y_: labels})

    return correctness 


def run_analysis_with(interval, file_name, start_time=None, use_pickle=True):
    if start_time is None:
        start_time = get_start_time_for(file_name)

    start = datetime.strptime(start_time, TIME_FORMAT)
    file_num = get_file_num(file_name)
    directory = 'runs_of_%ss/' % interval

    if not os.path.exists(directory):
        os.makedirs(directory)

    mls = ['dt', 'rf']

    print('starting %d %s' % (interval, file_name))
    if use_pickle:
        print('loading pickle')
        summaries = get_saved_data(interval, start_time, file_name)
        if summaries is None:
            print('failed to load pickle. Aggregating data')
            summaries = aggregate_file(interval, file_name, start)
            print('finished aggregating, pickling data...')
            pickle_summarized_data(interval, start_time, file_name, summaries)
            print('data pickled')
        else:
            print('loaded picke')
    else:
        print('aggregating data')
        summaries = aggregate_file(interval, file_name, start)
        print('finished aggregating, pickling data...')
        pickle_summarized_data(interval, start_time, file_name, summaries)
        print('data pickled')

    features, labels = get_feature_labels(summaries)
    for ml in mls:
        print('testing with %s' % ml)
        result = train_and_test_with(features, labels, ml)
        path = '%srun_%s_%s.txt' % (directory, file_num, ml)
        save_results(path, file_name, start_time, interval, result)

def tensorflow_analysis(interval, file_name, start=None):
    if start is None:
        start_time = get_start_time_for(file_name)

    start = datetime.strptime(start_time, TIME_FORMAT)
    file_num = get_file_num(file_name)
    directory = 'runs_of_%ss/' % interval

    if not os.path.exists(directory):
        os.makedirs(directory)

    print('starting %d %s' %  (interval, file_name))
    summaries = get_saved_data(interval, start_time, file_name)
    if summaries is None:
        summaries = aggregate_and_pickle(interval, file_name, start_time)

    features, labels = get_feature_labels(summaries)
    
    print('Running tensorflow...')
    accuracy = train_with_tensorflow(features, labels)
    print('Done.')
    
    path = '%srun_%s_tf.txt' % (directory, file_num)
    save_results(path, file_name, start_time, interval, {'accuracy': accuracy})

if __name__ == '__main__':
    all_intervals = [5, 10, 20, 30, 60]
    interval = 5  # in seconds

    binet_files = get_binetflow_files()

    for binet in binet_files:
        for i in all_intervals:
            tensorflow_analysis(i, binet)

