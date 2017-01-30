import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score
from datetime import datetime
from utils import save_results, get_classifier, get_file_num, \
        pickle_summarized_data, get_saved_data, get_binetflow_files, \
        get_feature_labels, to_tf_label, get_start_time_for, TIME_FORMAT
import tensorflow as tf
from summarizer import Summarizer


def train_and_test_with(features, labels, classifier):
    """
        classifier: the str rep machine learning algorithm being used

        :return A dictionary mapping a metric to it's value
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
    result['1'] = '%s, %s' % (attack_train, attack_test)
    result['0'] = '%s, %s' % (len(label_train)-attack_train,
                              len(label_test) - attack_test)
    return result


def train_and_test_step(features, labels, classifier, step):
    if classifier != 'tf':
        clf = get_classifier(classifier)
    last = 0
    acc = 0
    for i in range(step, len(features), step):
        if classifier == 'tf':
            # @Performance: This takes way too long to run.
            acc += train_with_tensorflow(features[last:i], labels[last:i],
                                         [features[i]], [labels[i]])
        else:
            clf.fit(features[last:i], labels[last:i])
            predicted = clf.predict([features[i]])
            acc += accuracy_score([labels[i]], predicted)
        last = i
    return acc / (len(features) - step)


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


def train_with_tensorflow(feat_train, label_train, feat_test=None,
                          label_test=None):
    correctness = 0
    if feat_test is None or label_test is None:
        feat_train, feat_test, label_train, label_test = train_test_split(
                            feat_train, label_train, test_size=0.5,
                            random_state=42)
    label_train = to_tf_label(label_train)
    label_test = to_tf_label(label_test)

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, 19])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        W = tf.Variable(tf.zeros([19, 2]))
        b = tf.Variable(tf.zeros([2]))

        sess.run(tf.global_variables_initializer())
        y = tf.matmul(x, W) + b
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
                cross_entropy)

        train_step.run(feed_dict={x: feat_train, y_: label_train})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        correctness = accuracy.eval(feed_dict={x: feat_test, y_: label_test})

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

    print('starting %d %s' % (interval, file_name))
    summaries = get_saved_data(interval, start_time, file_name)
    if summaries is None:
        summaries = aggregate_and_pickle(interval, file_name, start_time)

    features, labels = get_feature_labels(summaries)

    print('Running tensorflow...')
    result = {'accuracy': train_with_tensorflow(features, labels)}
    print('Done.')

    path = '%srun_%s_tf.txt' % (directory, file_num)
    save_results(path, file_name, start_time, interval, result)
    return result


if __name__ == '__main__':
    all_intervals = [.5, 1, 2, 5]
    interval = 5  # in seconds

    binet_files = get_binetflow_files()

    for binet in binet_files:
        for i in all_intervals:
            tensorflow_analysis(i, binet)
