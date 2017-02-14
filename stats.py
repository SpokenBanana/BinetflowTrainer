from main import train_and_test_step, aggregate_and_pickle, \
        train_and_test_with, train_with_tensorflow
from utils import get_feature_labels, get_binetflow_files, get_start_time_for,\
        get_saved_data, mask_features, get_classifier
from summarizer import Summarizer
import pytablewriter
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from binet_keras import keras_train_and_test
import numpy as np


windows = [.15]  # , 1, 2, 5]
binet_files = get_binetflow_files()


def run_files(f, window):
    if get_saved_data(window, f) is None:
        print('aggregating {} for {}s'.format(f, window))
        aggregate_and_pickle(window, f)


def window_shift(window):
    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = 'Window Shift Accuracy for {}s'.format(window)
    writer.header_list = ['File', 'Descision Tree', 'Random Forest']
    value_matrix = []
    for file_name in binet_files:
        values = []
        feature, label = get_feature_labels(
                get_saved_data(window, file_name))
        values += [file_name, '{0:.4f}'.format(
                        train_and_test_step(feature, label, 'dt', 500)),
                   '{0:.4f}'.format(train_and_test_step(
                        feature, label, 'rf', 500))]
        values.append('{0:.4f}\n'.format(
            train_and_test_step(feature, label, 'tf', 500)))
        value_matrix.append(values)

    writer.value_matrix = value_matrix
    writer.write_table()


def file_stats():
    mls = ['dt', 'rf']
    for window in windows:
        writer = pytablewriter.MarkdownTableWriter()
        writer.table_name = 'File Accuracy for {}s'.format(window)
        writer.header_list = ['File', 'Descision Tree', 'Random Forest',
                                                        'Tensorflow']
        value_matrix = []
        for name in binet_files:
            values = [name]
            feature, label = get_feature_labels(get_saved_data(window, name))
            feature = mask_features(feature)
            for ml in mls:
                r = train_and_test_with(feature, label, ml)
                values.append('{0:.4f}, {1:.4f}, {2:.4f}'.format(
                              r['accuracy'],
                              r['precision'],
                              r['recall']))
                clf = get_classifier(ml)
                values.append('{}'.format(cross_val_score(clf, feature, label,
                                                          cv=5,
                                                          scoring='f1_macro')))
                correctness, precision, recall = \
                    keras_train_and_test(feature, label, dimension=12)
                values.append('{0:.4f}, {1:.4f}, {2:.4f}\n'.format(correctness,
                                                                   precision,
                                                                   recall))
            value_matrix.append(values)

        writer.value_matrix = value_matrix
        writer.write_table()


def kfold_test():
    mls = ['dt', 'rf']
    for window in windows:
        writer = pytablewriter.MarkdownTableWriter()
        writer.table_name = 'KFold validation'
        writer.header_list = ['File', 'Decision Tree', 'Random Forest',
                              'Tensorflow']
        value_matrix = []
        for name in binet_files:
            values = [name]
            feature, label = get_feature_labels(get_saved_data(window, name))
            feature = mask_features(feature)
            for ml in mls:
                clf = get_classifier(ml)
                scores = cross_val_score(clf, feature, label, cv=10, n_jobs=2)
                values.append('{0:.4f}, {1:.4f}'.format(np.mean(scores),
                                                        np.std(scores)))
            kf = KFold(n_splits=10)
            accuracy = []  # , precision, recall = [], [], []
            for train_index, test_index in kf.split(feature):
                x_train, x_test = feature[train_index], feature[test_index]
                y_train, y_test = label[train_index], label[test_index]
                c, p, r = \
                    keras_train_and_test(x_train, y_train, x_test, y_test,
                                         dimension=12)
                accuracy.append(c)
                # precision.append(p)
                # recall.append(r)
            values.append('{0:.4f}, {1:.4f}'.format(np.mean(accuracy),
                                                    np.std(accuracy)))
            value_matrix.append(values)
        writer.value_matrix = value_matrix
        writer.write_table()


def bots_test():
    mls = ['dt', 'rf']
    for window in windows:
        writer = pytablewriter.MarkdownTableWriter()
        writer.table_name = 'Bot accuracy for {}s'.format(window)
        writer.header_list = ['Bot', 'Descision Tree', 'Random Forest',
                              'Tensorflow']
        bots = {
                'Neris': [1, 2, 9],
                'Rbot': [3, 4, 10, 11],
                'Virut': [5, 13],
                'Menti': [6],
                'Sogou': [7],
                'Murlo': [8],
                'NSIS.ay': [12]
        }
        value_matrix = []

        for bot, num in bots.items():
            summaries = []
            values = [bot]

            for s in num:
                summaries += get_saved_data(window, binet_files[s-1])
            feature, label = get_feature_labels(summaries)
            for ml in mls:
                r = train_and_test_with(feature, label, ml)
                values.append('{0:.4f}, {1:.4f}, {2:.4f}'.format(r['accuracy'],
                              r['precision'], r['recall']))
            correctness, precision, recall = keras_train_and_test(feature,
                                                                  label)
            values.append('{0:.4f}, {1:.4f}, {2:.4f}\n'.format(correctness,
                           precision, recall))
            value_matrix.append(values)
        writer.value_matrix = value_matrix
        writer.write_table()


# Parallel(n_jobs=2)(delayed(run_files)(name, 0.15) for name in binet_files)
# Parallel(n_jobs=2)(delayed(window_shift)(i) for i in windows)
# print('For tuned down features of size 12')
# file_stats()
# bots_test()
# window_shift(.15)
kfold_test()
