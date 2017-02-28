from main import train_and_test_step, aggregate_and_pickle, \
        train_and_test_with, train_with_tensorflow, test_and_train_bots
from utils import get_feature_labels, get_binetflow_files, get_start_time_for,\
        get_saved_data, mask_features, get_classifier
from summarizer import Summarizer
from sklearn.metrics import precision_recall_curve, auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pytablewriter
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from binet_keras import keras_train_and_test, BinetKeras
import numpy as np
import pickle
import random


windows = [.15]  # , 1, 2, 5]
binet_files = get_binetflow_files()


def run_files(f, window):
    print('aggregating {} for {}s'.format(f, window))
    aggregate_and_pickle(window, f)


def get_balance():
    for binet in binet_files:
        summary = get_saved_data(0.15, binet)
        _, label = get_feature_labels(summary)
        attacks = sum(label)
        nonattacks = len(label) - attacks
        print("{} | {} ".format(attacks, nonattacks))


def window_shift(window):
    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = 'Window Shift Accuracy for {}s'.format(window)
    writer.header_list = ['File', 'Descision Tree', 'Random Forest']
    value_matrix = []
    for file_name in binet_files:
        values = []
        feature, label = get_feature_labels(
                get_saved_data(window, file_name))
        feature = mask_features(feature)
        values += [
            file_name,
            '{0:.4f}'.format(train_and_test_step(feature, label, 'dt', 1000)),
            '{0:.4f}'.format(train_and_test_step(feature, label, 'rf', 1000))]
        values.append(
            '{0:.4f}'.format(train_and_test_step(feature, label, 'tf', 1000)))
        value_matrix.append(values)
    writer.value_matrix = value_matrix
    writer.write_table()


def file_stats():
    mls = ['dt', 'rf']  # , 'svm', 'nb']
    for window in windows:
        writer = pytablewriter.MarkdownTableWriter()
        writer.table_name = 'File Accuracy for {}s'.format(window)
        writer.header_list = ['File', 'Decision Tree', 'Random Forest',
                                                        'Tensorflow']
        value_matrix = []
        for name in binet_files:
            values = [name]
            feature, label = get_feature_labels(get_saved_data(window, name))
            # feature = mask_features(feature)
            feat_train, feat_test, label_train, label_test = train_test_split(
                feature, label, test_size=0.3, random_state=42)
            for ml in mls:
                r = train_and_test_with(feat_train, label_train, ml, feat_test,
                                        label_test)
                values.append('{0:.4f}, {1:.4f}, {2:.4f}'.format(
                              r['accuracy'],
                              r['precision'],
                              r['recall']))
                print(values)
            correctness, precision, recall = \
                keras_train_and_test(feat_train, label_train,
                                     feat_test, label_test, dimension=19)
            values.append('{0:.4f}, {1:.4f}, {2:.4f}'.format(correctness,
                                                               precision,
                                                               recall))
            print(values)
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
            feature = feature[:int(len(feature) * 10)]
            label = feature[:int(len(label) * 10)]
            kf = KFold(n_splits=10)

            # feature = mask_features(feature)
            for ml in mls:
                scores = []
                pr_scores = []
                for train, test in kf.split(feature):
                    clf = get_classifier(ml)
                    xtrain, ytrain = feature[train], label[train]
                    xtest, ytest = feature[test], label[test]
                    # scores = cross_val_score(clf, feature, label, cv=10, n_jobs=2)
                    clf.fit(xtrain, ytrain)
                    test_predicts = clf.predict(xtest)
                    test_score = accuracy_score(ytest, test_predicts)

                    scores.append(test_score)
                    proba = clf.predict_proba(xtest)

                    precision, recall, pr_thresholds = precision_recall_curve(
                            ytest, proba[:, 1])
                    pr_scores.append(auc(recall, precision))
                values.append('{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}'.format(
                                    np.mean(scores), np.std(scores),
                                    np.mean(pr_scores), np.std(pr_scores)))
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
    order = ['n_dports>1024', 'background_flow_count', 'n_s_a_p_address', 'avg_duration', 'n_s_b_p_address',
             'n_sports<1024', 'n_sports>1024', 'n_conn', 'n_s_na_p_address', 'n_udp', 'n_icmp', 'n_d_na_p_address',
             'n_d_a_p_address', 'n_s_c_p_address', 'n_d_c_p_address', 'normal_flow_count', 'n_dports<1024',
             'n_d_b_p_address', 'n_tcp']
    bots = {
        'Neris': [1, 2, 9],
        'Rbot': [3, 4, 10, 11],
        'Virut': [5, 13],
        'Menti': [6],
        'Sogou': [7],
        'Murlo': [8],
        'NSIS.ay': [12]
    }
    bot = ['Neris', 'Rbot', 'Virut', 'Menti', 'Sogou', 'Murlo', 'NSIS.ay']
    bot_data = [0 for _ in binet_files]
    for key, value in bots.items():
        for v in value:
            bot_data[v-1] = bot.index(key)
    for window in windows:
        writer = pytablewriter.MarkdownTableWriter()
        writer.table_name = 'Bot accuracy for {}s'.format(window)
        writer.header_list = ['Bot', 'Descision Tree', 'Random Forest',
                              'Tensorflow']
        value_matrix = []
        features = []
        labels = []
        """
        for i, b in enumerate(binet_files):
            summary = get_saved_data(window, b)
            for s in summary:
                classes = [0 for _ in range(len(bot))]
                if s.is_attack:
                    classes[bot_data[i]] = 1
                    labels.append(classes)
                    features.append([s.data[o] for o in order])
            print(len(summary), len(features))
        features = np.array(features)
        labels = np.array(labels)
        """
        with open('bot_features', 'rb') as f:
            features = pickle.load(f)
            # pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
        with open('bot_labels.pk1', 'rb') as f:
            labels = pickle.load(f)
            # pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
        values = [bot]
        for ml in mls:
            r = test_and_train_bots(features, labels, ml)
            values.append('{0:.4f}, {1:.4f}, {2:.4f}'.format(r['accuracy'],
                          r['precision'], r['recall']))
        correctness, precision, recall = keras_train_and_test(features,
                                                              labels,
                                                              out=8)
        values.append('{0:.4f}, {1:.4f}, {2:.4f}\n'.format(correctness,
                       precision, recall))
        value_matrix.append(values)
        writer.value_matrix = value_matrix
        writer.write_table()


def stats_on_best():
    best = [8, 9, 12]
    summaries = []
    for b in best:
        summaries += get_saved_data(0.15, binet_files[b])
    feature, label = get_feature_labels(summaries)
    scores = []
    for i in range(1, 5):
        feature = [[random.randrange(-(i*10), i*10) for f in feat] for feat in feature]
        acc, _, _ = keras_train_and_test(feature, label)
        scores.append(acc)
    print(scores)


# Parallel(n_jobs=2)(delayed(run_files)(name, 0.15) for name in binet_files)
# Parallel(n_jobs=2)(delayed(window_shift)(i) for i in windows)
# window_shift(0.15)
# print('For tuned down features of size 12')
# print("70/30 split")
# file_stats()
# get_balance()
stats_on_best()
# window_shift(.15)
# kfold_test()
