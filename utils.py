import numpy as np
import pickle
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os
from summarizer import Summarizer
import matplotlib as plt
from itertools import cycle

TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


def save_results(destination_path, file_name, start_time, interval, args,
                 print_contents=False):
    with open(destination_path, 'w+') as out:
        result = 'on file %s\n' % file_name
        result += 'start time = %s\n' % start_time
        result += 'window size = %ds\n' % interval
        for key, value in sorted(args.items()):
            result += '%s = %s\n' % (key, value)
        if print_contents:
            print(result)
        out.write(result)


def get_classifier(classifier):
    if classifier == 'svm':
        clf = svm.SVC()
    elif classifier == 'dt':
        # clf = tree.DecisionTreeClassifier()
        clf = tree.DecisionTreeClassifier(class_weight='balanced',
                                          criterion='entropy', splitter='best')
    elif classifier == 'nb':
        clf = GaussianNB()
    elif classifier == 'rf':
        # clf = RandomForestClassifier()
        clf = RandomForestClassifier(n_estimators=100,
                                     max_features=None,
                                     criterion='entropy',
                                     n_jobs=2)
    else:
        print("No classifier %s" % classifier)
        return None
    return clf


def get_file_num(file_name):
    """ Get the ending file number in the files, these are all that is
        really needed to distinguish between files at a glance
    """
    base = file_name.split('.')[0]
    dash_split = base.split('-')

    if len(dash_split) == 1:
        return dash_split[0][-2:]
    return '%s-%s' % (dash_split[0][-2:], dash_split[1])


def pickle_summarized_data(interval, file_name, summary, v2=False):
    directory = 'saved_v2/' if v2 else 'saved_data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    f_name = 'saved_%ss_%s.pk1' % (interval, get_file_num(file_name))
    with open('%s%s' % (directory, f_name), 'wb') as f:
        pickle.dump(summary, f, pickle.HIGHEST_PROTOCOL)


def get_saved_data(interval, file_name, v2=False):
    directory = 'saved_v2/' if v2 else 'saved_data/'
    f_name = 'saved_%ss_%s.pk1' % (interval, get_file_num(file_name))
    pickled_data_path = '%s%s' % (directory, f_name)
    if not os.path.isfile(pickled_data_path):
        return None

    with open(pickled_data_path, 'rb') as f:
        summaries = pickle.load(f)
    return summaries


def get_binetflow_files():
    files = []
    with open('binet_files.txt', 'r+') as f:
        files = f.readline().strip().split(',')
    return files


def get_files_with_bot(bot):
    if bot == 'Neris':
        return [1, 2, 9]
    if bot == 'Rbot':
        return [3, 4, 10, 11]
    if bot == 'Virut':
        return [5, 8]
    if bot == 'Menti':
        return [6]
    if bot == 'Sogou':
        return [7]
    if bot == 'Murlo':
        return [8]
    if bot == 'NSIS.ay':
        return [12]
    return []


def get_feature_order():
    return ['n_dports>1024',
            'background_flow_count', 'n_s_a_p_address', 'avg_duration',
            'n_s_b_p_address', 'n_sports<1024', 'n_sports>1024', 'n_conn',
            'n_s_na_p_address', 'n_udp', 'n_icmp', 'n_d_na_p_address',
            'n_d_a_p_address', 'n_s_c_p_address', 'n_d_c_p_address',
            'normal_flow_count', 'n_dports<1024', 'n_d_b_p_address', 'n_tcp']


def get_v2_order():
    # 22 features
    return ['n_dports>1024',
            'background_flow_count', 'n_s_a_p_address', 'avg_duration',
            'n_s_b_p_address', 'n_sports<1024', 'n_sports>1024', 'n_conn',
            'n_s_na_p_address', 'n_udp', 'n_icmp', 'n_d_na_p_address',
            'n_d_a_p_address', 'n_s_c_p_address', 'n_d_c_p_address',
            'normal_flow_count', 'n_dports<1024', 'n_d_b_p_address', 'n_tcp',
            'end_tcp', 'end_udp', 'end_icmp']


def get_feature_labels(summaries, v2=False):
    order = get_v2_order() if v2 else get_feature_order()
    features = np.array([[s.data[o] for o in order] for s in summaries])
    labels = np.array([s.is_attack for s in summaries])
    return features, labels


def to_tf_label(labels):
    tf_labels = []
    for label in labels:
        if label == 1:
            tf_labels.append([0, 1])
        else:
            tf_labels.append([1, 0])
    return tf_labels


def get_start_time_for(file_name):
    time = ''
    with open(file_name, 'r+') as f:
        f.readline()
        time = f.readline().strip().split(',')[0]
    return time


def mask_features(features):
    # computed from grid search
    feature_mask = [True,  True, True, True, True, True, True, True,
                    False, True, True, False, True, True, True, True, True,
                    True,  True]
    other_features = []
    for i in range(len(features)):
        good_features = []
        for j in range(len(features[i])):
            if feature_mask[j]:
                good_features.append(features[i][j])
        other_features.append(good_features)
    return np.array(other_features)


def plot_roc_curve(fpr, tpr, roc_auc, title='ROC curve'):
    plt.figure()
    lw = 2
    colors = cycle(['darkorange', 'blue', 'red', 'green', 'black', 'purple',
                   'yellow'])
    for i, color in zip(range(len(fpr), colors)):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC area {0:.4f}'.format(roc_auc[i]))
    # plt.plot(fpr, tpr, color='darkorange', lw=lw,
    #          label='ROC area under curve = {}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.show()


