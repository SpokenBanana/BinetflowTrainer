import numpy as np
import pickle
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os
from summarizer import Summarizer

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
        clf = tree.DecisionTreeClassifier()
    elif classifier == 'nb':
        clf = GaussianNB()
    elif classifier == 'rf':
        clf = RandomForestClassifier()
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


def pickle_summarized_data(interval, time, file_name, summary):
    slug_time_chars = [':', ' ', '/', '.']
    for slug in slug_time_chars:
        time = time.replace(slug, '_')
    directory = 'saved_data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    f_name = 'saved_%ss_%s_%s.pk1' % (interval, get_file_num(file_name), time)
    with open('%s%s' % (directory, f_name), 'wb') as f:
        pickle.dump(summary, f, pickle.HIGHEST_PROTOCOL)


def get_saved_data(interval, time, file_name):
    slug_time_chars = [':', ' ', '/', '.']
    for slug in slug_time_chars:
        time = time.replace(slug, '_')
    f_name = 'saved_%ss_%s_%s.pk1' % (interval, get_file_num(file_name), time)
    directory = 'saved_data/'
    pickled_data_path = '%s%s' % (directory, f_name)
    if not os.path.isfile(pickled_data_path):
        return None

    with open(pickled_data_path, 'rb') as f:
        summaries = pickle.load(f)
    return summaries


def get_binetflow_files():
    binet_files = [f for f in os.listdir(".") if f[-9:] == 'binetflow']
    binet_files = sorted(binet_files)
    files = []
    # get the files in the order they should be in
    order = [0, 1, 2, 5, 3, 8, 6, 7, 9, 11, 10, 12, 4]
    for o in order:
        files.append(binet_files[o])
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


def get_feature_labels(summaries):
    features = np.array([list(s.data.values()) for s in summaries])
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