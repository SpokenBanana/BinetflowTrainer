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

TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


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


def classify(ip):
    parts = ip.split('.')
    try:
        first = int(parts[0])
    except Exception:
        return 'na'

    # TODO: write a better way to classify this.
    if 1 <= first <= 126:
        return 'a'
    elif 128 <= first <= 191:
        return 'b'
    elif 192 <= first <= 223:
        return 'c'
    return 'na'


def train_and_test_with(summaries, classifier):
    """
        clf: the str rep machine learning algorithm being used
    """
    if classifier == 'svm':
        clf = svm.SVC()
    elif classifier == 'dt':
        clf = tree.DecisionTreeClassifier()
    elif classifier == 'nb':
        clf = GaussianNB()
    elif classifier == 'rf':
        clf = RandomForestClassifier()
    else:
        print('classifier not valid')
        return {}

    features = np.array([s.data.values() for s in summaries])
    labels = np.array([s.is_attack for s in summaries])

    feat_train, feat_test, label_train, label_test = train_test_split(
        features, labels, test_size=0.5, random_state=42)

    clf.fit(feat_train, label_train)
    result = {'score': clf.score(feat_test, label_test)}

    predicted_labels = clf.predict(feat_test)

    result['recall'] = recall_score(label_test, predicted_labels)
    result['accuracy'] = accuracy_score(label_test, predicted_labels)
    result['precision'] = precision_score(label_test, predicted_labels)
    result['f1 score'] = f1_score(label_test, predicted_labels)
    result['support'] = sum(labels)
    result['normal count'] = len(labels) - result['support']
    result['training size'] = len(feat_train)

    return result


def pickle_summarized_data(interval, time, file_name, summary=None):
    slug_time_chars = [':', ' ', '/', '.']
    for slug in slug_time_chars:
        time = time.replace(slug, '_')
    directory = 'saved_data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    f_name = 'saved_%ss_%s_%s.pk1' % (interval, get_file_num(file_name), time)
    with open('%s%s' % (directory, f_name), 'wb') as f:
        if summary is None:
            start = datetime.strptime(start_time, TIME_FORMAT)
            summary = review_data(interval, start, file_name)
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


def review_data(interval, start, file_name):
    """ Aggregate the data within the windows of time

        interval:       time in seconds to aggregate data
        start:          start time to record data
        file_name:      which file to record

        returns: score of the 50-50 test train and labels
                 of the data recorded.
    """
    summaries = [Summarizer() for _ in xrange(10)]
    with open(file_name, 'r+') as data:
        headers = data.readline().strip().lower().split(',')
        for line in data:
            args = line.strip().split(',')
            time = datetime.strptime(args[0], TIME_FORMAT)
            window = int((time - start).total_seconds() / interval)
            if window < 0:
                continue
            if window >= len(summaries):
                for i in xrange(window + 1):
                    summaries.append(Summarizer())
            item = dict(zip(headers, args))
            summaries[window].add(item)
    return [s for s in summaries if s.used]


def save_results(destination_path, file_name, start_time, interval, args,
                 print_contents=False):
    with open(destination_path, 'w+') as out:
        result = 'on file %s\n' % file_name
        result += 'start time = %s\n' % start_time
        result += 'window size = %ds\n' % interval
        for key, value in sorted(args.iteritems()):
            result += '%s = %s\n' % (key, value)
        if print_contents:
            print result
        out.write(result)


def get_file_num(file_name):
    """ Get the ending file number in the files, these are all that is
        really needed to distinguish between files at a glance
    """
    base = file_name.split('.')[0]
    dash_split = base.split('-')

    if len(dash_split) == 1:
        return dash_split[0][-2:]
    return '%s-%s' % (dash_split[0][-2:], dash_split[1])


def run_analysis_with(interval, start_time, file_name, use_pickle=False):
    start = datetime.strptime(start_time, TIME_FORMAT)
    file_num = get_file_num(file_name)
    directory = 'run_of_%s_%s/' % (file_num, start_time.split(' ')[1][:2])

    if not os.path.exists(directory):
        os.makedirs(directory)

    mls = ['dt', 'rf', 'nb', 'svm']

    if use_pickle:
        print 'loading pickle'
        summaries = get_saved_data(interval, start_time, file_name)
        if summaries is None:
            print 'failed to load pickle. Aggregating data'
            summaries = review_data(interval, start, file_name)
            print 'finished aggregating, pickling data...'
            pickle_summarized_data(interval, start_time, file_name,
                                   summary=summaries)
            print 'data pickled'
        else:
            print 'loaded picke'
    else:
        print 'aggregating data'
        summaries = review_data(interval, start, file_name)
        print 'finished aggregating, pickling data...'
        pickle_summarized_data(interval, start_time, file_name,
                               summary=summaries)
        print 'data pickled'

    for ml in mls:
        print 'testing with %s' % ml
        result = train_and_test_with(summaries, ml)
        path = '%srun_%ds_%s_%s.txt' % (directory, interval, file_num, ml)
        save_results(path, file_name, start_time, interval, result)


if __name__ == '__main__':
    start_time = '2011/08/16 01:08:00.0'
    interval = 1  # in seconds
    file_name = 'capture20110815-3.binetflow'

    run_analysis_with(interval, start_time, file_name, use_pickle=True)
