from sklearn import svm
import sys
import numpy as np
from sklearn.model_selection import train_test_split
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

    def add(self, item):
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


def train_and_test_with_svm(summaries):
    features = np.array([s.data.values() for s in summaries])
    labels = np.array([s.is_attack for s in summaries])

    feat_train, feat_test, label_train, label_test = train_test_split(
        features, labels, test_size=0.5, random_state=42)

    clf = svm.SVC()
    clf.fit(feat_train, label_train)

    return clf.score(feat_test, label_test), labels


def review_data(interval, start, max_windows, file_name):
    """ Aggregate the data within the windows of time and
        train them using SVM to detect whether a network is
        under attack.

        interval:       time in seconds to aggregate data
        start:          start time to record data
        max_windows:    amount of windows to record
        file_name:      which file to record

        returns: score of the 50-50 test train and labels
                 of the data recorded.
    """
    summaries = [Summarizer() for _ in xrange(max_windows + 1)]
    with open(file_name, 'r+') as data:
        headers = data.readline().strip().lower().split(',')
        for line in data:
            args = line.strip().split(',')
            time = datetime.strptime(args[0], TIME_FORMAT)
            window = int((time - start).total_seconds() / interval)
            if 0 <= window <= max_windows:
                item = dict(zip(headers, args))
                summaries[window].add(item)

    return train_and_test_with_svm(summaries)


if __name__ == '__main__':
    start_time = '2011/08/15 17:10:20.0'
    interval = 20
    total_windows = 60
    file_name = 'capture20110815-3.binetflow'
    start = datetime.strptime(start_time, TIME_FORMAT)

    score, labels = review_data(interval, start, total_windows, file_name)

    f_name = 'out.txt'
    if len(sys.argv) > 1:
        f_name = '%s.txt' % sys.argv[1]

    with open(f_name, 'w+') as out:
        result = 'on file %s\n' % file_name
        result += 'start time = %s\n' % start_time
        result += 'window size = %ds\n' % interval
        result += 'labels = %s\n' % str(labels)
        result += 'training size = %d\n' % total_windows
        result += 'score is %f%%\n' % score
        print result
        out.write(result)
