from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


def precision_fn(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_fn(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

class Binet_Keras():

    def __init__(self):
        self.out = 1

    def train(self, feat_train, label_train, dimension=19):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=dimension, kernel_initializer='uniform',
                       activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.out, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                           metrics=['accuracy', precision_fn, recall_fn])

        self.model.fit(feat_train, label_train, epochs=10, batch_size=32,
                       verbose=False)

    def test(self, feat_test, label_test):
        return self.model.evaluate(feat_test, label_test,
                                   batch_size=32, verbose=False)


def keras_train_and_test(feat_train, label_train,
                         feat_test=None, label_test=None, dimension=19, out=1):
    if feat_test is None or label_test is None:
        feat_train, feat_test, label_train, label_test = train_test_split(
                            feat_train, label_train, test_size=0.5,
                            random_state=42)
    ke = Binet_Keras()
    ke.out = out
    ke.train(feat_train, label_train, dimension=dimension)
    loss_and_metrics = ke.test(feat_test, label_test)

    # Accuracy, precision, recall
    return loss_and_metrics[1:]


def get_tf_model(dimension=19, out=1):
    model = Sequential()
    model.add(Dense(64, input_dim=dimension, init='uniform',
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                       metrics=['accuracy'])
    return model
