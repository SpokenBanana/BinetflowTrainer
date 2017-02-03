from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split


class Binet_Keras():

    def __init__(self):
        self.model = Sequential()

    def train_and_test(self, feat_train, label_train,
                       feat_test=None, label_test=None):
        if feat_test is None or label_test is None:
            feat_train, feat_test, label_train, label_test = train_test_split(
                                feat_train, label_train, test_size=0.5,
                                random_state=42)
        self.train(feat_train, label_train)
        loss_and_metrics = self.test(feat_test, label_test)

        # Accuracy, precision, recall
        return loss_and_metrics[1:]

    def train(self, feat_train, label_train):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=19, init='uniform',
                       activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                           metrics=['accuracy', 'precision', 'recall'])

        self.model.fit(feat_train, label_train, nb_epoch=10, batch_size=32,
                       verbose=False)

    def test(self, feat_test, label_test):
        return self.model.evaluate(feat_test, label_test,
                                   batch_size=32, verbose=False)
