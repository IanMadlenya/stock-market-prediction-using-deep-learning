import os, sys
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
sys.path.append(os.path.dirname(os.getcwd()))
from tests.utils import get_trend_df


def retrieve_model():
    model = Sequential()
    model.add(LSTM(
        # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
        # Otherwise, model.evaluate() will get error.
        batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        output_dim=CELL_SIZE,
        unroll=True,
    ))
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))
    adam = Adam(LR)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # get training and testing data
    df = get_trend_df('JPM', '2017-01-11')
    features = ['tweets', 'liked', 'pos_score', 'neg_score', 'com_score']
    X = scale(df[features].values)
    y = df['trend'].values

    TIME_STEPS = 5     # same as the height of the image
    INPUT_SIZE = 1     # same as the width of the image
    BATCH_SIZE = 10
    BATCH_INDEX = 0
    OUTPUT_SIZE = 2
    CELL_SIZE = 10
    LR = 0.001

    model = retrieve_model()

    # cross validation
    scores = []
    kf = StratifiedKFold(n_splits=3, shuffle=False)
    for train_idx, test_idx in kf.split(X, y):
        print("-"*12)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train = X_train.reshape(-1, TIME_STEPS, INPUT_SIZE)
        X_test = X_test.reshape(-1, TIME_STEPS, INPUT_SIZE)
        y_train = np_utils.to_categorical(y_train, nb_classes=2)
        # training
        for step in range(10000):
            # data shape = (batch_num, steps, inputs/outputs)
            X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE]
            Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE]
            cost = model.train_on_batch(X_batch, Y_batch)
            BATCH_INDEX += BATCH_SIZE
            BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
            if step % 500 == 0:
                y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
                y_pred = np.argmax(y_pred, axis=1)
                acc = np.equal(y_pred, y_test).astype(float).mean()
                print('testing accuracy: ', acc)
        scores.append(acc)
        model = retrieve_model()
    print(['{:.3f}'.format(score) for score in scores], '{:.3f}'.format(sum(scores)/len(scores)))
