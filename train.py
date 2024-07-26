import os
import random as rn

from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from skimage import io, transform
import tensorflow as tf
from sklearn.metrics import classification_report

from utils import print_baseline


def seed_randomness():


    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)
    
    if tf.__version__.startswith('2.'):
        # TensorFlow 2.x için uygun konfigürasyon
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    else:
        # TensorFlow 1.x için eski API kullanımı
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        k.set_session(sess)


def load_data_with_dev():

    # Load in picture paths and labels
    root = 'dataset/'
    train_x = np.loadtxt(root + 'train-x', dtype=str)
    train_y = np.loadtxt(root + 'train-y')

    # Shuffle indices
    idx = list(range(len(train_x)))
    np.random.shuffle(idx)

    # Create development set indices
    devidx = {-1}
    for i in range(int(len(train_x) / 10)):
        ridx = -1
        while ridx in devidx:
            ridx = np.random.randint(len(train_x))
        devidx.add(ridx)
    devidx.remove(-1)

    # Load in picture data into training and development sets
    tempx = []
    tempy = []

    tempdevx = []
    tempdevy = []

    class_weight_temp = {0:  0., 1: 0.}

    for i, xidx in enumerate(idx, 0):
        img_name = root + 'train/' + train_x[xidx]
        image = transform.resize(io.imread(img_name), (300, 300))
        if i in devidx:
            tempdevx.append(image)
            tempdevy.append(train_y[xidx])
        else:
            tempx.append(image)
            tempy.append(train_y[xidx])
            class_weight_temp[train_y[xidx]] += 1.0

    class_weight = {0: class_weight_temp[1], 1: class_weight_temp[0]}

    trainx = np.array(tempx)
    trainy = np.array(tempy)

    devx = np.array(tempdevx)
    devy = np.array(tempdevy)

    return trainx, trainy, devx, devy, class_weight


def construct_model():

    model = Sequential()

    model.add(Conv2D(8, 9, input_shape=(300, 300, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(20, 9, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 9, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def main():
    seed_randomness()

    trainx, trainy, devx, devy, class_weight = load_data_with_dev()
    model = construct_model()

    # Model eğitimi ve doğrulama seti üzerinde değerlendirme
    history = model.fit(trainx, trainy, epochs=15, batch_size=32, class_weight=class_weight, validation_data=(devx, devy))

    # Modelin doğrulama seti üzerindeki performansını değerlendirme
    val_loss, val_accuracy = model.evaluate(devx, devy, batch_size=32)
    print('Finished Training')
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}')

    # Tahminleri al ve sınıflandırma raporunu yazdır
    dev_predictions = model.predict(devx)
    dev_predictions_binary = (dev_predictions > 0.5).astype(int)
    report = classification_report(devy, dev_predictions_binary, target_names=['Not Van Gogh', 'Van Gogh'])
    print(report)

    model.save('model.ker')
    print_baseline(devy)



if __name__ == '__main__':
    main()



