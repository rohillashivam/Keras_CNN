import sys
import keras
import json
from pathlib import Path
from keras import Sequential
from keras.activations import softmax
from keras.datasets import cifar10

# global static variables
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam

dtype_mult = 255.0 # unit8
num_classes = 10
X_shape = (-1, 32, 32, 3)
epoch = 200
batch_size = 128

def retrieveDataset():
    sys.stdout.write('Loading Dataset\n')
    sys.stdout.flush()

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    return X_train, y_train, X_test, y_test

def processDataSet(X_train, Y_train, X_test, Y_test):
    sys.stdout.write('Preprocessing Dataset\n\n')
    sys.stdout.flush()

    X_train = X_train.astype('float32') / dtype_mult
    X_test = X_test.astype('float32') / dtype_mult
    y_train = keras.utils.to_categorical(Y_train, num_classes)
    y_test = keras.utils.to_categorical(Y_test, num_classes)

    return X_train, y_train, X_test, y_test

def generate_model():

    if Path('./models/convnet_model.json').is_file():
        sys.stdout.write('Loading existing model\n\n')
        sys.stdout.flush()

        with open('./models/convnet_model.json') as file:
            model = keras.models.model_from_json(json.load(file))
            file.close()

        # likewise for model weight, if exists load from saved state
        if Path('./models/convnet_weights_1.h5').is_file():
            model.load_weights('./models/convnet_weights_1.h5')

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model

    model = Sequential()

    ## Convolution-1 -- 32 * 32 * 3 => 30 * 30 (32)
    model.add(Conv2D(32, (3, 3), input_shape=X_shape[1:]))
    model.add(LeakyReLU(alpha=0.01))

    ## Convolution-2 -- 30 * 30 (32) => 28 * 28 (32)
    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=0.01))

    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(0.25))

    ## Convolution-3 -- 14 * 14 (32) => 12 * 12 (64)
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=0.01))

    ## Convolution-4 -- 12*12 (64) => 6 6 (64)
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=0.01))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # FC layers 3 3 (64) => 576
    model.add(Flatten())
    # Dense1 576 => 256
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))

    #Dense2 256 => 10
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    with open('./models/convnet_model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
        outfile.close()

    return model

def train(model, X_train, y_train, X_test, y_test):
    epoch_count = 0
    while epoch_count < epoch:
        epoch_count +=1
        print('Epoch count: {}'.format(epoch_count), end='\n')
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,
                  validation_data=(X_test, y_test))
        print('Epoch :: {} done, saving model to file \n\n'.format(epoch_count))
        model.save_weights('./models/convnet_weights_1.h5')

    return model

def main():
    X_train, y_train, X_test, y_test = retrieveDataset()
    X_train, y_train, X_test, y_test = processDataSet(X_train, y_train, X_test, y_test)
    model = generate_model()
    model = train(model, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()