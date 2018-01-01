from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Lambda, Cropping2D, Reshape, Conv2D, Dropout, Dense, Flatten
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from utils import *

def model_NVIDIA( image_shape=(160,320,3), crop_row=(70,25), drop_prob=0.5 ):
    """ Modified NVIDIA model.

    image_shape -- the shape of input image
    crop_row    -- the number of pixels to be cropped out (top, bottom)
    drop_prob   -- the drop out probability

    Return: the model from Keras
    """

    nrow, ncol, nch = image_shape
    feed_width, feed_height = 200, 66
    target_shape = (feed_height, feed_width, nch)

    # input_shape = image_shape

    model = Sequential()

    # cropping layer
    # model.add(Cropping2D(cropping=(crop_row, (0, 0)), input_shape=image_shape))

    # lambda layer for resizing
    # def resize_lambda(x):
    #     from keras.backend import tf as ktf
    #     return ktf.image.resize_images(x, (feed_height, feed_width))
    # model.add(Lambda(resize_lambda))

    # lambda layer for normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=target_shape))

    # change color space to HSV
    # def color_lambda(x):
    #     from keras.backend import tf as ktf
    #     return ktf.image.rgb_to_hsv(x)
    # model.add(Lambda(color_lambda, input_shape=input_shape))

    # convolution layers
    model.add(Conv2D(24, 5, strides=2, padding='same',
                     activation='relu', kernel_initializer='he_uniform', name='conv1'))

    model.add(Conv2D(36, 5, strides=2, padding='same',
                     activation='relu', kernel_initializer='he_uniform', name='conv2'))

    model.add(Conv2D(48, 5, strides=2, padding='valid',
                     activation='relu', kernel_initializer='he_uniform', name='conv3'))

    model.add(Conv2D(64, 3, strides=1, padding='valid',
                     activation='relu', kernel_initializer='he_uniform', name='conv4'))

    model.add(Conv2D(64, 3, strides=1, padding='valid',
                     activation='relu', kernel_initializer='he_uniform', name='conv5'))

    model.add(Conv2D(80, 3, strides=1, padding='valid',
                     activation='relu', kernel_initializer='he_uniform', name='conv6'))

    # flatten
    model.add(Flatten())

    # apply drop out
    model.add(Dropout(drop_prob))

    # fully connected layers
    model.add(Dense(100, activation='relu', name='fc1'))

    model.add(Dense(50, activation='relu', name='fc2'))

    model.add(Dense(10, activation='relu', name='fc3'))

    # output layer
    model.add(Dense(1, name='out'))

    return model

def train_model(model, data_train, train_indices, data_valid, valid_indices,
                learning_rate=0.0002, decay_rate=0.01,
                batch_size=32, epochs=5, batches_per_epoch=2000):
    """ Train the model.

    model             -- the Keras model
    data_train        -- the data map of training data
    train_indices     -- the indices for training
    data_valid        -- the data map of validation data
    valid_indices     -- the indices for validation
    learning_rate     -- the learning_rate used in Adam optimizer
    decay_rate        -- the decay rate for learning_rate in Adam optimizer
    batch_size        -- the size of a mini-batch
    epochs            -- the number of epochs for training
    batches_per_epoch -- number of batches to yield from generator in each epoch
    """

    # Adam optimizer
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay_rate))

    # train
    history = model.fit_generator(generator(data_train, train_indices, batch_size, True),
                                  batches_per_epoch, epochs,
                                  validation_data=generator(data_valid, valid_indices,
                                                            batch_size, False),
                                  validation_steps=len(valid_indices),
                                  verbose=1, max_q_size=1)
    return history

# ==> Run the model <==

data = load_driving_csv()

nentries = len(data['steering'])
train_indices, valid_indices = train_test_split(range(nentries), test_size=0.1)

data_train = {}
data_valid = {}
for name in data:
    data_train[name] = np.array(data[name][train_indices])
    data_valid[name] = np.array(data[name][valid_indices])

steps = 1
losses = {'loss': {}, 'val_loss': {}}

for i in range(steps):
    data_masked = mask_small_steering(data_train, 0.20)
    ntrain = len(data_masked['steering'])
    nvalid = len(data_valid['steering'])

    try:
        model = load_model('model{}.h5'.format(i-1))
    except:
        model = model_NVIDIA()
    history = train_model(model, data_masked, range(ntrain), data_valid, range(nvalid),
                          epochs=8, batch_size=16, batches_per_epoch=2500,
                          learning_rate=0.001)

    # save losses
    losses['loss'][i] = history.history['loss']
    losses['val_loss'][i] = history.history['val_loss']

    # save model
    model.save('model{}.h5'.format(i))
    # model.save_weights('model.h5')

# print model summary
print (model.summary())

# save losses to file
import json
with open("loss.json", "w") as w:
    json.dump(losses, w)
