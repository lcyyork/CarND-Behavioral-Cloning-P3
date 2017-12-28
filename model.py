from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Cropping2D, Conv2D, Dropout, Dense, Flatten
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from utils import *

def model_NVIDIA( image_shape=(160,320,3), crop_row=(65,25), drop_prob=0.5 ):
    """ Modified NVIDIA model.

    image_shape -- the shape of input image
    crop_row    -- the number of pixels to be cropped out (top, bottom)
    drop_prob   -- the drop out probability

    Return: the model from Keras
    """

    nrow, ncol, nch = image_shape
    feed_width, feed_height = 200, 66
    input_shape = (feed_height, feed_width, nch)

    model = Sequential()

    # cropping layer
    model.add(Cropping2D(cropping=(crop_row, (0, 0)), input_shape=image_shape))

    # lambda layer for resizing
    # def resize_lambda(x):
    #     from keras.backend import tf as ktf
    #     return ktf.image.resize_images(x, (feed_height, feed_width))
    # model.add(Lambda(resize_lambda))

    # lambda layer for normalization
    model.add(Lambda(lambda x: x / 255.0, input_shape=input_shape))

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

def train_model(model, data, train_indices, valid_indices,
                learning_rate=0.001, decay_rate=0.01,
                batch_size=32, epochs=5, batches_per_epoch=2000,
                validation_steps=1000):
    """ Train the model.

    model -- the Keras model
    data -- the data map
    train_indices     -- the indices for training
    valid_indices     -- the indices for validation
    learning_rate     -- the learning_rate used in Adam optimizer
    decay_rate        -- the decay rate for learning_rate in Adam optimizer
    batch_size        -- the size of a mini-batch
    epochs            -- the number of epochs for training
    batches_per_epoch -- number of batches to yield from generator in each epoch
    validation_steps  -- number of batches to yield from generator
    """

    # Adam optimizer
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay_rate))

    # train
    model.fit_generator(generator(data, train_indices, batch_size, True),
                        batches_per_epoch, epochs,
                        validation_data=generator(data, valid_indices, batch_size, False),
                        validation_steps=validation_steps, verbose=1)
    return

# ==> Run the model <==

data = load_driving_csv()
data = mask_small_steering(data)

nentries = len(data['steering'])
train_indices, valid_indices = train_test_split(range(nentries), test_size=0.2)

model = model_NVIDIA()
train_model(model, data, train_indices, valid_indices, epochs=1,
            batch_size=16, batches_per_epoch=8, validation_steps=16)
# model.save_weights('model.h5')

print (model.summary())
# plot_model(model, to_file='model.png')

model.save('model.h5')
