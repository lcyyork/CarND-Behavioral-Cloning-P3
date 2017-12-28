import cv2, sklearn
import pandas as pd
import numpy as np
import matplotlib.image as mpimg

# ==> Data loading <== #

def load_driving_csv(data_path='data'):
    """ Load the driving_log.csv file under data_path. """

    path = '/'.join([data_path, 'driving_log.csv'])
    data = pd.read_csv(path, names=['center', 'left', 'right',
                                    'steering', 'throttle', 'brake', 'speed'],
                       header=0)
    data.apply(pd.to_numeric, errors='ignore')

    # strip white spaces in paths of center, left, right images
    for camera in ['left', 'center', 'right']:
        data[camera] = data[camera].str.strip()

    # print data summary
    print ("=> Summary of {} <=".format(path), '\n')

    print ("{:10} {:>10}".format('Name', 'Data Type'))
    for name in data:
        print ("{:10} {:>10}".format(name, str(data[name].dtypes)))

    print ()
    print ("Number of entries:", data.shape[0])
    print ("Number of images:", data.shape[0] * 3)

    print ()
    print ("Steering Description")
    print (data['steering'].describe())

    return data

def mask_small_steering(data, keep=0.25, small=0.005):
    """ Mask small steering angles in data. """

    small_mask = np.abs(data['steering']) < small
    nsmall = np.sum(small_mask)

    small_indcies = np.argwhere(small_mask).flatten()
    keep_indices = np.random.choice(small_indcies, int(nsmall * keep))

    # append "large" steering angles to the list
    keep_indices = np.append(keep_indices,
                             np.argwhere(np.logical_not(small_mask)).flatten() )

    masked_data = {}

    for name in data:
        masked_data[name] = np.array(data[name][keep_indices])

    # print data summary
    print ()
    print ("=> Data Mask Summary <=", '\n')

    print ("Number of entries:", masked_data['steering'].shape[0])
    print ("Number of images:", masked_data['steering'].shape[0] * 3)

    print ()
    print ("Steering Description")
    print (pd.DataFrame(masked_data['steering']).describe())

    return masked_data

# ==> Image processing <== #

def read_image(path):
    """ Read an image to RGB format (hight by width by 3). """

    return mpimg.imread(path)

def image_path(image_path_csv, data_path='data'):
    """ Return the image path under data_path/image_path_csv. """

    return '/'.join([data_path, image_path_csv])

def change_brightness(image, offsets=(-0.35, 0.10)):
    """ Changes the brightness of an RGB image. """

    low, high = offsets
    transformer = 1.0 + np.random.uniform(low, high)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hs = hsv[:,:,2] * transformer
    hs[hs > 255] = 255
    hsv[:,:,2] = hs
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def flip_image(image, steering_angle):
    """ Flip an image vertically. """

    return cv2.flip(image, 1), -steering_angle

def gaussian_blur(image):
    """ Apply Gaussian blur to image. """

    return cv2.GaussianBlur(image, (3,3), 0)

def resize_image(image, width=200, height=66):
    """ Resize an image. """

    return cv2.resize(image, (width, height), cv2.INTER_AREA)

def augment_image(data, data_index, camera='random', camera_corr=(0.175, 0.225)):
    """ Augment a single image by applying random flip and brightness change.

    data        -- the data map
    data_index  -- the index of the image in data
    camera      -- the camera perspective
    camera_corr -- the correction for left and right cameras

    Return: the processed image
    """

    camera_angles = ['left', 'center', 'right']
    if camera not in camera_angles:
        p = np.random.choice(3)
        camera = camera_angles[p]

    steering_angle = data['steering'][data_index]
    if camera != 'center':
        l, h = camera_corr
        random_corr = np.random.uniform(l, h)
        steering_angle += random_corr if camera == 'left' else -random_corr

    image = read_image(image_path(data[camera][data_index]))

    if np.random.rand() < 0.5:
        image, steering_angle = flip_image(image, steering_angle)

    if np.random.rand() < 0.5:
        image = change_brightness(image)

    return image, steering_angle

# ==> Generator for Keras fit_generator <== #

def generator(data, data_indices, batch_size, train_mode=True):
    """ Generator passed to Keras fit_generator.
    If steering angle > 0.5, augment all left, center, right images.
    If steering angle > 0.25, augment (50 % chance) an image from (left, center, right).
    Else, augment (75 % chance) an image from (left, center, right).
    Otherwise, use the center image.

    data        -- the data map
    data_indice -- the list of image indices in data
    batch_size  -- the size of a batch
    train_mode  -- training mode if True, validation mode if False

    Yield: training set and corresponding steering angles
    """

    ndata = len(data_indices)

    images = []
    steering_angles = []

    def append_image_angle(image, angle):
        images.append(gaussian_blur(image))
        steering_angles.append(angle)

    def default_append(i):
        image = read_image(image_path(data['center'][i]))
        append_image_angle(image, data['steering'][i])

    while True:
        count = 0

        for i in np.random.choice(data_indices, batch_size):
            steering_angle = data['steering'][i]

            if train_mode:
                if abs(steering_angle) > 0.5:
                    for camera in ['left', 'center', 'right']:
                        image, steering_angle = augment_image(data, i, camera)
                        append_image_angle(image, steering_angle)
                    count += 3
                elif abs(steering_angle) > 0.25:
                    if np.random.rand() < 0.5:
                        image, steering_angle = augment_image(data, i)
                        append_image_angle(image, steering_angle)
                    else:
                        default_append(i)
                    count += 1
                else:
                    if np.random.rand() < 0.75:
                        image, steering_angle = augment_image(data, i)
                        append_image_angle(image, steering_angle)
                    else:
                        default_append(i)
                    count += 1
            else:
                default_append(i)
                count += 1

        X = np.array(images)
        y = np.array(steering_angles)

        yield sklearn.utils.shuffle(X, y)
