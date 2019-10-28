import csv, random, math
import numpy as np
from scipy import ndimage

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

import sklearn
import matplotlib.pyplot as plt

DATA_PATH='data'
#  DATA_PATH='behavioral_data/all'
DRIVING_LOG='driving_log.csv'

IMG_WIDTH=320
IMG_HEIGHT=160
IMG_COMPONENTS=3

BATCH_SIZE=32

def load_data(path):
    lines = []
    with open(path, "r") as f:
        # Udacity sample data has a ", " delimiter
        reader = csv.reader(f, skipinitialspace=True, delimiter=',')
        #  reader = csv.reader(f, delimiter=',')
        # Skip header for Udacity sample data
        next(reader) # Skip header
        lines = [line for line in reader]
        assert len(lines[0]) == 7
    return lines


def load_image(image_path):
    filename = image_path.split('/')[-1]
    image = ndimage.imread('{}/IMG/{}'.format(DATA_PATH, filename))
    # Check image shape only slows processing
    #  assert image.shape == (IMG_HEIGHT, IMG_WIDTH, IMG_COMPONENTS)
    return image


def load_model():
    model = Sequential()
    # Preproceesing layer
    # Normalize the image by dividing each element with 255
    #  which is maximum vale of image pixel
    # Once the image is normalized between 0 and 1,
    #  mean centre by subtracting with 0.5 from each element
    #  which will shift the element from 0.5 to 0
    # Training and validation loss are now much smaller
    print("Lambda preprocessing start...")
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_COMPONENTS)))
    print("...end preprocessing")
    # Remove 50px from top and 20px from bottom
    print("Cropping images start...")
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    print("...end cropping")
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def get_train_data_gen(samples, batch_size=32):
    correction = 0.2 # this is a parameter to tune
    samples_len = len(samples)
    while True:
        random.shuffle(samples)
        for offset in range(0, samples_len, batch_size):
            images = []
            measurements = []
            batch_samples = samples[offset:offset+batch_size]
            for batch_sample in batch_samples:
                # Feature: Center Image
                img_cen = load_image(batch_sample[0])
                img_left = load_image(batch_sample[1])
                img_right = load_image(batch_sample[2])

                img_cen_f = np.fliplr(img_cen)
                img_left_f = np.fliplr(img_left)
                img_right_f = np.fliplr(img_right)
                # Label: Steering measurement
                sterring = float(batch_sample[3])
                steering_left = sterring + correction
                steering_right = sterring - correction
                sterring_f = -sterring
                steering_left_f = -steering_left
                steering_right_f = -steering_right

                images.append(img_cen)
                measurements.append(sterring)
                images.append(img_cen_f)
                measurements.append(sterring_f)
                images.append(img_left)
                measurements.append(steering_left)
                images.append(img_left_f)
                measurements.append(steering_left_f)
                images.append(img_right)
                measurements.append(steering_right)
                images.append(img_right_f)
                measurements.append(steering_right_f)

        features = np.array(images)
        labels = np.array(measurements)
        yield sklearn.utils.shuffle(features, labels)


if __name__ == "__main__":
    print("Load driving log. start...")
    # Indexes
    # center[0], left[1], right[2], steering[3], throttle[4], brake[5], speed[6]
    samples = load_data("{}/{}".format(DATA_PATH, DRIVING_LOG))
    train_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    print("...done\nLoad train data: start...")
    train_generator = get_train_data_gen(train_samples, batch_size=BATCH_SIZE)
    validation_generator = get_train_data_gen(validation_samples, batch_size=BATCH_SIZE)
    print("...done\nCompile model: start...")

    # Model Part
    model = load_model()

    history_object = model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(len(train_samples) / BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=math.ceil(len(validation_samples) / BATCH_SIZE),
        epochs=3,
        verbose=1
    )
    model.save('model.h5')
    print("...done\nSave model")

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("examples/result.png")
    plt.show()