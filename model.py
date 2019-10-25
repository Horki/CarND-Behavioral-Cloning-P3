import csv, cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense

DATA_PATH='data'
#  DATA_PATH='behavioral_data/all'
DRIVING_LOG='driving_log.csv'

IMG_WIDTH=320
IMG_HEIGHT=160
IMG_COMPONENTS=3

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
    image = cv2.imread('{}/IMG/{}'.format(DATA_PATH, filename))
    # Check image shape only slows processing
    #  assert image.shape == (IMG_HEIGHT, IMG_WIDTH, IMG_COMPONENTS)
    return image


# Most basic neural network
def load_model():
    model = Sequential()
    model.add(Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_COMPONENTS)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def get_train_data(samples):
    images = []
    measurements = []
    for line in samples:
        # Feature: Center Image
        images.append(load_image(line[0]))
        # Label: Steering measurement
        measurements.append(float(line[3]))
    features = np.array(images)
    labels = np.array(measurements)
    return features, labels


if __name__ == "__main__":
    print("Load driving log. start...")
    # Indexes
    # center[0], left[1], right[2], steering[3], throttle[4], brake[5], speed[6]
    samples = load_data("{}/{}".format(DATA_PATH, DRIVING_LOG))
    print("...done\nLoad train data: start...")
    X_train, y_train = get_train_data(samples)
    print("...done\nCompile model: start...")
    # Model Part
    model = load_model()
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
    print("...done\nSave model")
    model.save('model.h5')
