# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/center.jpg
[image2]: ./examples/dataset.png "Dataset"
[image3]: ./examples/result.png "Result"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation

---

### Files Submitted & Code Quality

#### 0. Download Sample Driving Data

This example is using [sample dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity.

Download and unzip from CLI.

```sh
wget -r --tries=5 "https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip" && unzip data.zip
```

Double check all images height and width.

```sh
mediainfo '--Inform=Image; (%Height% %Width%)\n' data/IMG/*.jpg | uniq
```

as alternative I have created special repo with [custom trained datasets](https://github.com/Horki/behavioral_data).

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64.

```python
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
```

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

```python
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
```

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model contains dropout layers in order to reduce overfitting.

```py
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I choose two training data types to work with.

##### 4.a Custom

[Training data](https://github.com/Horki/behavioral_data) was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving smoothly around corners (track two).

For details about how I created the training data, see the next section.

##### 4.b Udacity

Data provided by Udacity.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Using preprocessing Lambda layer, normalized the image by dividing each element with 255 (maximum vale of image pixel).
Once the image is normalized between 0 and 1, mean centre by subtracting with 0.5 from each element which will shift the element from 0.5 to 0.
Training and validation loss are now much smaller.

```python
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
```

I have also added Cropping2D layer(to remove unnecessary part of image), and removing 50px from top and 20px from bottom, so we have less data process.

```python
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
```

<!--
The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....
-->

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

Here is a code snippet of the architecture.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving, clockwise and counter-clockwise. Here is an example image of center lane driving:

* _Clockwise_
* ![alt text][image1]
* <img src="https://raw.githubusercontent.com/Horki/behavioral_data/master/examples/test_1.gif" />

* _Counter Clockwise_
* <img src="https://raw.githubusercontent.com/Horki/behavioral_data/master/examples/test_2.gif" />

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery. These video show what a recovery looks like starting from right to left to center:

<img src="https://raw.githubusercontent.com/Horki/behavioral_data/master/examples/test_3.gif" />

Then I repeated this process on track two (one lap) in order to get more data points.

<img src="https://raw.githubusercontent.com/Horki/behavioral_data/master/examples/test_4.gif" />

To augment the data sat, I also flipped images and angles thinking that this would add effect similar to driving counter-wise, and then cropped images(removed 50x from top and 20px from bottom).
For example, here are images that has then been flipped and cropped:

![alt text][image2]

After the collection process, I had (8036 * 6) 48216 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by smallest loss value. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image3]