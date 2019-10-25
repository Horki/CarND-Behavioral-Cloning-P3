# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./../examples/center.jpg

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

My model consists of a most basic convolution neural network setup.
<!--
My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).
-->

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the ~~vehicle could stay on the track~~ vehicle on this generation easily runs out of track.

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

#### 2. Final Model Architecture

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
