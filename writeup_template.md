# **Traffic Sign Recognition** 

### Writeup / README - [rubric points](https://review.udacity.com/#!/rubrics/481/view) 

You're reading it! and here is a link to my [project code](https://view5f1639b6.udacity-student-workspaces.com/notebooks/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier-Final.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 104397 (3 times the size of original data set)
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

> Histogram with a breakdown of number of individual classes can be found in code cell #6


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

Initiall I tried to convert the images in grayscale on the fly, but I ran into performance issues. So, I decided to create a untility method that will go through all the images in training set and create a grayscale version of the images and create a pickle file that I can use to re-load augmented data. 

I also augmented the data set by also create a normalized version of the images. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             | 
| CONVOLUTION 3x3       | 1x1 stride, valid padding, outputs 30x30x4    |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| CONVOLUTION 4x4       | 1x1 stride, valid padding, outputs 27x27x16   |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| CONVOLUTION 4x4       | 1x1 stride, valid padding, outputs 24x24x32   |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| MAXPOOLING            | 2x2 stride, valid padding, outputs 12x12x32   |
| DROPOUT               | 15 %                                          |
| CONVOLUTION 3x3       | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| CONVOLUTION 3x3       | 1x1 stride, valid padding, outputs 8x8x128    |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| MAXPOOLING            | 2x2 stride, valid padding, outputs 4x4x128    |
| DROPOUT               | 15 %                                          |
| FLATTEN               | Input 4x4x128, output  2048                   |
| FULLY CONNECTED       | Input 2048, output  1024                      |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| FULLY CONNECTED       | Input 1024, output  512                       |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| FULLY CONNECTED       | Input 512, output  256                        |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| FULLY CONNECTED       | Input 256, output  128                        |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| FULLY CONNECTED       | Input 128, output  84                         |
| RELU                  |                                               |
| DROPOUT               | 15 %                                          |
| FULLY CONNECTED       | Input 84, output  43                          |
| SOFTMAX               | etc.                                          |
|-----------------------------------------------------------------------| 
 
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used 75 epochs to with a batch size of 500 to train the model. I also used a learning rate of .001 and dropout rate of 15%.

This information can be found in code cell 10.

* I went through the training data set images and created a normalized version and greyscale version.
* I also created new pickle files that contains the new images. 
* Pickle files allow me to quickly load additional images for training. These files can be found at '/home/workspace/data/train_gray.p' and '/home/workspace/data/train_normalized.p'

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

* What was the first architecture that was tried and why was it chosen?

> I started out with LeNet architecture but the accuracy was very low even after augmenting data with greyscale and normalized images.

* What were some problems with the initial architecture?
> Accuracy of the model was very low even after augmenting data with greyscale and normalized images.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

> I augmented the data with greyscale and normalized images. 
>
> To increase accuracy, I added 3 additional convolution layers and 4 additional layer to the fully-connected layer. I also added 
dropout of 15% layer to select convolution and fully-connected layers.

* Which parameters were tuned? How were they adjusted and why?

> Added additional convolution and fully connected layers along with random dropout of 15%.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Test images can be found in code cell # 15.

Most of the images are very straight forward and model didn't have any difficuly classifying the images. 

* Model was unsuccesful classifying image that had show in the foreground.
* It also had difficuly classifying image of double crose for some reason. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

> Model correctly predicted 6 out of 8 new images, with an accuracy of 75% on a very small sample set.
> 
> Results of the prediction can be found in code cell # 48:

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

> Results of the prediction can be found in code cell # 48:


