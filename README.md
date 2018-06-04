# **Traffic Sign classifier using Deep Learning and CNN** 

**Build a Traffic Sign Recognition Project**

In this project, I am applying what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I am training and validating a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I tried it out on images of German traffic signs that were found on the web.


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture (LeNet)
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results 

[//]: # (Image References)

[image1a]: ./TrainingHist.png "Training Data Freq. Distribution"
[image1b]: ./ValidationHist.png "Validation Data Freq. Distribution"
[image1c]: ./TestHist.png "Test Data Freq. Distribution"
[image2]: ./BeforeVsAfter.png "Preprocessing"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./germantrafficsigns/image1.jpg "Traffic Sign 1"
[image5]: ./germantrafficsigns/image2.jpg "Traffic Sign 2"
[image6]: ./germantrafficsigns/image3.jpg "Traffic Sign 3"
[image7]: ./germantrafficsigns/image4.jpg "Traffic Sign 4"
[image8]: ./germantrafficsigns/image5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mdeopujari/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread out with classes on x-axis and the frequency count on y-axis.

In each of the datasets, there are too many samples for only certain classes. This may lead to a biased network that would lean towards these few classes while classifiying test images. 

![Training Data Freq. Distribution][image1a]
![Validation Data Freq. Distribution][image1b]
![Test Data Freq. Distribution][image1c]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because of the anticipated gains in training accuracy as seen in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf "Traffic Sign Recognition with Multi-Scale Convolutional Networks"). This strategy is helpful since there is little structural ovelap between different categories of images.

Also, I used histogram equalization for generating better contrast in the image which helps make the edges stand out.

As a last step, I normalized the image data because this helps to make the training network less sensitive to the variations in scale of features. It also helps the training algorithm to converge faster.

Here is an example of an original image before and after preprocessing:

![Preprocessing][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale Normalized image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten Layer					|	outputs	400										|
| Fully connected		| outputs 120       									|
| RELU					|												|
| Fully connected		| outputs 84       									|
| RELU					|												|
| Fully connected		| outputs 43       									|
| Softmax				| Final output as probabilities        									|
|						|												|

 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer to minimize the loss function which was cross entropy with logits. This type of optimizer was chosen as it uses moving averages of the parameters (momentum) and this enables Adam to use a larger effective step size without fine tuning.

The batch size was kept at 128 with 50 epochs and a learning rate of 0.0005.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of ~95% 
* test set accuracy of ~92%

 There are certain similarities in handwriting recognition and traffic sign classification. The label set is limited and known and also there is a fair amount of variability amongst images with the same label. Hence, well known LeNet architecture, which was orginally invented for handwriting recognition, was chosen for this task due to the similarities mentioned above.

The model appears to be overfitting because by the end of the training epochs, training accuracy reaches 100% followed by validation accuracy of ~95% while the test accuracy is ~92%.
This can be remedied by using dropout layers in training and by augmenting the training dataset with additional images that are same as original images but with minor translation/rotation/random noise applied. Augmentation of data should be such that the dataset becomes more balanced across all categories. All these techniques make the training network more robust to variations in angle, orientation, lighting etc. of the samples in the training dataset. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because of a non-uniform background (trees) and slight slant.

The second image should not be too difficult to classify because of the glare. Preprocessing the image would take care of that.

The third image should not be too difficult to classify because it is well lit and at the correct angle. However, pattern of the bridge rails in the background might affect slightly.

The fourth image catergory is very rare in the dataset and may cause problems in classification. Also, edge detection in this sample may be hard in grayscale due to a very uniform background.

The fifth image should not be too difficult to classify because it is well lit and at the correct angle. However, pattern of clouds in the background might affect slightly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road Work    			| Road Work										|
| 30 km/h					| 30 km/h											|
| Turn Right ahead	      		| Ahead	only				 				|
| Right-of-way at next intersection			| Right-of-way at next intersection     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ~92%. The only time it had trouble identifying correctly was for a rare category of image.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

For the first image, the model is 100% sure that this is a stop sign (probability of ~1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.0         			| Stop sign   									| 
| .95e-8     				| Turn right ahead 										|
| .46e-12					| Keep right											|
| .17e-12	      			| Turn left ahead					 				|
| .12e-12				    | Yield     							|


For the second image, the model is 97% sure that this is a road work sign and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Road work   									| 
| .23e-1     				| Double curve 										|
| .25e-6					| Dangerous curve to the left											|
| .21e-6	      			| Turn right ahead					 				|
| .09e-7				    | Beware of ice/snow    							|

For the third image, the model is 99% sure that this is a Speed limit (30km/h) and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (30km/h)							| 
| .12e-6     				| Speed limit (20km/h)										|
| .18e-7					| Speed limit (80km/h)											|
| .13e-8	      			| Speed limit (50km/h)					 				|
| .14e-9				    | General caution    							|

For the fourth image, the model is 99% sure that this is a Ahead only sign but the image does not contain a Ahead only sign. It is instead a Turn right ahead sign which is not amongst the top five soft max probabilities:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Ahead only							| 
| .69e-2     				| Speed limit (80km/h)										|
| .87e-4					| Speed limit (60km/h)											|
| .81e-4	      			| Right-of-way at the next intersection					 				|
| .17e-5				    | Turn left ahead    							|

For the fifth image, the model is 99% sure that this is a Ahead only sign but the image does not contain a Ahead only sign. It is instead a Turn right ahead sign which is not amongst the top five soft max probabilities:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Right-of-way at the next intersection					| 
| .18e-1     				| Beware of ice/snow										|
| .16e-7					| Road narrows on the right											|
| .74e-10	      			| Road work					 				|
| .17e-5				    | Speed limit (50km/h)    							|
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


