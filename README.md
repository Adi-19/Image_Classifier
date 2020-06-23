# Deep Learning
# Project: Image Classifier
Source
# Project 2 from Udacity's Intro to Machine Learning Nanodegree

# Description
Developed an image classifier with Tenorflow, then converted it into a command line application.

1. Loaded training data, validation data, testing data, label mappings, and applied transformations (random scaling, cropping, resizing, flipping) to training data
2. Normalized means and standard deviations of all image color channels, shuffled data and specified batch sizes
3. Loaded pre-trained VGG16 network
4. Defined a new untrained feed-forward network as a classifier, using ReLU activations, and Dropout
5. Defined Negative Log-Likelihood Loss, Adam Optimizer, and learning rate
6. Trained the classifier layers with backpropagation in a CUDA GPU using the pre-trained network to ~90% accuracy on the validation set
7. Graphed training/validation/testing loss and validation/testing accuracy to ensure convergence to a global (or sufficient local) minimum
8. Saved and loaded model to perform inference later
9. Preprocessed images (resize, crop, normalized means and standard deviations) to use as input for model testing
10. Visually displayed images to ensure preprocessing was successful
11. Predicted the class/label of an image using the trained model and plotted top 5 classes to ensure the validity of the prediction
# Install
This project requires Python 3.x and the following Python libraries installed:

NumPy
Pandas
Matplotlib
Seaborn
Tensorflow
You will also need to have software installed to run and execute an iPython Notebook

It's recommended to install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.
