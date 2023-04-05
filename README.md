
# Beverage Image Classification (Alcoholic vs. Non-Alcoholic)
This repository consists of a project that classifies images of alcoholic and non-alcoholic beverages using deep learning. The project comprises two Jupyter notebooks: 'data downloader.ipynb' and 'model training.ipynb'.

## Introduction
This project focuses on classifying alcoholic and non-alcoholic beverage images using deep learning techniques. The goal is to build a machine-learning model that can accurately identify and classify pictures of beverages as alcoholic or non-alcoholic based on the visual characteristics of the images.

The project consists of two main parts: data collection and preparation and model training. The first part involves downloading and preprocessing the data, which includes a large dataset of beverage images scraped from various sources online. The data downloader script (data downloader.ipynb) downloads the images and organizes them into separate training and testing directories. The second part involves building and training the machine learning model using a convolutional neural network (CNN) architecture. The model training script (model training.ipynb) builds the CNN model, preprocesses the data using image augmentation techniques, trains the model on the training data, and evaluates its performance on the testing data.

The project leverages various machine learning and deep learning libraries, such as TensorFlow, Keras, and scikit-learn, along with other Python libraries for data manipulation and visualization. The trained model can be used to classify new images of beverages as alcoholic or non-alcoholic and can be further improved by fine-tuning on more data or modifying the model architecture.

## Transfer Learning
This project used transfer learning to build a deep-learning model for classifying images of alcoholic and non-alcoholic beverages. Transfer learning is a technique in deep learning where a pre-trained model is used as a starting point for a new model instead of training a model from scratch. The pre-trained model has already learned how to recognize features in images and can be used to extract features from new photos.

We used the MobileNet architecture as our pre-trained model. MobileNet is a lightweight neural network architecture designed for mobile and embedded devices. We used the pre-trained weights from the MobileNet model trained on the ImageNet dataset as a starting point for our model. We added a new fully connected layer at the end of the pre-trained model and trained this layer to classify our images as alcoholic or non-alcoholic. We used the transfer learning technique to take advantage of the pre-trained model's knowledge of features in images and speed up the training process of our model.

## Code Sections (IPYNB Files)
### Data Downloader
The data_downloader.ipynb notebook is responsible for downloading and organizing a dataset of images of alcoholic and non-alcoholic beverages. It uses the Open Images Dataset API to search for and download images that match specific keywords.

The notebook starts by importing the necessary libraries, including the openimages library, which provides a Python API for accessing the Open Images Dataset. It then sets up the API key for the Open Images Dataset API, which is required to make requests to the API.

Next, the notebook defines a list of keywords related to alcoholic and non-alcoholic beverages, which will be used to search for images in the dataset. It then sets up a loop that searches for images related to each keyword and downloads them to the local directory. The downloaded images are stored in separate folders for alcoholic and non-alcoholic beverages, which are created if they do not already exist.

The notebook also includes code to resize the downloaded images to a specified size and to create a training-validation split of the data.

Once the data is downloaded, the user can use the model_training.ipynb notebook to build and train a machine learning model to classify the images as alcoholic or non-alcoholic.

### Model Training
This notebook uses the images downloaded by the data downloader.ipynb notebook to train a convolutional neural network (CNN) to classify images of alcoholic and non-alcoholic beverages.

The first step in the notebook is to load the data using the ImageDataGenerator class from the tensorflow.keras.preprocessing.image module. The ImageDataGenerator class generates batches of tensor image data with real-time data augmentation. It also rescales the pixel values of the images to the range [0, 1]. The data is loaded from the directories created by the data downloader.ipynb notebook, where each directory represents a class (alcoholic or non-alcoholic).

Next, we build the CNN model using the transfer learning technique. We use the MobileNet architecture as our pre-trained model. The pre-trained weights from the MobileNet model trained on the ImageNet dataset are used as a starting point for our model. We add a new fully connected layer at the end of the pre-trained model and train this layer to classify our images as alcoholic or non-alcoholic. We freeze the weights of the pre-trained layers during the initial training to keep the learned features intact.

After building the model, we compile it with the adam optimizer and use binary crossentropy as the loss function since we are performing binary classification. We then fit the model to the training data, which includes data augmentation techniques like rotation, zoom, flip, and shift, to avoid overfitting. We also set up an early stopping callback to prevent overfitting and reduce training time.

Once the model is trained, we evaluate its performance on the testing set using the evaluate() function from the tensorflow.keras.Model class. We use classification metrics like accuracy, precision, recall, and F1 score to evaluate the model's performance. We also generate a confusion matrix to visualize the model's performance on the testing set.

Finally, we save the trained model to a file in the models directory using the save() method of the tensorflow.keras.Model class. The saved model can then be loaded and used for prediction on new images.

The model_training.ipynb file leverages various machine learning and deep learning libraries, such as TensorFlow, Keras, and scikit-learn, along with other Python libraries for data manipulation and visualization.
## Requirements
The project uses several Python libraries, which can be installed using the following command:

``` pip install -r requirements.txt ```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This project was inspired by a competition held by Iran's National Elites Institute. The goal of the competition was to classify images of beverages as alcoholic or non-alcoholic using machine learning techniques. Our project implements this classification task using transfer learning with the MobileNet architecture in TensorFlow.
