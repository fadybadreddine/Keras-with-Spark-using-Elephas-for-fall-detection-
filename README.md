Project to integrate Keras with Spark using Elephas
This project presents an example of how to integrate a deep neural network (RNN) architecture built with Keras into a Spark infrastructure for training and evaluation.
In this project, two distinct datasets were utilised for training and validation: 
The primary training dataset, accessible through this link (http://fenix.ur.edu.pl/mkepski/ds/uf.html) , comprises sequences of video data captured by cutting-edge motion capture and computer vision devices. These include Microsoft Kinect cameras and sensors such as the PS Move and x-IMU. The dataset consists of 70 detailed video sequences, with 30 dedicated to simulating falls and 40 for recording daily activities. This dataset provides essential data for training our model to accurately differentiate between these two types of events.
For the validation of our model, a secondary dataset was employed. This dataset, obtained from "Auvinet, C. Rougier, J. Meunier, A. St-Arnaud, and J. Rousseau" in January 2011, is known as the "Multiple Cameras Fall Data Set " (https://www.researchgate.net/publication/267693256_Multiple_cameras_fall_data_set)
is based on a multicamera system consisting of eight economical IP cameras strategically positioned to cover an entire room. The recorded video sequences reflect common challenges that can lead to segmentation errors, including high video compression, shadows, reflections, complex backgrounds, lighting variations, worn objects, and occlusions
Installation
To install the necessary dependencies, run the following command:
pip install elephas

Code description
1.imports:
The code begins by importing the necessary libraries such as Keras, Spark, sklearn, pandas, etc.
2.connection to Google Drive :
If you use Google Colab, the code contains a step to mount your Google Drive so you can access the files on it.
3.loading data :
A load_dataset function is provided to load your data from a CSV file. 
4.data preparation :
Data is split into training and test sets, then converted into Spark DataFrames.
5.model creation :
Two model architectures are provided: one using the LSTM layer and the other using the GRU layer.
6.Elephas configuration:
The Keras model is then converted into an Elephas model for integration with Spark.
7.training and evaluation:
The model is trained using Spark and evaluated on the test set.
8.display results:
Predictions are converted into a binary format according to a threshold. Finally, various evaluation metrics, such as confusion matrix, accuracy and classification ratio, are displayed.

How to run
If you're using Google Colab, download the code into a notebook and run each cell sequentially.
Make sure that data files (such as df.csv) are accessible from the specified location, or update the paths accordingly.
