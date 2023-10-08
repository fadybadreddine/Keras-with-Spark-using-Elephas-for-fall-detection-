Project to integrate Keras with Spark using Elephas
This project presents an example of how to integrate a deep neural network (RNN) architecture built with Keras into a Spark infrastructure for training and evaluation.


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

Comment Exécuter
Si vous utilisez Google Colab, téléchargez le code dans un notebook et exécutez chaque cellule séquentiellement.
Assurez-vous que les fichiers de données (comme df.csv) sont accessibles à partir de l'emplacement spécifié ou mettez à jour les chemins en conséquence.