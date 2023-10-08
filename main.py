
"Installation"
#pip install elephas

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Input, Dense, Embedding, LSTM , GRU, Dropout, Dense
from keras.models import Sequential
from keras import optimizers
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Spark Session, Pipeline, Functions, and Metrics
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import DenseVector
from pyspark.sql import SparkSession
from elephas.ml_model import ElephasEstimator
# Créer ou obtenir une session Spark
spark = SparkSession.builder.getOrCreate()

# Obtenir le SparkContext à partir de la session Spark
sc = spark.sparkContext


from google.colab import drive
drive.mount('/content/drive')

#%cd /content/drive/MyDrive


def load_dataset(filename):
# charger l'ensemble de données sous forme de pandas DataFrame
  data = pd.read_csv('dataframefinal.csv', header=None)
  # extraire le tableau numpy
  dataset = data.values
# séparer les variables d'entrée (X) et de sortie (y)
  X = dataset[:, :-1]
  y = dataset[:,-1]
# formater tous les champs en chaîne de caractères
  X = X.astype(str)
# remodeler la sortie pour qu'elle soit un tableau 2D
  y = y.reshape((len(y), 1))
  return X, y

X, y = load_dataset('df.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# Convertissez X_train en listes de DenseVector
X_dense = [DenseVector(row) for row in X_train]

# Conservez y_train comme il est, en supposant que c'est déjà un tableau unidimensionnel de nombres
y_dense = [float(elem) for elem in y_train] 

# Créez des tuples (X, y)
data_tuples = list(zip(X_dense, y_dense))

# Créez un DataFrame Spark à partir des tuples
df = spark.createDataFrame(data_tuples, ["features", "label"])

df.show()


# Convertissez X_train en listes de DenseVector
X_dense1 = [DenseVector(row) for row in X_test]

# Conservez y_train comme il est, en supposant que c'est déjà un tableau unidimensionnel de nombres
y_dense1 = [float(elem) for elem in y_test]  

# Créez des tuples (X, y)
data_tuples = list(zip(X_dense1, y_dense1))

# Créez un DataFrame Spark à partir des tuples
test_df  = spark.createDataFrame(data_tuples, ["features", "label"])

test_df.show()

def create_model(input_length):
    model = Sequential()
    model.add(Embedding(input_dim=168, output_dim=50, input_length=input_length))
    model.add(LSTM(units=256, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=256, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=256, activation='relu', recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model



def create_model(input_length):
    model = Sequential()
    model.add(Embedding(input_dim=95, output_dim=50, input_length=input_length))
    model.add(GRU(units=256, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(units=256, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(units=256, activation='relu', recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# Créer le modèle Keras
model = create_model(16) # à remplacer par la longueur appropriée

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

sgd = optimizers.SGD(learning_rate=0.01)
optimizer_conf = optimizers.serialize(sgd)

# Convertir le modèle Keras en modèle Elephas
estimator = ElephasEstimator()
estimator.set_keras_model_config(model.to_json())
estimator.set_optimizer_config("adam")
estimator.set_mode("asynchronous")
estimator.set_loss("binary_crossentropy")
estimator.set_metrics(['acc'])
estimator.set_epochs(500)
estimator.set_batch_size(16)
estimator.set_validation_split(0.1)
estimator.set_categorical_labels(False)
estimator.set_nb_classes(2)

# Entraîner le modèle avec Spark
pipeline = Pipeline(stages=[estimator])
fitted_pipeline = pipeline.fit(df)

# Evaluate the fitted pipeline model on test data.
prediction = fitted_pipeline.transform(test_df)
df2 = prediction.select("label", "prediction")
df2.show(100,50)



def apply_threshold(value):
    return 1.0 if float(value[0]) >= 0.5 else 0.0

threshold_udf = udf(apply_threshold, DoubleType())

df2 = df2.withColumn("final_prediction", threshold_udf("prediction"))
df2.show(100)



# Convertir les prédictions en DataFrame
df_predictions = df2.toPandas()

# Matrice de confusion
print(confusion_matrix(df_predictions['label'], df_predictions['final_prediction']))

# Accuracy
print("Accuracy:", accuracy_score(df_predictions['label'], df_predictions['final_prediction']))

# Autres métriques