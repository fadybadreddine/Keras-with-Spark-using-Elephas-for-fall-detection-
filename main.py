
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

import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(filename):
    # Charger l'ensemble de données sous forme de pandas DataFrame en utilisant le paramètre 'filename'
    data = pd.read_csv(filename, header=None)
    # Extraire le tableau numpy
    dataset = data.values
    # Séparer les variables d'entrée (X) et de sortie (y)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    # Formater tous les champs en chaîne de caractères
    X = X.astype(str)
    # Remodeler la sortie pour qu'elle soit un tableau 2D
    y = y.reshape((len(y), 1))
    return X, y

# Utilisation de la fonction avec un nom de fichier
X, y = load_dataset('chemin/vers/votre/fichier.csv')

# Séparation des données en ensembles d'entraînement et de test
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
#Veuillez choisir d'utiliser soit un modèle LSTM soit un modèle GRU
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
    model.add(Embedding(input_dim=168, output_dim=50, input_length=input_length))
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
paramGrid = ParamGridBuilder().addGrid(estimator.epochs, [10, 50]).addGrid(estimator.batch_size, [16, 32]).build()
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label", metricName="areaUnderROC")
crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cvModel = crossval.fit(df)


predictions = cvModel.transform(test_df)
predictions.select("features", "label", "prediction").show()


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="final_prediction", metricName="accuracy")
accuracy = evaluator.evaluate(test_df)
print("Accuracy du modèle : ", accuracy)


from pyspark.sql.functions import col
confusion_matrix = test_df.groupBy('label').pivot('final_prediction', [0.0, 1.0]).count().na.fill(0)
confusion_matrix.show()
TP = confusion_matrix.filter(col("label") == 1.0).select("`1.0`").first()[0]
FN = confusion_matrix.filter(col("label") == 1.0).select("`0.0`").first()[0]
FP = confusion_matrix.filter(col("label") == 0.0).select("`1.0`").first()[0]
TN = confusion_matrix.filter(col("label") == 0.0).select("`0.0`").first()[0]

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
print("Précision: ", precision)

specificite = TN / (TN + FP) if (TN + FP) > 0 else 0
print("Spécificité: ", specificite)

recall = TP / (TP + FN) if (TP + FN) > 0 else 0
print("Rappel: ", recall)

f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print("Score F1: ", f1_score)





