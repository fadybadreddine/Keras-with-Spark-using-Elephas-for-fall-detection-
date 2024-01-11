The primary training dataset, accessible through this link (http://fenix.ur.edu.pl/mkepski/ds/uf.html) , comprises sequences of video data captured by cutting-edge motion capture and computer vision devices. These include Microsoft Kinect cameras and sensors such as the PS Move and x-IMU.The dataset consists of 70 detailed video sequences, with 30 dedicated to simulating falls and 40 for recording daily activities. This dataset provides essential data for training our model to accurately differentiate between these two types of events.
Annotation Process:

Manual Annotation of Position:

Each photo in the dataset is annotated manually with its specific position information. This  include details such as the location or pose of the subject in the image.
Assembly into Series of 16:

The photos are organized into series, with each series consisting of 16 images. These images are  selected to represent a sequence or scenario related to a particular fall event.
CSV File ("1.csv"):

The manual annotations, including position information, are recorded in a CSV file associated with each fall scenario folder. The file follows a structure where each row corresponds to a different position in the series of 16 photos, and the 17th column contains the label. Google Drive Dataset: 
(https://drive.google.com/drive/folders/1mkED6QJ2ynfnySBCuQpLGb0aQHnFiZ6e?usp=sharing)


I am sharing a Google Drive link containing various folders for different fall scenarios. Each folder follows a naming convention specifying the fall number, camera, and color format (grayscale).
Explanation of the Drive Link:

The provided link  leads to a Google Drive directory where you can find subdirectories for each fall scenario.
Inside each fall scenario directory, you'll find the annotated photos, a "1.csv" file with position and label information and  a "labels" file providing details about individual photos.

For the validation of our model, a secondary dataset was employed. This dataset, obtained from "Auvinet, C. Rougier, J. Meunier, A. St-Arnaud, and J. Rousseau" in January 2011, is known as the "Multiple Cameras Fall Data Set " (https://www.researchgate.net/publication/267693256_Multiple_cameras_fall_data_set)
The link to the dataset on Google Drive is as follows:  (https://drive.google.com/drive/folders/1tOWTiIRAkWbq8vrtN7ljpZVcmWM1TFz0?usp=sharing)
is based on a multicamera system consisting of eight economical IP cameras strategically positioned to cover an entire room. The recorded video sequences reflect common challenges that can lead to segmentation errors, including high video compression, shadows, reflections, complex backgrounds, lighting variations, worn objects, and occlusions .
This dataset is built on a multicamera system comprising eight cost-effective IP cameras strategically positioned to cover an entire room. The recorded video sequences encapsulate common challenges leading to segmentation errors, including high video compression, shadows, reflections, complex backgrounds, lighting variations, worn objects, and occlusions.

Additionally, a second CSV file containing an example of a Spark DataFrame has been added to this Google Drive repository .This file can be used to explore and understand the data structure within the context of the model we are developing.
