import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split 
from matplotlib.pyplot import imread
import tensorflow as tf
from IPython.display import Image, display
import tensorflow as tf 
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dir_path = "./dog-breed-identification/"
os.listdir(dir_path)

['sample_submission.csv','test', 'train', 'labels.csv']

train_images_path = "./dog-breed-identification/train/"
test_images_path = "./dog-breed-identification/test/"

labels_df = pd.read_csv(dir_path + 'labels.csv')
Image(train_images_path + '000bec180eb18c7604dcecc8fe0dba07.jpg')

labels_df['breed'].value_counts().plot.bar(figsize=(20,10))

#Criando vetor de treino
filenames = [train_images_path + fname + '.jpg' for fname in labels_df['id']]
filenames[:10]

class_names = labels_df['breed'].unique()
class_names[:10]

target_labels = [breed for breed in labels_df['breed']]
target_labels[:10]

target_labels_encoded = [label == np.array(class_names) for label in target_labels]
target_labels_encoded[:2]

NUM_IMAGES = 2000

# Split dados
X_train, X_val, Y_train, Y_val = train_test_split(filenames[:NUM_IMAGES], target_labels_encoded[:NUM_IMAGES], test_size=0.2, random_state=42)

IMAGE_SIZE = 224

# Processamento da Imagem
def process_image(image_path): 

    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels =3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    return img

    # Linkando imagem com label
def get_image_label(image_path, label):

    image = process_image(image_path)
    return image, label

get_image_label(X_train[10], Y_train[10])

BATCH_SIZE = 32

# Função para criar os Batches nos dados de teste. (Função extraida da web, recomendada para o problema dog breed)
def create_data_batches(X, y=None, batch_size = BATCH_SIZE, valid_data= False, test_data=False): 

    if test_data: 
        print("Creating Test data")
        test_data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
        test_data = test_data.map(process_image).batch(BATCH_SIZE) 
        return test_data 
    
    #Create validation data
    if valid_data: 
        print("Creating Validation data")
        valid_data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
        valid_data = valid_data.map(get_image_label).batch(BATCH_SIZE)
        return valid_data
    
    #Shuffle and create training data
    else: 
        print("Creating Training Data") 
        train_data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y))).shuffle(buffer_size = len(X))
        train_data = train_data.map(get_image_label).batch(BATCH_SIZE) 
        return train_data 

train_data = create_data_batches(X_train, Y_train)
valid_data = create_data_batches(X_val, Y_val, valid_data= True)

sample =next(iter(train_data))
sample[0][0]


# Criação de modelo pronto utilizando mobileNetV2 -> Dropout=0.7 foi o melhor caso para evitar
# o overfitting do modelo.
def create_model():
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top = False, 
                                                     classes = len(class_names)) 
    base_model.trainable = False 

    inputs = layers.Input(shape = (224,224,3))
    x = base_model(inputs, training = False) 
    x = tf.keras.layers.GlobalAveragePooling2D(name= "global_average_pooling")(x)
    x = layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

    ModelDogBreed = tf.keras.Model(inputs, outputs) 

    ModelDogBreed.compile(loss = "categorical_crossentropy", 
                         optimizer = tf.keras.optimizers.Adam(), 
                         metrics=["accuracy"]) 

    return ModelDogBreed


model = create_model()
EarlyStoppingCallbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, baseline=None, restore_best_weights=True
)

ModelDogBreed_History = model.fit(train_data, 
                                         steps_per_epoch = len(train_data),
                                         epochs = 30, 
                                         validation_data= valid_data, 
                                         validation_steps = len(valid_data),
                                         callbacks = [EarlyStoppingCallbacks])
model.evaluate(valid_data)