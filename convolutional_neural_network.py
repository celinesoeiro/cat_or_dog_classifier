# Convolutional Neural Network

# Importing the libraries
from os import listdir
#from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
tf.__version__

# Part 1 - Data Preprocessing

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]
    ))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN
cnn.summary()

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
history = cnn.fit(
        x = training_set,
        epochs = 20,
        validation_data = test_set,
        )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Precisão do treinamento')
plt.plot(epochs, val_acc, 'b', label='Precisão da validação')
plt.title('Precisão do treinamento e da validação')
plt.legend(loc=0)
plt.figure()

plt.show()

# Predicting
listPredicted = list()
listAnimals = list()
listFiles = list()
listOriginal = list()
for filename in listdir('dataset/test_image'):
    path = 'dataset/test_image/' + filename
    img = image.load_img(path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # images = np.vstack([x])
    # classes = cnn.predict(images, batch_size=10)
    classes = cnn.predict(x)
    training_set.class_indices
    if classes[0][0] == 1:
        animal = 'cachorro'
        classe = 1
    else:
        animal = 'gato'
        classe = 0

    # if (classes < 0.5):
    #     classes = 0
    #     animal = 'gato'
    # else: 
    #     classes = 1
    #     animal = 'cachorro'
    
    if (filename.find('cat')):
        listOriginal.append(0)
    elif (filename.find('dog')):
        listOriginal.append(1)
    
    listPredicted.append(classe)
    listAnimals.append(animal)
    listFiles.append(filename)
    
class Result(object):
    def __init__(self, filename, predicted, animal):
        self.filename = filename
        self.predicted = predicted
        self.animal = animal
    
r = {}
for i in range(len(listAnimals)):
    r[i] = Result(listFiles[i],listPredicted[i],listAnimals[i])

#confusion matrix
cm = confusion_matrix(listOriginal, listPredicted)
print(cm)

