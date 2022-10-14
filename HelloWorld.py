from keras import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Jist print the format of the dataset
train_images.shape
len(train_labels)
train_labels

from keras import models
from keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
#softmax is a layer in wich the sum of every output sums to 1
network.add(layers.Dense(10, activation='softmax'))
