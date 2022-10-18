from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def title_of_array(to_decode):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decoded_newswire)

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
# codifica dati
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# codifica risultati con one-hot, ovvero un array in cui è tutto zero tranne il valore
# che è assunto dalla lable
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
# in realtà ci sono funzioni già definite in keras:
#from keras.utils.np_utils import to_categorical
#one_hot_train_labels = to_categorical(train_labels)
# a pag 83 un modo diverso per farlo
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# IL MODELLO
# L'esercizio sembra simile a quello precedente ma siccome abbiamo molte più classi
# la dimensionalità dello spazio di uscita deve essere molto maggiore e siccome ogni layer
# può accedere solo alle informazioni del layer precedente con 16 layer intermedi possono essere pochi
# causando una perdita di informazioni quindi useremo layer più larghi (almeno il numero di classi)
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
# l'ultimo layer avrà 46 output (uno per ogni possibile classe)
# l'attivazione userà softmax quindi la somma delle probabilità sarà uno
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)

print(results)

# vediamo come prevediamo le cose
predictions = model.predict(x_test)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()