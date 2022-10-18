from keras.datasets import imdb
#per aggiungere la regolarizzazione ai modelli
from keras import regularizers

# con num_words decidi di tenere solo le top 10000 parole più frequenti, così eliminando le
# parole più rare si ottengono vettori di dati di dimensione gestibile
# train data e test data sono liste di indici di parole (tra 0 e 9999)
# lable sono 0 o 1 e indicano se una recensione è positiva o negativa
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# TRASFORMIAMO I DATI
# traduciamo le nostre liste di numeri in vettori di 0 e 1, che saranno lunghi 10000
# se una parola è presente il suo indice sarà 1 altrimenti sarà zero
import numpy as np
#funzione per trasformare i dati
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
#vettorizziamo i dati di training
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
#vettorizziamo i lables
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# CREIAMO LA RETE NEURALE
# un modello che funziona bene è un semplice stack di fully connected layers con
# attivazione 'relu' (quella vista nel primo esempio)
# Dense(16, activation= 'relu') |||||| output = relu(dot(W, input) + b)
# 16 è il numero di unità nascoste, che significa che la matrice di pesi avrà la forma (input_dim, 16),
# intuitivamente il numero di layer nascosti può essere interpretato come la libertà del modello,
# ma rende il modello più computazionalmente costoso e può portare a effetti indesiderati
#per ora devi credere allo scrittore e accettare che questo sia il modello adatto:
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
# sigmoid ritorna una valore tra 0 e 1 che può essere interpretato come una probabilità
model.add(layers.Dense(1, activation='sigmoid'))
# adesso compiliamo il modello,
# scegliamo come funzione di perdita una delle più comuni per la classificazione binaria
# la binary_crossentropy; e come ottimizzatore scegliamo rmsprop
model.compile(optimizer='rmsprop',
            loss='mse',
            metrics=['accuracy'])
# dividiamo training e validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# TRAINING
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=5,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# FACCIAMO UN DISEGNINO
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
print(results)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

def classify(my_review):
    my_review_4_input = np.zeros((1, 10000))
    for word in my_review.split(" "):
        try:
            if word_index[word] < 9999 and word_index[word] > -1:
                my_review_4_input[0, word_index[word]] = 1.
        except Exception as e:
            a = 3
    return model.predict(my_review_4_input)[0,0]