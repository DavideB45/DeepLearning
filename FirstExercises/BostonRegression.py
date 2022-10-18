from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# questi dati sono un po' peggio di quelli che abbiamo usato fino ad ora
# perché alcuni usano unità di misura diverse, quindi dobbiamo normalizzarli
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
# in questo caso abbiamo anche pochi dati e quindi richiamo di avere un po' più
# overfitting di prima, per attenuare il problema useremo un modello più semplice
from keras import models
from keras import layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # usiamo mean square error come funzione di perdita
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
# siccome abbiamo pochi dati dobbiamo usare k-fold cross-validation
import numpy as np
k=4
num_val_samples = len(train_data) // k 
num_epochs = 500
all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    # prepara validation per la i iterazione
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    # prepara test per la i iterazione
    partial_train_data = np.concatenate( [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate( [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    # crea il modello (già compilato)
    model = build_model()
    history = model.fit(partial_train_data, 
                        partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# dopo aver visto che il modello va in overfitting dall' 80 iterazione in poi
# possiamo scegliere i parametri migliori e allenare un nuovo modello usando tutti i dati
# puoi anche provare a cambiare cose come il numero di layer nascosti ecc
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mse_score)
print(test_mae_score)
