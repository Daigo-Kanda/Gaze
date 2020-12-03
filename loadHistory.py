import pickle
import matplotlib.pyplot as plt

with open("/mnt/data2/model/Git_Original/Original_model/trainHistoryDict.pickle", "rb") as f:
    history = pickle.load(f)


mae = history['mae']
val_mae = history['val_mae']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(len(mae))

plt.plot(epochs, mae, 'bo', label='Training acc')
plt.plot(epochs, val_mae, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("mae.jpg")
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("mse.jpg")
plt.show()