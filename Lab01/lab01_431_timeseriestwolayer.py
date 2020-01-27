import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


def mackey_glass_euler(n):
    sequence = []
    for i in range(n):
        if i > 25:
            sequence.append(sequence[-1] + (0.2*sequence[-26] /
                                            (1 + sequence[-26]**10)) -
                            0.1 * sequence[-1])
        elif i == 0.0:
            sequence.append(1.5)
        else:
            sequence.append(sequence[-1] + (0.2 * 0.0 /
                                            (1 + 0.0 ** 10)) -
                            0.1 * sequence[-1])
    return sequence

full_seq = mackey_glass_euler(1510)

plt.plot(full_seq)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.savefig("plots/mackey_glass.svg")



input, output = [], []

for t in range(300, 1500):
    input.append([full_seq[t-20], full_seq[t-15], full_seq[t-10],
                       full_seq[t-5], full_seq[t]])
    output.append(full_seq[t+5])

output = np.array(output)
input = np.array(input)

train_input = input[:1000]
train_output = output[:1000]
test_input = input[1000:]
test_output = output[1000:]

def build_model():
  model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=[5]),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
EPOCHS = 100
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
  train_input, train_output,
  epochs=EPOCHS, validation_split=0.2, verbose=2,
  callbacks=[early_stop])

plt.plot(history.history["mse"])
plt.plot(history.history["val_mse"])
plt.ylabel('MSE [MPG]')
plt.show()

pred_output = model.predict(test_input)

plt.plot(pred_output, label='pred')
plt.plot(test_output, label='true')
plt.legend()
plt.show()


