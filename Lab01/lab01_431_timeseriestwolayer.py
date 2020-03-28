import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def mackey_glass_euler(n):
    sequence = []
    for i in range(n):
        if i > 25:
            sequence.append(sequence[-1] + (0.2 * sequence[-26] /
                                            (1 + sequence[-26] ** 10)) -
                            0.1 * sequence[-1])
        elif i == 0.0:
            sequence.append(1.5)
        else:
            sequence.append(sequence[-1] + (0.2 * 0.0 /
                                            (1 + 0.0 ** 10)) -
                            0.1 * sequence[-1])
    return sequence


full_seq = mackey_glass_euler(1510)

### Add Noise
amount = 0.09
# plt.plot(full_seq)
full_seq = np.add(full_seq, np.random.normal(0.0, amount, size=len(full_seq)))

# plt.plot(full_seq)
# plt.xlabel("t")
# plt.ylabel("x(t)")
# plt.savefig("plots/mackey_glass.svg")
# plt.show()

input, output = [], []

for t in range(300, 1500):
    input.append([full_seq[t - 20], full_seq[t - 15], full_seq[t - 10],
                  full_seq[t - 5], full_seq[t]])
    output.append(full_seq[t + 5])

output = np.array(output)
input = np.array(input)

train_input = input[:1000]
train_output = output[:1000]
test_input = input[1000:]
test_output = output[1000:]


def build_model():
    model = keras.Sequential([
        layers.Dense(4, activation='relu', input_shape=[5],
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(4, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

epochs = []
loss = []
val_loss, test_loss, timepass = [], [], []
weights = []
for i in range(3):
    tic = time.perf_counter()
    model = build_model()
    EPOCHS = 1000
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        train_input, train_output,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[early_stop])

    toc = time.perf_counter()
    timepass.append(toc - tic)
    results = model.evaluate(test_input, test_output)

    test_loss.append(results[1])

    epochs.append(history.epoch[-1])
    loss.append(history.history["mse"][-1])
    val_loss.append(history.history["val_mse"][-1])

# weights = np.array(weights).flatten()

# plt.hist(weights, bins=20, range=(-1, 1))
# plt.title("Reg. factor = 0.0001")
# plt.xlabel("size of weights")
# plt.ylabel("frequency")
# plt.savefig("plots/histogram_reg00001")

print("{0:.3f} ({1:.3f})".format(np.mean(np.array(loss)), np.std(np.array(loss))))
print("{0:.3f} ({1:.3f})".format(np.mean(np.array(val_loss)), np.std(np.array(val_loss))))
print("{0:.3f} ({1:.3f})".format(np.mean(np.array(epochs)), np.std(np.array(epochs))))

print("Test {0:.3f} ({1:.3f})".format(np.mean(np.array(test_loss)), np.std(np.array(test_loss))))
print("Time {0:.3f} ({1:.3f})".format(np.mean(np.array(timepass)), np.std(np.array(timepass))))

# plt.plot(history.history["mse"])
# plt.plot(history.history["val_mse"])
# plt.ylabel('MSE [MPG]')
# plt.show()


pred_output = model.predict(test_input)

plt.plot(pred_output, label='pred')
plt.plot(test_output, label='true')
plt.legend()
plt.show()
