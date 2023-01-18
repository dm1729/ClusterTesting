# Simple python script to run on GPU cluster, written by ChatGPT
# Need to activate venv from command line before using libraries

import tensorflow as tf
import numpy as np

# Dummy data
# training data
X_train = np.random.rand(1000, 100)
y_train = sum(np.transpose(X_train))

# val data
X_val = np.random.rand(500, 100)
y_val = sum(np.transpose(X_val))

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print statements should go to output file listed in bash script

model.save(filepath=f'{data_dir}/params')