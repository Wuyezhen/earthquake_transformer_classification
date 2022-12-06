import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def make_model():
    inputs = keras.Input(shape=(32768, 1))
    x = layers.Conv1D(64, 7, strides=4, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    block_0_output = layers.MaxPooling1D(pool_size=3, strides=4, padding="same")(x)

    x = layers.Conv1D(64, 3, strides=1, padding='same')(block_0_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 3, strides=1, padding='same')(x)
    block_1_output = layers.add([x, block_0_output])
    block_1_output = layers.ReLU()(block_1_output)

    x = layers.Conv1D(64, 3, strides=1, padding='same')(block_1_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 3, strides=1, padding='same')(x)
    block_2_output = layers.add([x, block_1_output])
    block_2_output = layers.ReLU()(block_2_output)

    x = layers.Conv1D(128, 3, strides=4, padding='same')(block_2_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 3, strides=1, padding='same')(x)
    y = layers.Conv1D(128, 1, strides=4, padding='same')(block_2_output)
    block_3_output = layers.add([x, y])
    block_3_output = layers.ReLU()(block_3_output)

    x = layers.Conv1D(128, 3, strides=1, padding='same')(block_3_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 3, strides=1, padding='same')(x)
    block_4_output = layers.add([x, block_3_output])
    block_4_output = layers.ReLU()(block_4_output)

    x = layers.Conv1D(256, 3, strides=4, padding='same')(block_4_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same')(x)
    y = layers.Conv1D(256, 1, strides=4, padding='same')(block_4_output)
    block_5_output = layers.add([x, y])
    block_5_output = layers.ReLU()(block_5_output)

    x = layers.Conv1D(256, 3, strides=1, padding='same')(block_5_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same')(x)
    block_6_output = layers.add([x, block_5_output])
    block_6_output = layers.ReLU()(block_6_output)


    x = layers.Conv1D(512, 3, strides=4, padding='same')(block_6_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same')(x)
    y = layers.Conv1D(512, 1, strides=4, padding='same')(block_6_output)
    block_7_output = layers.add([x, y])
    block_7_output = layers.ReLU()(block_7_output)

    x = layers.Conv1D(512, 3, strides=1, padding='same')(block_7_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same')(x)
    block_8_output = layers.add([x, block_7_output])
    block_8_output = layers.ReLU()(block_8_output)

    x = layers.GlobalAveragePooling1D()(block_8_output)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="resnet-18")
    return model