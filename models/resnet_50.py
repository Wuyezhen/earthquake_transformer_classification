import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def make_model():
    inputs = keras.Input(shape=(2048,1))
    x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    block_0_output = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)


    x = layers.Conv1D(64, 1, strides=1, padding='same')(block_0_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 1, strides=1, padding='same')(x)
    y = layers.Conv1D(256, 1, strides=1, padding='same')(block_0_output)

    block_1_output = layers.add([x, y])
    block_1_output = layers.ReLU()(block_1_output)

    x = layers.Conv1D(64, 1, strides=1, padding='same')(block_1_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 1, strides=1, padding='same')(x)

    block_2_output = layers.add([x, block_1_output])
    block_2_output = layers.ReLU()(block_2_output)

    x = layers.Conv1D(64, 1, strides=1, padding='same')(block_2_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 1, strides=1, padding='same')(x)

    block_3_output = layers.add([x, block_2_output])
    block_3_output = layers.ReLU()(block_3_output)

    x = layers.Conv1D(128, 1, strides=1, padding='same')(block_3_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 1, strides=1, padding='same')(x)
    y = layers.Conv1D(512, 1, strides=2, padding='same')(block_3_output)

    block_4_output = layers.add([x, y])
    block_4_output = layers.ReLU()(block_4_output)

    x = layers.Conv1D(128, 1, strides=1, padding='same')(block_4_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 1, strides=1, padding='same')(x)

    block_5_output = layers.add([x, block_4_output])
    block_5_output = layers.ReLU()(block_5_output)

    x = layers.Conv1D(128, 1, strides=1, padding='same')(block_5_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 1, strides=1, padding='same')(x)

    block_6_output = layers.add([x, block_5_output])
    block_6_output = layers.ReLU()(block_6_output)

    x = layers.Conv1D(128, 1, strides=1, padding='same')(block_6_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 1, strides=1, padding='same')(x)

    block_7_output = layers.add([x, block_6_output])
    block_7_output = layers.ReLU()(block_7_output)

    x = layers.Conv1D(256, 1, strides=1, padding='same')(block_7_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(1024, 1, strides=1, padding='same')(x)
    y = layers.Conv1D(1024, 1, strides=2, padding='same')(block_7_output)

    block_8_output = layers.add([x, y])
    block_8_output = layers.ReLU()(block_8_output)

    x = layers.Conv1D(256, 1, strides=1, padding='same')(block_8_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(1024, 1, strides=1, padding='same')(x)

    block_9_output = layers.add([x, block_8_output])
    block_9_output = layers.ReLU()(block_9_output)

    x = layers.Conv1D(256, 1, strides=1, padding='same')(block_9_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(1024, 1, strides=1, padding='same')(x)

    block_10_output = layers.add([x, block_9_output])
    block_10_output = layers.ReLU()(block_10_output)

    x = layers.Conv1D(256, 1, strides=1, padding='same')(block_10_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(1024, 1, strides=1, padding='same')(x)

    block_11_output = layers.add([x, block_10_output])
    block_11_output = layers.ReLU()(block_11_output)

    x = layers.Conv1D(256, 1, strides=1, padding='same')(block_11_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(1024, 1, strides=1, padding='same')(x)

    block_12_output = layers.add([x, block_11_output])
    block_12_output = layers.ReLU()(block_12_output)

    x = layers.Conv1D(256, 1, strides=1, padding='same')(block_12_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(1024, 1, strides=1, padding='same')(x)

    block_13_output = layers.add([x, block_12_output])
    block_13_output = layers.ReLU()(block_13_output)

    x = layers.Conv1D(512, 1, strides=1, padding='same')(block_13_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(2048, 1, strides=1, padding='same')(x)
    y = layers.Conv1D(2048, 1, strides=2, padding='same')(block_13_output)

    block_14_output = layers.add([x, y])
    block_14_output = layers.ReLU()(block_14_output)

    x = layers.Conv1D(512, 1, strides=1, padding='same')(block_14_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(2048, 1, strides=1, padding='same')(x)

    block_15_output = layers.add([x, block_14_output])
    block_15_output = layers.ReLU()(block_15_output)

    x = layers.Conv1D(512, 1, strides=1, padding='same')(block_15_output)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 3, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(2048, 1, strides=1, padding='same')(x)

    block_16_output = layers.add([x, block_15_output])
    block_16_output = layers.ReLU()(block_16_output)

    x = layers.GlobalAveragePooling1D()(block_16_output)

    outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="resnet_50")

    return model