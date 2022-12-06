import tensorflow as tf
from tensorflow import keras
from datasets import *
from models.swin_shuffle_transformer import SwinTransformerModel
# from models.VIT import ViT
# from models.swin_transformer import SwinTransformerModel
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from hparams import hparams


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16384)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)



data_path1 = "./data/earthquake_wave/1"
data_path2 = "./data/earthquake_wave/2"
data_path3 = "./data/earthquake_wave/4"
data_path4 = "./data/earthquake_wave/5"
data_path5 = "./data/explosion_wave/1"
data_path6 = "./data/explosion_wave/2"
data_path7 = "./data/explosion_wave/4"
data_path8 = "./data/explosion_wave/5"
test_path1 = "./data/earthquake_wave/3"
test_path2 = "./data/explosion_wave/3"
eq_list = get_file_list(data_path1)
eq_list += get_file_list(data_path2)
eq_list += get_file_list(data_path3)
eq_list += get_file_list(data_path4)
ep_list = get_file_list(data_path5)
ep_list += get_file_list(data_path6)
ep_list += get_file_list(data_path7)
ep_list += get_file_list(data_path8)

test_eq_list = get_file_list(test_path1)
test_ep_list = get_file_list(test_path2)



EPOCHS = 1000
BATCH_SIZE = 64


# 将numpy数据转换为tf训练用的数据
train_examples, train_labels, SHUFFLE_BUFFER_SIZE = load_data_to_array(eq_list, ep_list)
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_examples, test_labels, Vaild_size = load_data_to_array(test_eq_list, test_ep_list)
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(Vaild_size).batch(64)


# 损失和优化
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# 损失和优化 指标的记录
train_loss = tf.keras.metrics.Mean('train_loss',dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss',dtype= tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# L2或者L1正则化
regularizer_histories = {}


# 训练步骤
def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training = True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y_train, predictions)

def test_step(model, x_test, y_test):
    predictions = model(x_test,training=False)
    loss = loss_object(y_test, predictions)

    test_loss(loss)
    test_accuracy(y_test, predictions)

# 日志文件的写入函数
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'log/shuffle_swin_3/' + current_time + '/train'
test_log_dir = 'log/shuffle_swin_3/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

model = SwinTransformerModel()
# model.summary()
os.makedirs('log', exist_ok=True)
# 保存模型
step = tf.Variable(0)
checkpoint_dir = './gx/shuffle_swin_3'
os.makedirs(checkpoint_dir, exist_ok= True)
checkpoint = tf.train.Checkpoint(optimizer = optimizer, model = model, step=step)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
checkpoint.restore(ckpt_manager.latest_checkpoint)


# 训练
acc_best = 0
for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    checkpoint.step.assign_add(1)
    for index, (x_train, y_train) in enumerate(tqdm(train_dataset)):
        train_step(model, optimizer, x_train, y_train)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step= epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for index, (x_test, y_test) in enumerate(tqdm(test_dataset)):
        test_step(model, x_test, y_test)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_accuracy.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step= epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(int(checkpoint.step), train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),test_accuracy.result()*100))

    temp_acc = test_accuracy.result()
    if acc_best < temp_acc:
        ckpt_manager.save(int(checkpoint.step))
        print(' The epoch test accuracy:{} rather than temp best accuracy:{}'.format(temp_acc*100, acc_best*100))
        acc_best = temp_acc
    else:
        print(' The epoch test accuracy:{} less than temp best accuracy:{}'.format(temp_acc * 100, acc_best * 100))


