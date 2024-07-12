# os是一个标准库模块，提供方便的使用操作系统功能的方式
import os

# TF_CPP_MIN_LOG_LEVEL 是一个环境变量，设置此环境变量可以开工至TensorFlow运行输出的日志数量
# 这对于减少控制台的日志干扰或调试Tensorflow代码非常有用
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# 加载MNIST数据集，分为训练接（x，y）和（x_val,y_val）
(x, y), (x_val, y_val) = datasets.mnist.load_data()

# 将训练集的图像数据转换为TensorFlow的张量，并将其数据类型转换为浮点数
# 同时将x的像素值范围自从[0,255]标准化到[0,1]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.

# 将训练集的数据标签y转换为Tensorflow张量，数据类型为整数
y = tf.convert_to_tensor(y, dtype=tf.int32)

# 将标签数据y进行one-hot编码，转换为10类标签的二进制形式（因为MNIST有10个数字类别）
y = tf.one_hot(y, depth=10)

# 打印转换后的x和y的维度，查看数据结构
# 上面的归一化操作仅涉及数据类型和值的范围调整，并不涉及改变数据的维度
print(x.shape, y.shape)
# 输出(60000, 28, 28) (60000, 10) 证明有60000个图像样本，每个图像是‘28*28’的


# tf.data.Dataset.from_tensor_slices 函数创建一个 Dataset 对象，该对象会将输入的两个张量 x 和 y 切片，
# 生成一个包含对应样本的数据集。这里，x 是图像数据，y 是标签数据
# 这种数据组织方式使得数据可以被按批次迭代处理，这对于训练机器学习模型非常有用
# 因为它允许逐批次加载和处理数据，而不是一次性将所有数据加载到内存中

train_dataset = tf.data.Dataset.from_tensor_slices((x, y))

# batch() 方法将数据集的元素组合成指定大小的批次。在这个例子中，批次大小设置为200，
# 意味着每个批次将包含200个图像及其对应的标签。
train_dataset = train_dataset.batch(200)


model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])
optimizer = optimizers.SGD(learning_rate=0.001)


# 定义一个函数 train_epoch，接受一个函数epoch，表示当前的训练周期
def train_epoch(epoch):
# 在训练数据集 train_dataset 上进行迭代，每次迭代提供数据 x 和标签 y，并跟踪迭代步骤 step
    for step, (x,y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # [b,28,28] → [b,784]
            x = tf.reshape(x, (-1 , 28*28))

        # step1:计算output
# 使用模型model对x的值进行预测，得到输出out，输出最终形状为[b,10]
            out = model(x)


        # step2:计算loss
# 使用平方差损失函数，然后将其总和除以批次大小x.shape(0)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]


        # step3:optimize and update w1,w2,w3,b1,b2,b3
            grads = tape.gradient(loss, model.trainable_variables)
            # w' = w - lr * grad
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()