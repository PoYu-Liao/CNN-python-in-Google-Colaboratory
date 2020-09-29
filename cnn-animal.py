import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
#import tensorflow_data_validation as tfdv






tf.reset_default_graph()
sess = tf.InteractiveSession()

data = {}

def resize(image, tw=150, th=150):
    w, h =np.shape(image)
    
    if w>150 and h>150:
        dw = int((w - tw)/2)
        dh = int((h - th)/2)
        image = image[dw:-dw, dh:-dh]
    image =cv2.resize(image, (tw, th))
    return image[:, :, np.newaxis]


def load(dir_name):
    tmp=[]
    filenames = []
    for file in os.listdir(dir_name):
        filename = "{}/{}".format(dir_name, file)
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = resize(image)
        tmp.append(image)
        filenames.append(file)
    return tmp,  filenames

data["cat"],_ = load("./MyDrive/Colab Notebooks/CNN/training/cat")
data["dog"],_ = load("./MyDrive/Colab Notebooks/CNN/training/dog")
data["horse"],_ = load("./MyDrive/Colab Notebooks/CNN/training/horse")
data["chicken"],_ = load("./MyDrive/Colab Notebooks/CNN/training/chicken")

image_width = 150
image_height = 150
image_depth = 1

vocab_size = 4

types = {
    "cat": 0,
    "dog": 1,
    "horse": 2,
    "chicken": 3
    
    
    
}

train_input = tf.placeholder(tf.float32, (None, image_height, image_width, image_depth), "train_input")
train_label = tf.placeholder(tf.int32, (None,), "train_label")

def part(tag, under, upper):
    L = len(data[tag])
    a = i%L
    b = (i+2)%L
    if b > a:
        v = np.array(data[tag][a:b])
    elif b==0:
        v = np.array(data[tag][a:])
    else:
        v1 = np.array(data[tag][a:])
        v2 = np.array(data[tag][:b])
        v = np.concatenate((v1, v2), 0)
    return v

def feed(i):
    X = part("cat", i, i+2)
    X = np.concatenate((X, part("dog", i, i+2)), 0)
    X = np.concatenate((X, part("horse", i, i+2)), 0)
    X = np.concatenate((X, part("chicken", i, i+2)), 0)
    Y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    return {
        train_input: X,
        train_label: Y
    
    }

#convolution 1
conv1 = tf.layers.conv2d(inputs = train_input, filters=8, kernel_size=[50, 50], padding="same")
pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2, 2], strides=2)

#convolution 2
conv2 = tf.layers.conv2d(inputs = conv1, filters=32, kernel_size=[35, 35], padding="same")
pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size=[2, 2], strides=1)


#convolution 3
conv3 = tf.layers.conv2d(inputs = conv2, filters=32, kernel_size=[25, 25], padding="same")
pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size=[3, 3], strides=3)


#convolution 4
conv4 = tf.layers.conv2d(inputs = pool3, filters=64, kernel_size=[10, 10], padding="same")
pool4 = tf.layers.max_pooling2d(inputs = conv4, pool_size=[2, 2], strides=2)

#convolution 5
conv5 = tf.layers.conv2d(inputs = pool4, filters=128, kernel_size=[5, 5], padding="same")
pool5 = tf.layers.max_pooling2d(inputs = conv5, pool_size=[5, 5], strides=5)



#Flatten
flat = tf.contrib.layers.flatten(pool5)

#fully Connected
output = tf.contrib.layers.fully_connected(flat, vocab_size, activation_fn=None)
#print(output)

#Prediction
prediction_rate =tf.nn.softmax(output)
prediction_result = tf.argmax(prediction_rate, 1)

#Cost
target = tf.one_hot(train_label, depth=vocab_size, dtype=tf.float32)
loss_function = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)
loss = tf.reduce_mean(loss_function)


#Optimizer
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

vs =[]
t = []


for i in range(0, 850):
    fd = feed(i)
    _, v = sess.run([optimizer, loss], fd)
    print("time: {}, loss: {}".format(i, v))
    vs.append(v) 
    t.append(i)

#750 800 850 900 950

#loss curve
plt.plot(t, vs)
plt.xlabel('time')
plt.ylabel('loss')
plt.show()

#Test
test_data, files = load("./MyDrive/Colab Notebooks/CNN/test")

result = sess.run(prediction_rate, { train_input: test_data})
for i in range(0, 8):
    r = np.round(result[i]*100, 2)
    print("filename: {}\t cat:{}%, dog:{}%, horse:{}%, chicken:{}%".format(files[i], r[0], r[1], r[2], r[3]))
