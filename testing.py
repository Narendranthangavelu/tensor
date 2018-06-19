import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def read_dataset():
    df = pd.read_csv("sonar.csv")
    x = df[df.columns[0:60]].values
    y1 = df[df.columns[60]].values
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)
    return (x, Y,y1)

X, Y,y1 = read_dataset()
X, Y = shuffle(X, Y, random_state=1)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = 2
model_path = os.getcwd()+"\model"

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
#W = tf.Variable(tf.zeros([n_dim, n_class]))
#b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


# Define the Model

def multilayer_perceptron(X, weights, biases):
    # Hidden layer with sigmoid activation
    X = tf.cast(X, tf.float32)
    print(X)
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid([layer_1])

    # Hidden layer with sigmoid activation
    print(layer_1, ">>>", weights['h2'])

    layer_1 = tf.reduce_mean(layer_1, 0)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Hidden layer with sigmoid activation

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Hidden layer with sigmoid activation

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)

    # Output layer with Linear Activation
    output_layer = tf.matmul(layer_4, weights['out']) + biases['out']

    return output_layer


# Define the weights and biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
     'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
}

# Call your model defined

y = multilayer_perceptron(x, weights, biases)

# Initialize all the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess,model_path)

#predictions
prediction=tf.argmax(y, 1)
correct_predicton = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predicton, tf.float32))

for i in range(10,100):
    pred=sess.run(prediction,{x:X[i].reshape(1,n_dim)})
    accu=sess.run(accuracy,{x:X[i].reshape(1,n_dim),y_:Y[i].reshape(1,n_class)})
    print("real:",y1,' pred:',pred,' accu:',accu,'\n')