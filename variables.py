import tensorflow as tf
a=tf.Variable(0.3,dtype=tf.float32)
b=tf.Variable(-0.3,dtype=tf.float32)
#input and output
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
#calculated out
linear=a*x+b

#loss
square=tf.square(linear-y)
loss=tf.reduce_sum(square)
#optimizer for gradient descent
optimizer=tf.train.GradientDescentOptimizer(.01)
train_data=optimizer.minimize(loss)
#initializing variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train_data,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    print(sess.run([a,b]))