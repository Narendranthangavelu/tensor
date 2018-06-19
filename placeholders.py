import tensorflow as tf
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
add= a+b
with tf.Session() as sess:
    print(sess.run(add,{a:[1,5],b:[5,8]}))
