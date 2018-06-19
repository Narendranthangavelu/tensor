import tensorflow as tf
import os
a= tf.constant(3.0)
b= tf.constant(4.0)
c= a*b
path= os.getcwd()+'\graph'
print(path)
with tf.Session() as sess:
    filewrite=tf.summary.FileWriter(path,sess.graph)
    print(sess.run(c))
