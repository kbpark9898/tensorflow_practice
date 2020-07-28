import tensorflow._api.v2.compat.v1 as tf
import numpy as np
tf.reset_default_graph() 
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
tf.global_variables_initializer()
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 1])
w=tf.Variable(tf.random_normal([4, 1]), name="weight")
b=tf.Variable(tf.random_normal([1]), name="bias")
hypo = tf.matmul(x, w) + b
saver = tf.train.Saver()
test_arr=[[12, 6.5, 15.7, 10.8]]
sess2 = tf.Session()
saver.restore(sess2, "./saved.ckpt")
predict = sess2.run(hypo, feed_dict={x:test_arr})
print(predict[0])