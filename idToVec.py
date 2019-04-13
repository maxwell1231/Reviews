import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    testArray = np.zeros((5,5))
    testVariable = tf.get_variable("test", initializer=testArray)
    tf.initializers.global_variables().run()
    print(testVariable.eval())
    saver = tf.train.Saver()
    saver.save(sess, "/tmp/my-model")
