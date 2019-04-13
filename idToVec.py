import tensorflow as tf
import numpy as np

# with tf.Session() as sess:
#     testArray = np.zeros((5,5))
#     testVariable = tf.get_variable("test", initializer=testArray)
#     tf.initializers.global_variables().run()
#     print(testVariable.eval())
#     saver = tf.train.Saver()
#     saver.save(sess, "./my-model")

with tf.Session() as sess:
    file = open("glove30k.txt")
    counter = 0
    arrayOfWordVecs = np.empty((0, 300))
    tf.initializers.global_variables().run()
    saver = tf.train.Saver()
    for line in file.readlines():
        wordVec = line.split()
        arrayOfWordVecs = np.append(arrayOfWordVecs, np.array([wordVec[1:]]), axis = 0)
        counter += 1
        print(counter)
    wordVecVariable = tf.get_variable("wordVecs", initializer=arrayOfWordVecs)
    saver.save(sess, "./my-model")
