import nltk
import json
import os
import numpy
import tensorflow as tf
#nltk.download('punkt')



g = open("MediumWordToVecDict.txt")
lines=g.readlines()

def _float_feature(value):
  #"""Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value):
  #"""Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def makeFeature(summarySumVector,sumVector,score):

  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.

  feature = {
      'summaryVector': _float_feature(summarySumVector),
      'vector': _float_feature(sumVector),
      'score': _float_feature([score]),

  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
def getVectors(filename):

    f= open(filename + ".json");
    #g= open(filename + "MediumWordToVecDict.json")
    writer = tf.python_io.TFRecordWriter(filename+"average.tfr")
    count = 0
    for line in open(filename + ".json").xreadlines(  ): count += 1
    for j in range(1800,1900):
        line = json.loads(f.readline())
        sumVector = numpy.empty((300))
        summarySumVector = numpy.empty((300))
        words = 0
        summarywords = 0

        for i in range(len(line['ids'])):
            vector = lines[i]
            vector = vector.split(" ")[1:]
            vector[299]=vector[299][:-2]
            vector = numpy.array(vector, dtype = 'float')
            #print(vector)
            sumVector+= numpy.array(vector)
            words+=1
        if(words>0):
            sumVector = sumVector / words

        for i in range(len(line['summary_ids'])):
            vector = lines[i]
            vector = vector.split(" ")[1:]
            vector[299]=vector[299][:-2]
            vector = numpy.array(vector, dtype = 'float')
            #print(vector)
            summarySumVector+= numpy.array(vector)
            summarywords+=1
        if(summarywords>0):
            summarySumVector = sumVector / summarywords
        #print("{'summarySumVector':"+summarySumVector+", 'SumVector':"+sumVector+", 'score':"+ line['overall']+"}")
        writer.write(makeFeature(summarySumVector,sumVector,line['overall']))
        if(j%100 == 0):
            print(j)
        #my_dict =
        #print(sumVector)

getVectors("parsedTrain")
