import tensorflow as tf
feature_columns = [
    tf.feature_column.numeric_column('vector',shape=(300,), dtype=tf.float32),
    tf.feature_column.numeric_column('summaryVector',shape=(300,), dtype=tf.float32),


]

label_column = tf.feature_column.numeric_column('score',shape=(1,), dtype=tf.float32)

features_spec = tf.feature_column.make_parse_example_spec(
    feature_columns + [label_column]
)

def input_fn_train(file, num_epochs):
    #print("hi")
    dataset = tf.contrib.data.make_batched_features_dataset(
        file_pattern = [file],
        batch_size = 10,
        features=features_spec,
        num_epochs=num_epochs,
        shuffle=True,
        shuffle_buffer_size=10
    )
    it = dataset.make_one_shot_iterator()
    features = it.get_next()
    print(features)
    labels = features.pop('score')
    labels = tf.Print(labels, [features['summaryVector']], 'features=')
    return features, labels
def input_fn_eval(file, num_epochs):
    #print("hi")
    dataset = tf.contrib.data.make_batched_features_dataset(
        file_pattern = [file],
        batch_size = 10,
        features=features_spec,
        num_epochs=num_epochs,
        shuffle=True,
        shuffle_buffer_size=10
    )
    it = dataset.make_one_shot_iterator()
    features = it.get_next()
    print(features)
    labels = features.pop('score')
    labels = tf.Print(labels, [features['summaryVector']], 'features=')
    labels = tf.Print(labels, [labels], 'labels=')
    return features, labels

model_dir="model_dir"


estimator = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    )

inputs = input_fn_train("parsedTrainaverage.tfr",1)
with tf.Session() as sess:
    print(sess.run(inputs))


estimator.train(input_fn=lambda: inputs,steps=1)
#metrics = estimator.evaluate(input_fn=input_fn_eval)
#predictions = estimator.predict(input_fn=input_fn_predict)
