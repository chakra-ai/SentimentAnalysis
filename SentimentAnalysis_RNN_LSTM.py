import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
from PreProcessing import extract_label_data
warnings.filterwarnings("ignore")

# Import data
df = pd.read_csv("Roman Urdu DataSet.csv",encoding='utf8', header=None)
df.columns = ['text','target','junk']
df.drop('junk',axis=1, inplace=True)
df.dropna(inplace=True)
df = df[df['target'] != 'Neative']
data = df[df['target'] != 'Neutral']

# Extract Cleaned reviews and labels
labels, reviews = extract_label_data(data)

# Identify the Maximum length of the document and set the MAX_SEQUENCE_LENGTH accordingly. 
max_document_length = max([len(x.split(' ')) for x in reviews])
print(max_document_length)

MAX_SEQUENCE_LENGTH = 50

# Build Vocab Processor
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SEQUENCE_LENGTH)
x_data = np.array(list(vocab_processor.fit_transform(reviews)))
y_output = np.array(labels)

vocabulary_size = len(vocab_processor.vocabulary_)

np.random.seed(22)
shuffle_indices = np.random.permutation(np.arange(len(x_data)))

x_shuffled = x_data[shuffle_indices]
y_shuffled = y_output[shuffle_indices]

TRAIN_DATA = 9000
TOTAL_DATA = len(labels)

train_data = x_shuffled[:TRAIN_DATA]
train_target = y_shuffled[:TRAIN_DATA]

test_data = x_shuffled[TRAIN_DATA:TOTAL_DATA]
test_target = y_shuffled[TRAIN_DATA:TOTAL_DATA]

tf.reset_default_graph()

x = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH])
y = tf.placeholder(tf.int32, [None])

batch_size = 25
embedding_size = 50
max_label = 2

embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

embeddings = tf.nn.embedding_lookup(embedding_matrix, x)

#print(embeddings)

lstmCell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob = 0.75)
_, (encoding, _) = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype = tf.float32)
logits = tf.layers.dense(encoding, max_label, activation=None)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels = y)
loss = tf.reduce_mean(cross_entropy)
prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
num_epochs = 20

with tf.Session() as sess:
    init.run()

    for epoch in range(num_epochs):

        num_batches = int(len(train_data) // batch_size) + 1

        for i in range(num_batches):

            min_ix = i * batch_size
            max_ix = np.min([len(train_data), ((i+1)* batch_size)])

            x_train_batch = train_data[min_ix:max_ix]
            y_train_batch = train_target[min_ix:max_ix]

            train_dict = {x: x_train_batch, y:y_train_batch} 
            sess.run(train_step, feed_dict = train_dict)

        test_dict = {x: test_data, y: test_target}
        test_loss, test_acc = sess.run([loss, accuracy], feed_dict = test_dict)

        print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.5}'.format(epoch + 1, test_loss, test_acc))
