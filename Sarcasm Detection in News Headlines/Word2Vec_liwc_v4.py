import pandas as pd
import json, gensim.models, string
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from numpy import zeros
from bs4 import BeautifulSoup


##################################################################

# Custom callback
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.90:
            self.model.stop_training = True

##################################################################


def parse_data(file):
    for l in open(file, 'r'):
        yield json.loads(l)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


def confusion(prediction_labels, test_labels):

    print(classification_report(test_labels, prediction_labels, target_names=['Not Sarcastic', 'Sarcastic']))
    cm = confusion_matrix(test_labels, prediction_labels)

    print('confusion matrix:\n')
    print(cm, "\n\n\n\n")


##################################################################

df = pd.read_json("LIWC-22 Results - Sarcasm_Headlines_Dataset - LIWC Analysis.ndjson", lines=True)

# LIWC_Tweets = df[["i", "they", "article", "auxverb", "certain", "Colon", "Comma", "conj", "friend",
# "male", "negate", "negemo", "Parenth", "QMark", "risk", "sad", "SemiC", "swear", "WC", "WPS"]]
# liwc = np.array(LIWC_Tweets)

LIWC_Debate_Forums = df[["i", "they", "shehe", "adverb", "affiliation", "assent", "auxverb", "compare",
                         "Exclam", "focusfuture", "friend", "function", "health", "informal", "interrog",
                         "netspeak", "number", "quant", "reward", "sad"]]
liwc = np.array(LIWC_Debate_Forums)

a, b = liwc.shape



##################################################################



##################################################################
##################################################################

# Dataset available on Kaggle
# https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

table = str.maketrans('', '', string.punctuation)
datastore = list(parse_data('./Sarcasm_Headlines_Dataset.json'))

sentences = []
labels = []
urls = []

for item in datastore:
    sentence = item["headline"].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    sentence = sentence.replace("'", " ' ")
    sentence = sentence.replace(":", " : ")
    soup = BeautifulSoup(sentence, features="html.parser")
    words = sentence.split()
    filtered_sentences = ""
    for word in words:
        word = word.translate(table)
        filtered_sentences = filtered_sentences + word + " "
    sentences.append(filtered_sentences)
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])


##################################################################
# We calculate unique words
temp_list = []
for sentence in sentences:
    words = sentence.split()
    for word in words:
        temp_list.append(word)
temp_set = set(temp_list)

# We calculate how many words are in each sentence
temp_list_v2 = []
for sentence in sentences:
    temp_list_v2.append(len(sentence.split()))
temp_np=np.array(temp_list_v2)


##################################################################

# Hyperparameters
vocab_size = len(temp_set)
embedding_dim = 300
max_length = temp_np.max() # 39
trunc_type = 'post'
padding_type = 'post'
oov_tok = ""


##################################################################
##################################################################

from sklearn.model_selection import train_test_split

training_sentences, testing_sentences, training_labels, testing_labels, temp_list, test_temp_list = \
    train_test_split(sentences, labels, liwc, test_size=0.15, random_state=42)

###################################################################
###################################################################

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
# tokenizer.fit_on_texts(training_sentences)
tokenizer.fit_on_texts(sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

####################################################################
####################################################################

filename = "GoogleNews-vectors-negative300.bin"
W2V_GoogleNews = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    try:
        # update embedding matrix using Google's pretrained model
        embedding_vector = W2V_GoogleNews.get_vector(word, norm=True)
        embedding_matrix[i] = embedding_vector
    except:
        # when word isn't found in pretrained model, we keep the embedding matrix unchanged at that index (assigned to zero)
        pass

####################################################################

# We convert lists to np array
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

####################################################################
####################################################################

print("Training padded shape:", training_padded.shape)
print("Temp_list shape: ", temp_list.shape)
print("Training Label shape: ", training_labels.shape)


print("Testing Label shape: ", testing_labels.shape)
print("Test temp list:", test_temp_list.shape)
print("Testing padded shape:", testing_padded.shape)

####################################################################

# Model
x1 = tf.keras.layers.Input(shape=[max_length, ])
x2 = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(x1)
x3 = tf.keras.layers.Conv1D(64, kernel_size=1)(x2)
x4 = tf.keras.layers.MaxPool1D(2)(x3)

# Bidirectional LSTM
x5 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.3))(x4)

# LSTM
# x5 = tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.3)(x4)

# Bidirectional GRU
# x5 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, dropout=0.5, recurrent_dropout=0.3))(x4)

# # GRU
# x5 = tf.keras.layers.GRU(64, dropout=0.5, recurrent_dropout=0.3)(x4)


y1 = tf.keras.layers.Input(shape=[b, ])
y2 = tf.keras.layers.Dense(20, activation='relu')(y1)

z1 = tf.keras.layers.Concatenate()([x5, y2])
z2 = tf.keras.layers.Dense(25, activation='relu')(z1)
z3 = tf.keras.layers.Dropout(0.2)(z2)
z4 = tf.keras.layers.Dense(1, activation='sigmoid')(z3)

model = tf.keras.Model(inputs=[x1, y1], outputs=z4)

model.summary()

adam = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                beta_1=0.9,
                                beta_2=0.999,
                                amsgrad=False)


model.compile(loss="binary_crossentropy",
              optimizer=adam,
              metrics="accuracy")

my_callback = MyCallback()
history = model.fit([training_padded, temp_list],
                    training_labels,
                    epochs=200,
                    callbacks=[my_callback],
                    validation_split=0.176)

model.summary()

tf.keras.utils.plot_model(
    model,
    to_file='model.png',
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96,
    layer_range=None,
)

####################################################################

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

Model_predictions = np.round_(model.predict([testing_padded, test_temp_list]))
confusion(Model_predictions, testing_labels)

