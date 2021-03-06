import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter
import os
import getEmbeddings2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd


top_words = 5000
epoch_num = 8
batch_size = 25

def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    #cm = confusion_matrix(yte, ypred, labels=[0,1])
    #print(cm)
    report = classification_report(yte, ypred, labels=[0,1])
    print(report)
    cm = confusion_matrix(yte, ypred, labels=[0,1])
    print(cm)
    df_cm = pd.DataFrame(cm)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.savefig('baseline-cm-lstm-custom-8e-25b-nojacquard-anmol.png')
  #  svm = sns.heatmap(df_cm, annot=True, cmap='Blues', linecolor='white', linewidths=1)
#    figure = svm.get_figure()
 #   figure.savefig('matrix.png')
 
 
#if not os.path.isfile('xtr_shuffled.npy') or \
 #   not os.path.isfile('xte_shuffled.npy') or \
  #  not os.path.isfile('ytr_shuffled.npy') or \
   # not os.path.isfile('yte_shuffled.npy'):

getEmbeddings2.clean_data()

xtr = np.load('xtr_shuffled.npy', allow_pickle=True)
xte = np.load('xte_shuffled.npy', allow_pickle=True)
y_train = np.load('ytr_shuffled.npy', allow_pickle=True)
y_test = np.load('yte_shuffled.npy', allow_pickle=True)

cnt = Counter()
x_train = []
for x in xtr:
    print(x)
    x_train.append(str(x).split())
    for word in x_train[-1]:
        cnt[word] += 1

# Storing most common words
most_common = cnt.most_common(top_words + 1)
word_bank = {}
id_num = 1
for word, freq in most_common:
    word_bank[word] = id_num
    id_num += 1

# Encode the sentences
for news in x_train:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]

y_train = list(y_train)
y_test = list(y_test)

# Delete the short news
i = 0
while i < len(x_train):
    if len(x_train[i]) > 10:
        i += 1
    else:
        del x_train[i]
        del y_train[i]

# Generating test data
x_test = []
for x in xte:
    x_test.append(str(x).split())

# Encode the sentences
for news in x_test:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]


# Truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# Convert to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Create the model
embedding_vecor_length = 64
model = Sequential()
model.add(Embedding(top_words+2, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_num, batch_size=batch_size)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy= %.2f%%" % (scores[1]*100))

# Draw the confusion matrix
y_pred = model.predict_classes(X_test)
plot_cmat(y_test, y_pred)
