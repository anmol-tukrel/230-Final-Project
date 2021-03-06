
import numpy as np
import string
import pandas as pd
from textblob import TextBlob
#import nltk.corpora
import nltk
#from nltk.corpora import wordnet as wn
from collections import Counter

def addPolarity():

    path = 'coronaCombinedShuffled.csv'

    data = pd.read_csv(path)

    data['text'] = data['text'].apply(str)

    polarities = []
    subjectivities = []
    distances = []
    adjCount = []

    for i in range(len(data)):
        text = TextBlob(data.loc[i, 'text'])
        textPol = text.sentiment.polarity
        polarities.append(textPol)
        textSub = text.sentiment.subjectivity
        subjectivities.append(textSub)
        #print('textpol: ' +  str(textPol))
        #print('sub: ' + str(textSub))
#        print(type(data.loc[i, 'title']))
        tokensHeadline = nltk.word_tokenize(str(data.loc[i, 'title']))
        #print(text)
        tokensText = nltk.word_tokenize(str(text))

        jdistance = nltk.jaccard_distance(set(tokensHeadline), set(tokensText))
        print('jdist ' + str(jdistance))
        distances.append(jdistance)

        adjCount.append(countAdjectives(text))

    data['polarity'] = polarities
    data['subjectivity'] = subjectivities
    data['jaccarddistance'] = distances
    data['adjectivecount'] = adjCount

    data.to_csv('coronaDatasetWithFeatures.csv')


def countAdjectives(text):

    #adjectivesList = pd.read_csv('adjectives-list.csv')
#    numAdjectives = 0
#    adj = TextBlob.adjective_phrases
    #for word in text:
     #   if word in adjectivesList['adjectives']:
      #      numAdjectives += 1

    adj_list = []
    adj_tag_list = ['JJ','JJR','JJS']
#    print(text)
    blobed = text
    counts = Counter(tag for word,tag in blobed.tags)
    for (a, b) in blobed.tags:
        if b in adj_tag_list:
            adj_list.append(a)
    
    return len(adj_list)

addPolarity()
