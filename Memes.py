import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import io
import urllib
#import cv2
#from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import copy
import numpy as np

def cleanData(data):
    data = data[~(data == 0).any(axis=1)]
    text_data = []

    for key, value in data.iterrows():
        text_data.append(value[3].split())


    for sentence_index in range(len(text_data)):
        for word_index in range(len(text_data[sentence_index])):
            if '\\n' in text_data[sentence_index][word_index]:
                text_data[sentence_index][word_index] = text_data[sentence_index][word_index][:-2]
            if "IG:" in text_data[sentence_index][word_index]:
                text_data[sentence_index] = text_data[sentence_index][:word_index]
                break
            if word_index == len(text_data[sentence_index]) - 1:
                if "@" in text_data[sentence_index][word_index]:
                    text_data[sentence_index].remove(text_data[sentence_index][word_index])

    for sentence_index in range(len(text_data)):
        text_data[sentence_index] = " ".join(text_data[sentence_index])

    return [text_data,data]


def getSent(text_data):
    sentiment_data = []



    sid = SentimentIntensityAnalyzer()
    for sentence in text_data:
        temp_sentiment_data = []
        ss = sid.polarity_scores(sentence)
        for k in sorted(ss):
            temp_sentiment_data.append(ss[k])

        sentiment_data.append(temp_sentiment_data)

    return sentiment_data
    #compound negative neutral positive

def normalizeScores(data):
    min_val = 0
    for key, value in data.iterrows():
        if value[6] < min_val:
            min_val = value[6]
    min_val = abs(min_val) + 1

    normalized_scores = []
    min_imgur = 9999
    max_imgur = 0
    min_facebook = 9999
    max_facebook = 0
    min_instagram = 9999
    max_instagram = 0
    min_twitter = 9999
    max_twitter = 0

    for key, value in data.iterrows():
        if value[5] == "imgur":
            if math.log(value[6]+min_val) < min_imgur:
                min_imgur = math.log(value[6]+min_val)

            elif math.log(value[6]+min_val) > max_imgur:
                max_imgur = math.log(value[6]+min_val)

        elif value[5] == "facebook":
            if math.log(value[6]+min_val) < min_facebook:
                min_facebook = math.log(value[6]+min_val)

            elif math.log(value[6]+min_val) > max_facebook:
                max_facebook = math.log(value[6]+min_val)

        elif value[5] == "instagram":
            if math.log(value[6]+min_val) < min_instagram:
                min_instagram = math.log(value[6]+min_val)

            elif math.log(value[6]+min_val) > max_instagram:
                max_instagram = math.log(value[6]+min_val)

        elif value[5] == "twitter":
            if math.log(value[6]+min_val) < min_twitter:
                min_twitter = math.log(value[6]+min_val)

            elif math.log(value[6]+min_val) > max_twitter:
                max_twitter = math.log(value[6]+min_val)

    imgur_distance = float(max_imgur - min_imgur)
    facebook_distance = float(max_facebook - min_facebook)
    instagram_distance = float(max_instagram - min_instagram)
    twitter_distance = float(max_twitter - min_twitter)

    for key, value in data.iterrows():
        if value[5] == "imgur":
            temp_score = abs(math.log(value[6]+min_val) - min_imgur) / imgur_distance
            normalized_scores.append(temp_score)
        elif value[5] == "facebook":
            temp_score = abs(math.log(value[6]+min_val) - min_facebook) / facebook_distance
            normalized_scores.append(temp_score)
        elif value[5] == "instagram":
            temp_score = abs(math.log(value[6]+min_val) - min_instagram) / instagram_distance
            normalized_scores.append(temp_score)
        elif value[5] == "twitter":
            temp_score = abs(math.log(value[6]+min_val) - min_twitter) / twitter_distance
            normalized_scores.append(temp_score)

    return normalized_scores


def qual2quant(normalized):
    new_scores = []
    sorted_norm = copy.copy(normalized)
    sorted_norm.sort()
    max_length = len(sorted_norm)
    bin_size = int(max_length/5)
    first_20 = sorted_norm[bin_size]
    second_20 = sorted_norm[bin_size*2]
    third_20 = sorted_norm[bin_size*3]
    fourth_20 = sorted_norm[bin_size*4]



    for score in normalized:
        if score < first_20:
            new_scores.append(0)
        elif first_20 <= score < second_20:
            new_scores.append(1)
        elif second_20 <= score < third_20:
            new_scores.append(2)
        elif third_20 <= score < fourth_20:
            new_scores.append(3)
        elif fourth_20 <= score:
            new_scores.append(4)

    return new_scores











def main():

    sid = SentimentIntensityAnalyzer()

    #data = pd.read_csv('Bern.csv', encoding="ISO-8859-1")
    data = pd.read_csv('hillary memes.csv', encoding="ISO-8859-1")

    cleaned_text = cleanData(data)[0]
    cleaned_data = cleanData(data)[1]

    normalized = normalizeScores(cleaned_data)
    normalized_bin = qual2quant(normalized)

    sent_data = getSent(cleaned_text)





    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='softplus'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softplus'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softplus'))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(np.array(sent_data), np.array(normalized_bin), epochs=30, batch_size=20)
    scores = model.evaluate(np.array(sent_data), np.array(normalized_bin))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))





main()