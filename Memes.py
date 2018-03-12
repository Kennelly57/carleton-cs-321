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
import re
from sklearn.cluster import KMeans

def cleanData(data):
    data = data[~(data == 0).any(axis=1)]
    text_data = []

    for key, value in data.iterrows():
        text_data.append(value[3].split())


    for sentence_index in range(len(text_data)):
        for word_index in range(len(text_data[sentence_index])):
            if '\\n' in text_data[sentence_index][word_index]:
                text_data[sentence_index][word_index] = text_data[sentence_index][word_index][:-2]
    for sentence_index in range(len(text_data)):
        text_data[sentence_index] = " ".join(text_data[sentence_index])


    inds_to_remove = []
    for sentence_ind in range(len(text_data)):
        matchObj1 = re.search('([0-9]|[0-9][0-9]):[0-9][0-9]',text_data[sentence_ind])
        matchObj2 = re.search('[0-9]{5,}', text_data[sentence_ind])
        matchObj3 = re.search('[0-9]+/[0-9]+/[0-9]+', text_data[sentence_ind])
        matchObj4 = re.search('www\..*\.com', text_data[sentence_ind])
        matchObj5 = re.search('@', text_data[sentence_ind])
        matchObj6 = re.search('IG:', text_data[sentence_ind])
        if matchObj1 or matchObj2 or matchObj3 or matchObj4 or matchObj5 or matchObj6:
            inds_to_remove.append(sentence_ind)

    inds_to_remove.reverse()

    for ind in inds_to_remove:
        del text_data[ind]
        data.drop(data.index[ind],inplace=True)

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


def getTime(data):
    date_list = []
    for key, value in data.iterrows():
        time = value[0].split()[1]
        time = time.replace(":","")
        date_list.append(time)

    return date_list


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


def qual2quant(normalized, size):
    new_scores = []
    sorted_norm = copy.copy(normalized)
    sorted_norm.sort()
    max_length = len(sorted_norm)

    if size == 2:
        bin_size = int(max_length/2)
        first_half = sorted_norm[bin_size]
        for score in normalized:
            if score < first_half:
                new_scores.append(0)
            else:
                new_scores.append(1)

    if size == 5:
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



def dataCombiner(sent_data, dates):
    for sent_ind in range(len(sent_data)):
        sent_data[sent_ind].append(dates[sent_ind])

    return sent_data


def cluster(sent_data, normalized):
    compound = []
    for sentiment in sent_data:
        compound.append(sentiment[0])
    df = pd.DataFrame({'x': compound, 'y': normalized})

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(df)

    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_

    print(labels)
    print(centroids)

    fig = plt.figure(figsize=(5, 5))

    plt.scatter(df['x'], df['y'], c=kmeans.labels_, cmap='rainbow')
    plt.xlim(-1.05,1.05)
    plt.ylim(-.05, 1.05)
    plt.xlabel("Compound Sentiment Score")
    plt.ylabel("Normalized Score")
    plt.title("2 Means Clustering Donald Trump")
    plt.show()






def main():

    data = pd.read_csv('trump memes.csv', encoding="ISO-8859-1")
    [cleaned_text,cleaned_data] = cleanData(data)

    getTime(cleaned_data)
    normalized = normalizeScores(cleaned_data)
    normalized_bin = qual2quant(normalized,2)

    sent_data = getSent(cleaned_text)
    dates = getTime(cleaned_data)

    #cluster(sent_data,normalized)


    combined_data = dataCombiner(sent_data,dates)

    model = Sequential()
    model.add(Dense(12, input_dim=5, activation='softplus'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softplus'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softplus'))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(np.array(sent_data), np.array(normalized_bin), epochs=30, batch_size=20)
    scores = model.evaluate(np.array(combined_data), np.array(normalized_bin))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))





main()