
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

def main():
    bern_text_data = []
    bern = pd.read_csv('Bern.csv',encoding = "ISO-8859-1")
    for key,value in bern.iterrows():
        bern_text_data.append(value[2].split())

    for sentence_index in range(len(bern_text_data)):
        for word_index in range(len(bern_text_data[sentence_index])):
            if '\\n' in bern_text_data[sentence_index][word_index]:
                bern_text_data[sentence_index][word_index] = bern_text_data[sentence_index][word_index][:-2]



    for sentence_index in range(len(bern_text_data)):
        bern_text_data[sentence_index] = " ".join(bern_text_data[sentence_index])

    #print(bern_text_data)

    # para = "Hello good friend"
    # lines_list = tokenize.sent_tokenize(para)
    # print(lines_list)

    sid = SentimentIntensityAnalyzer()

    # for sentence in bern_text_data:
    #     print(sentence)
    #     ss = sid.polarity_scores(sentence)
    #     for k in sorted(ss):
    #         print('{0}: {1}, '.format(k, ss[k]), end='')
    #         print("")


    ss = sid.polarity_scores("bern")
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')






main()