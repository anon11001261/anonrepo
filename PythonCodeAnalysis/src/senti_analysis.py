# modules used
import os
os.chdir('')
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from prep_comments import prep_comments, raw_comments, comment_dir
from generate_dictionary import gen_pos, gen_bigram_dict, dict_pos_map
#import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer

comment_dir = comment_dir
data_dir = ''
# read peer review (experimentation purposes only)
def read_peer_data():
    # read in data from dir input
    with open(data_dir, 'r') as file:
        df = pd.read_csv(file, index_col=0)

    return df

def get_fdist():
    """ This functions lemmatizes and calculates the frequency in which words are found
    within the comment list.
    Returns:
        FreqDist: A frequency distribution of the lemmatized comments
    """
    # shorten words
    pos_comments = gen_pos()

    lemmatizer = WordNetLemmatizer()
    lemmatized_comments = []
    for comment in pos_comments:
        lemmatized_comment = []
        for tuple in comment:
            tmp = tuple[0]
            if tuple[1] == 'NNP' or tuple[1] == 'NNPS':
                continue
            if tuple[1][:2] in dict_pos_map.keys():
                tmp = lemmatizer.lemmatize(tuple[0], pos=dict_pos_map[tuple[1][:2]])
            lemmatized_comment.append(tmp)
        lemmatized_comments.append(lemmatized_comment)

    # generate overall frequency distribution
    fdist = nltk.FreqDist(word for words in lemmatized_comments for word in words)

    return fdist

def get_sentiments():
    """ This functions calculates the sentiment of the tokens generated from the comments
    using vader's 'SentimentIntensityAnalyzer'. The compound polarities are then used to
    store each of the comments sentiment (positive, negative, neutral, and negating).
    
    In progress: store the total length of each comment.
    Returns:
        DataFrame: A pandas DataFrame containing the full string comment, its tokenized
        version, and the sentiment determined by the sentiment analyzer.
    """
    # store full comments for polarities
    comments = raw_comments()

    # store tokenized comment list
    tk_comments = prep_comments()
    
    # store bigram dictionary
    bigram_dict = gen_bigram_dict()

    vader = SentimentIntensityAnalyzer()
    # store comment polarities
    pos_count = 0
    neg_count = 0
    neu_count = 0
    # store a list of comment lengths
    # Note: update in the future to read by student
    c_length_list = []
    # negating count with created dictionary

    # calculate and store comments length and polarity scores
    sentiment_levels = []
    for c in comments:
        comment_length = 0
        for comment in c:
            comment_length += len(comment)
            sentiment = vader.polarity_scores(comment)

            if sentiment['compound'] >= 0:
                pos_count += 1
                sentiment_levels.append('positive')

            elif sentiment['compound'] <= 0:
                neg_count += 1
                sentiment_levels.append('negative')

            else:
                neu_count += 1
                sentiment_levels.append('neutral')
        c_length_list.append(comment_length)

    full_comments = [item for comment in comments for item in comment]
   
    senti_data = pd.DataFrame(list(zip(full_comments, tk_comments, bigram_dict, sentiment_levels)),\
        columns = ['Full Comment','Tokenized Comments','Bigrams','Sentiment'])
    print(senti_data)
    
    senti_data.to_csv('data/senti_data.csv', index=False, header=True)
    
    return senti_data

read_peer_data()
get_fdist()
get_sentiments()

#def overall_sentiment():

## Group overall sentiments based on project letter grades and POS used
# Note: randomize comment sentiments when grouping by letter grade received
#def grouped_sentiments():
