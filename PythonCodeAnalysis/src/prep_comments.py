# run django as default server

import os
os.chdir('')
import re
import nltk
from nltk.corpus import stopwords
from textblob import Word

comment_dir = ''

def raw_comments():
    """ This function is used to read and store full peer review comments.
    Returns:
        list: A list of stored comments
    """
    # use default path or user's file location
    try:
        file_name = str(comment_dir or input("Enter student comments' directory: "))
    except FileNotFoundError:
        file_name = comment_dir
    
    # read comments file from input
    with open(file_name, 'r') as file:
        lines = file.readlines()
        comments = []
        for line in lines:
            comments.append(line.strip().replace('\n', ' ').replace('"', ' ')\
                .strip().split(';'))
    
    comment_list = []
    for comments in comments:
        for comment in comments:
            comment_list.append(re.sub(r'[^a-zA-Z]', ' ', comment).split(';'))

    return comment_list

def prep_comments():
    """ The following function read, stores, pre-processes, and tokenizes peer review comments.
    Preprocessing consists of regular expression removal, word validation, and stop word removal.
    Returns:
        list: A nested list of tokenized words from comments
    """
    full_comments = raw_comments()
    # clean comments, allowing only text
    comment_list = []
    for comments in full_comments:
        for comment in comments:
            comment_list.append(re.sub(r'[^a-zA-Z]', ' ', comment).split(';'))
    
    # tokenize and store real dictionary words
    tk_comments = []
    for comments in comment_list:
        for comment in comments:
            tmp = []
            comment = nltk.word_tokenize(comment)
            for word in comment:
                word = Word(word.lower())
                result = word.spellcheck()
                if word == result[0][0]:
                    tmp.append(word)
            tk_comments.append(tmp)
    
    # remove stop words using updated set
    stop_words = set(stopwords.words('english'))
    stop_words.update(['none','submission','wa','doe','rstudio','r','html','rdr','qa','ha','rd','etc','pdf','cod','ggplot','readme','professor'])
    no_stops = [list(set(comment).difference(stop_words)) for comment in tk_comments]
    # remove empty lists (Source: https://www.geeksforgeeks.org/python-remove-empty-list-from-list/)
    no_stops = list(filter(None, no_stops))

    return no_stops

raw_comments()
prep_comments()
