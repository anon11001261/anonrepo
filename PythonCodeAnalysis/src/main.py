# modules used
import os
os.chdir('')
import csv
import nltk
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV
dict_pos_map = {
    'NN': NOUN,
    'VB': VERB,
    'JJ' : ADJ,
    'RB': ADV
}
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from gensim.models import TfidfModel
from gensim.corpora.dictionary import Dictionary
from prep_comments import prep_comments, comment_dir # import preprocessing module/script

comment_dir = comment_dir
# generate Parts-of-Speech mapped to tokenized comments
def gen_pos():
    # store comments
    comments = prep_comments()
    # create list with pos tagged words from comments
    comments = ([nltk.pos_tag(comment) for comment in comments])

    return comments

def gen_bigram_dict():
    """ This functions generates a bigram dictionary mapped to its
    number of instances found.
    Returns:
        tuple: A tuple dictionary of trigram tokens with the frequency as its value.
    """
    # lemmatize(shorten) words
    pos_comments = gen_pos()

    lemmatizer = WordNetLemmatizer()
    lemmatized_comments = []
    for comment in pos_comments:
        lemmatized_comment = []
        for tuple in comment:
            tmp = tuple[0]
            if tuple[1].startswith('NN'):
                continue
            if tuple[1][:2] in dict_pos_map.keys():
                tmp = lemmatizer.lemmatize(tuple[0], pos=dict_pos_map[tuple[1][:2]])
            lemmatized_comment.append(tmp)
        lemmatized_comments.append(lemmatized_comment)

    # inizialize dictionary
    dict = Dictionary(lemmatized_comments)
    
    # create corpus vocab w/ the tokenized comments
    corpus = [dict.doc2bow(doc, allow_update=True) for doc in lemmatized_comments]
    tfidf = TfidfModel(corpus, smartirs='ntc')

    # tf-idf for setting importance values to comments
    tfidf_corpus = tfidf[corpus]
        
    # store keys over 0.1
    word_dict = {dict.get(id): round(value, ndigits=1) for doc in tfidf_corpus for id,\
        value in doc if value > 0.1}

    # store sorted word dictionary in data folder
    rows = []
    with open('data/FriedmanDictionary.csv', newline="") as d:
        reader = csv.reader(d)
        for row in reader:
            rows.append(row)
    for key, val in sorted(word_dict.items()):
        val = round(val, ndigits=1)
        rows.append([key, val])
    with open('data/final_peer_dictionary.csv', 'w') as f: # use filecmp to replace old csv
        w = csv.writer(f)
        # sort rows before writing: not yet applied
        w.writerows(rows)
    
        
    # split each comment into bigrams along with the number of ngrams stored
    bigram_dict = [list(ngrams(comment, 2)) for comment in lemmatized_comments]

    bigram_dict = list(filter(None, bigram_dict))

    return bigram_dict

gen_pos()
gen_bigram_dict()
