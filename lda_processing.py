import codecs, re, os
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np

with open('./data/stopwords.txt','r',encoding='utf-8') as f:
    stopwords = f.read().split('\n')

def lda_process(split_text, embedding_size, wordvec):
    new_line, new_dict = [], []
    for line in split_text:
        for w in line.split():
            if w in stopwords: continue
            new_line.append(w)
        new_dict.append(new_line)
        new_line = []

    dictionary = Dictionary(new_dict)

    corpus = [ dictionary.doc2bow(text) for text in new_dict ]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=1,passes=20)

    _, title_terms = lda.print_topics(num_words = 5)[0]

    title_vec = []

    sub_terms = title_terms.split('+')
    for term in sub_terms:
        listItems = term.split('*')
        try:
            title_vec.append(float(listItems[0]) * wordvec[re.findall(r'\"(.+)\"',listItems[1])[0]])
        except KeyError:
            title_vec.append(float(listItems[0]) * np.zeros(embedding_size))
    #print(wordvec[re.findall(r'\"(.+)\"',listItems[1])])

    title_vector = np.average(np.array(title_vec), axis=0)

    return title_vector.reshape(1,300)

