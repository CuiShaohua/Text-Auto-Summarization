import pickle as pkl
import os, jieba, re
import numpy as np
from gensim.models import KeyedVectors
from sents2vec import sents2vec
from preprocess_autosummary.sentence_and_doc import Word, Sentence, Document
from sklearn.metrics.pairwise import cosine_similarity
from lda_processing import lda_process
from keywords_weight import keywords_weight
from knn_smooth import knn_smooth
from my_threading import MyThread

# 恢复标点
def dict_leagal(sents, subsents_dict):
    sub_dict = {}
    sub_sents_set = set()
    m = subsents_dict[0][1]
    sub_dict[subsents_dict[0][0]] = sents[m]

    for j, k in subsents_dict[1:]:

        m += k + 1
        if j not in sub_sents_set:
            sub_sents_set.add(j)
            try:
                if sents[m] not in '。，？！：':
                    print('输入的句子不合法，标点符号缺失')
                else:
                    sub_dict[j] = sents[m]

            except IndexError:
                return "句子不合法"
    return sub_dict


def read_wordvec_pkl(wordvec_file):

    if os.path.exists(wordvec_file):
        f = open(wordvec_file, 'rb')
        wordvec = pkl.load(f)
        f.close()
    else:
        wordvecPath = "./model/sgns.wiki.bigram"
        wordvec = KeyedVectors.load_word2vec_format(wordvecPath)
        with open('./wiki_bigram.vec', 'wb') as f:
            pkl.dump(wordvec, f)
    return wordvec

def read_wordfreq_pkl(freq_file):

    if os.path.exists(freq_file):
        f = open(freq_file, 'rb')
        mydict = pkl.load(f)
        f.close()
    else:
        with open('./model/word_freq', 'rb') as foup:
            mydict = pkl.load(foup)  # return mydict
    return mydict


def time_and_place_match(text, patten):
    for i in text:
        if re.findall(patten, string=i):
            return i
    return 0
    # print('task' + str(patten) + 'has done!')


import time
# main function
def autoSummary(para):

    # pre-pate: load pre-trained wordvec form pkl-file

    wordvec_file = './model/wiki_bigram.vec'
    read_wordvec = MyThread(func=read_wordvec_pkl, args=(wordvec_file,))
    # read freq_file
    freq_file = "word_freq"
    read_wordfreq = MyThread(func=read_wordfreq_pkl, args=(freq_file,))

    read_wordvec.start()
    read_wordfreq.start()
    read_wordvec.join()
    read_wordfreq.join()

    wordvec = read_wordvec.get_result()
    mydict = read_wordfreq.get_result()

    # 0 提示用户输入文章
    #sentence = input("请输入文档：")
    sentence = para
    pattern = re.compile(r'[。，？！：]')
    start = time.time()
    # 提取关键词
    keywords = keywords_weight(sentence)

    # split是包含子句子的结果，也是程序的入口
    split = pattern.sub(' ', sentence).split()

    # 如果句子条数小于3，那么直接返回该句子
    if len(split) <=3:
        return sentence
    # 建立标点字典
    subsents_dict = [(w, len(w)) for w in split]
    sub_dict = dict_leagal(sentence, subsents_dict)

    # 分词
    sent_split = [' '.join(jieba.cut(n, cut_all=False)) for n in split if n]  # 存储的句子列表  --> 已经分词之后的结果

    # 从这开始分支，PCA 和 LDA分别进行
    # PCA
    embedding_size = 300
    word_set, sents_set = [], []

    for sent in sent_split:
        for word in sent.split():
            try:
                if word in keywords.keys():
                    word_set.append(Word(word, wordvec[word]* keywords[word]))
                else:
                    word_set.append(Word(word, wordvec[word]))
            except KeyError:
                word_set.append(Word(word, np.zeros(embedding_size)))
        sents_set.append(Sentence(word_set))
        word_set = []
    # 句子向量sentvec_set 文章向量
    doc_vec = Document(sents_set)
    # 1 输出句子向量和文章向量的余弦相似度
    sents_vector = sents2vec(sents_set, embedding_size, mydict)
    doc_vector  = sents2vec([doc_vec], embedding_size, mydict)
    # 相似度，子句子和文章的相似度
    pca_cos_similarity = cosine_similarity(sents_vector,Y=doc_vector)
    L = pca_cos_similarity.reshape(-1)
    # 2 提炼主题句子  LDA
    lda_vector = lda_process(split_text=sent_split,embedding_size=embedding_size,wordvec=wordvec)
    lda_cos_similarity = cosine_similarity(sents_vector,Y=lda_vector)
    M = lda_cos_similarity.reshape(-1)
    # 3 KNN平滑
    #  加权相似度。0.5 * sub句子/文章向量 + （1-0.5）* sub句子/主题向量
    pca_weight = 0.6
    total_similarity = pca_weight * knn_smooth(L) + (1-pca_weight) * knn_smooth(M)

    # 4 句子重组和输出
    # 4.1 考虑首句和尾句，此处没有考虑首句和尾句的重要作用，只考虑了是谁报道，什么时间发生
    time_patten = r'[.+月.+日|当地时间.+|北京时间.+|.+月|.+日]'
    place_patten = r'[.+电，|据.+报道]'
    time_match = time_and_place_match(split[:3], time_patten)
    place_match = time_and_place_match(split[:3], place_patten)
    # 4.2 检测字数
    word_num = 80
    # 4.3 标点的恢复。
    t_initial = 0.4  # similarity threshoud initial
    t_end = 0.45
    biaodian = []
    for i in range(len(total_similarity)):
        if total_similarity[i] >= t_initial:
            sents_result = ''.join([j.text for j in sents_set[i].word_list])
            #print(sents_result, end=sub_dict[sents_result])
            biaodian.append((sents_result, sub_dict[sents_result]))

    try:

        if time_match != 0 and np.sum((np.array([i[0] for i in biaodian]) == time_match).astype('float')) == 0.:
        #if np.sum((np.array([i[0] for i in biaodian]) == time_match).astype('float')) == 0.:
            biaodian.insert(split.index(time_match), (time_match, sub_dict[time_match]))
        if place_match != 0 and np.sum((np.array([i[0] for i in biaodian]) == place_match).astype('float')) == 0.:
            biaodian.insert(split.index(place_match), (place_match, sub_dict[place_match]))
        '''
        if np.sum((np.array([i[0] for i in biaodian]) == place_match).astype('float')) == 0.:
            biaodian.insert(split.index(place_match), (place_match, sub_dict[place_match]))
        '''
    except Exception as E:
        print(E)
    #for Iterms in biaodian:
    return ''.join([iterm[0]+iterm[1] for iterm in biaodian])

def main():
    return autoSummary()

if __name__ == '__main__':
    print(main())
