# AutoSummarizzation
#####  text zuto summarization for news 
### 0 写在前
>>  <font size=16> &ensp;&ensp; 目前不论是大小云厂商还是提供SaaS服务的AI企业，大多推出了文本自动摘要这一功能。华为云去年之前还未做这个功能，但目前已经在EI里推出了这个功能。笔者在根据开课吧学习AI知识之后，将文本自动摘要采用无监督的方式进行实现，并且结合Flask进行项目展示。  
>>    &ensp;&ensp;自有文本摘要的研究以来，研究方法大致分为两大类，一种是生成式的文本摘要，另一种是抽取式文本摘要。对于中文训练集来说，由于尚且没有诸如英语文摘的训练集，所以很少能见到有关学者是采用生成式的方法进行中文的摘要训练；而对于英语生成式文本摘要代表作，可以查看[A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685)，引用原文一句```“Summarization based on text extraction is inherently limited, but generation-style abstractive methods have proven challenging to build. In this work, we propose a fully data-driven approach to abstractive sentence summarization. Our method utilizes a local attention-based model that generates each word of the summary conditioned on the input sentence. While the model is structurally simple, it can easily be trained end-to-end and scales to a large amount of training data. The model shows significant performance gains on the DUC-2004 shared task compared with several strong baselines.”```，可以看出深度学习方法的生成式存在着抽取式无法比拟的众多优势。笔者个人认为，生成式方法虽然目前存在一些尚未改进的地方（a.采用RNN结构，训练速度慢；b.深度学习框架不容易调测；c.深度学习框架依赖于训练集，训练集若集中于某个行业领域，则训练出的模型的普适性不高），但生成式方法一定是文本摘要的未来（a. 生成的句子更灵活，不用写一堆“蹩脚”的规则；b. 引入高层语义，更像人的思考之后说过的话）。  </font>
____
* 开始介绍本篇  
### 1 项目代码逻辑结构  
>> 在上代码之前，我们先思考几个问题：  
>>> （1）默认共识——摘要的句子都是这篇文本中重要的句子，那么怎么衡量这个句子的重要性？  
>>> （2）一篇文章总有一个主题，主题句在摘要中发挥的作用怎么体现？  
>>> （3）文章中的关键词做不做考虑，怎么找到关键词？  
>>> （4）如果上述的问题都解决了（问题1用PCA、问题2用LDA、问题3用TextRank），那么如何协调每种“摘要方法”之间的关系？分别配置的权重是多少？  
>>> （5）另外，结合项目针对的新闻领域，如果新闻本身是有标题的，那么标题是不是要考虑进去；并且输出结果还要符合新闻类文本摘要的一个基本要求，发生时间，人物（主人公）等需要考虑进去。  
>>> （6）笔者能够考虑的最后一点，需不需要设置字数限制？  

### 2 实现代码
____
* 当我们将上述的几个问题分别解决后，该项目也就完成的差不多了。  
____
&ensp;&ensp; 首先，我们先把每个句子进行一个多维的向量表征（主流使用使用词向量，可以是基于Word2vec，cove， glove，bert等一些主流的词向量训练方法），当然采用keras进行训练的话，也可以不用获取词向量，让其自动训练一个Embedding层即可，但本文采用的是已经训练好的模型（语料来源于中文wiki百科和搜狗新闻语料）[另外也可参考连接](https://github.com/Embedding/Chinese-Word-Vectors)，该种方法在深度学习上也叫采用预训练模型的方式。  

#### 2.1 如何表征一个句向量，并且怎么衡量该句向量对文章向量的重要程度？
&ensp;&ensp; 目前已经获取句向量的方法，常见的获取句向量的是加权句子包含词向量的思想，简单粗暴的有累加、平均、TF-IDF做权值和ISF加权。本文使用的是普林斯顿研究的方法，属于ISF方法，具体可以参考论文[A Latent Variable Model Approach to PMI-based Word Embeddings](https://arxiv.org/abs/1502.03520)，该论文17年发表，在有监督学习盛行的情况下，仍能曾获得业内的较高评价，可见方法可以获得较高的准确度。该方法需要在PCA的基础上求解协方差矩阵的第一特征向量作为句子向量的代表。具体PCA的原理，可以参考我的一篇[文章](https://github.com/CuiShaohua/NLP-/blob/master/%E6%89%8B%E5%86%99%E5%AE%9E%E7%8E%B0%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90PCA.ipynb)。当然也可以借助sklearn上的PCA方法，但是需要注意的是，如果把文章想句子一样看做是一个特别长的句子，那么PCA计算协方差贡献率的时候需要进行更改，更改的部分贴在下方：
```Python
        # Get variance explained by singular values
        if n_samples != 1:
            explained_variance_ = (S ** 2) / (n_samples - 1)
        else:
            explained_variance_ = (S ** 2) / n_samples
            # explained_variance_ is zero
        total_var = explained_variance_.sum()
        #print(total_var)
        if total_var != 0:
            explained_variance_ratio_ = explained_variance_ / total_var

        else:
            explained_variance_ratio_ = np.ones_like(explained_variance_)
```
>> *  向量法盛行天下开始，在计算广告学等领域，推荐系统的本身是相似度和排序的使用，那么上述PCA将单个句子和单个文本都拿做一个向量来处置是十分恰当的，PCA降维之后，我们能够获得协方差矩阵第一特征向量，其在图上相当于做了某个方向的取向，句子向量与文本向量的余弦相似度越小，那么说明两者的取向相近，用这种似有监督却无监督的方法可以衡量每个句子对文本向量取向的贡献程度，也就是第一个问题解决了。  
#### 2.2 借助LDA求取文章的主题句
>> [LDA是什么？](https://zhuanlan.zhihu.com/p/31470216)， LDA是一种有监督的数据降维方法。我们知道即使在训练样本上，我们提供了类别标签，在使用PCA模型的时候，我们是不利用类别标签的，而LDA在进行数据降维的时候是利用数据的类别标签提供的信息的。 
从几何的角度来看，PCA和LDA都是讲数据投影到新的相互正交的坐标轴上。只不过在投影的过程中他们使用的约束是不同的，也可以说目标是不同的。PCA是将数据投影到方差最大的几个相互正交的方向上，以期待保留最多的样本信息。样本的方差越大表示样本的多样性越好，在训练模型的时候，我们当然希望数据的差别越大越好。否则即使样本很多但是他们彼此相似或者相同，提供的样本信息将相同，相当于只有很少的样本提供信息是有用的。样本信息不足将导致模型性能不够理想。这就是PCA降维的目标：将数据投影到方差最大的几个相互正交的方向上。而LDA是将带有标签的数据降维，投影到低维空间同时满足三个条件：尽可能多地保留数据样本的信息（即选择最大的特征是对应的特征向量所代表的的方向）。寻找使样本尽可能好分的最佳投影方向。投影后使得同类样本尽可能近，不同类样本尽可能远。具体实现代码可以参考我的一篇[文章]()。部分关键代码如下：  
```Ptthon  

```
### 2.3 关键词textRank  
>> 想必做NLP方向的各位同袍，应该对TextRank比较熟悉，TextRank可以看做是PageRank的2.0版本，而[PageRank](http://pr.efactory.de/e-pagerank-algorithm.shtml)可是谷歌发家致富的重要基石，通过爬取大量网页被引用指向，构成G引用矩阵，最终计算G矩阵的特征向量（每个元素代表一个网页的PageRank）$$ G=S*a+（1-a）*U*\frac{1}{n} $$。那么我们来看下jieba库中的TextRank如何实现的  
```Python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from __future__ import absolute_import, unicode_literals
import sys
from operator import itemgetter
from collections import defaultdict
import jieba.posseg
from .tfidf import KeywordExtractor
from .._compat import *
 
 
class UndirectWeightedGraph:
    d = 0.85
 
    def __init__(self):
        self.graph = defaultdict(list)
 
    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))
 
    def rank(self):
        ws = defaultdict(float)
        outSum = defaultdict(float)
 
        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)
 
        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())
        for x in xrange(10):  # 10 iters
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s
 
        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])
 
        for w in itervalues(ws):
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w
 
        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)
 
        return ws
 
 
class TextRank(KeywordExtractor):
 
    def __init__(self):
        self.tokenizer = self.postokenizer = jieba.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.pos_filt = frozenset(('ns', 'n', 'vn', 'v'))
        self.span = 5
 
    def pairfilter(self, wp):
        return (wp.flag in self.pos_filt and len(wp.word.strip()) >= 2
                and wp.word.lower() not in self.stop_words)
 
    def textrank(self, sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
        """
        Extract keywords from sentence using TextRank algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v'].
                        if the POS of w is not in this list, it will be filtered.
            - withFlag: if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        self.pos_filt = frozenset(allowPOS)
        g = UndirectWeightedGraph()
        cm = defaultdict(int)
        words = tuple(self.tokenizer.cut(sentence))
        for i, wp in enumerate(words):
            if self.pairfilter(wp):
                for j in xrange(i + 1, i + self.span):
                    if j >= len(words):
                        break
                    if not self.pairfilter(words[j]):
                        continue
                    if allowPOS and withFlag:
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp.word, words[j].word)] += 1
 
        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()
        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)
 
        if topK:
            return tags[:topK]
        else:
            return tags
 
    extract_tags = textrank

```
* 读到这里，我们可以发现，已经将底层的部分全部做完了。在上层之后，我们该思考，如何确定将这些表征句子权重、主题打分、关键词进行一个有效的结合？  
#### 2.4 加权是王道，但又如何控制一个有效的权重？  
>>> 这个问题其实不好解答，笔者在写项目的过程中也是求教了几位行业人士，但在权重配置上，很难有一个足够精确的方法，是给予sentence vector的权重高一些，还是给LDA的权重高一些，很难评判。那么人难以评判，机器能不能来做评判？（机器学习的思想），答案是否定的，据我了解，目前对摘要的准确度评测方法大多还没有公开见效（中文）。但异想天开一把，如果翻译模型足够强大，将现有的英文摘要（基于深度学习的方法）训练一个模型，然后将该模型（先可以理解为Encoder-Decoder模型），再通过迁移学习，是否有可能嫁接到中文摘要的test集上，当然本文调测的方法还是以具有代表性的新华社新闻稿当做validation集，选择50-100个新闻稿，最终将Sentence Vector权重定为0.6，而主题Vector的权重定为0.4来进行预测。

#### 2.5 其他容易被忽视的小问题   

>>> * 新闻稿总有个人物、时间；  

>>> * 新闻稿也总有个字数的缩句要求（但本文还是主张自由开放一些，没有写规则定义字数限制这一点。）  

*** 最后献上主代码程序  
```Python
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
        wordvecPath = "D:\AI\Jupyter\sgns.wiki.bigram/sgns.wiki.bigram"
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
        with open('./word_freq', 'rb') as foup:
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
def autoSummary():

    # pre-pate: load pre-trained wordvec form pkl-file

    wordvec_file = 'wiki_bigram.vec'
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
    sentence = input("请输入文档：")
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

```
### 3 前端展示部分  

#### 3.1 目录结构
.
├── app.py
├── index.html
├── __init__.py
├── keywords_weight.py
├── knn_smooth.py
├── lda_processing.py
├── main_pro2.py
├── model
│   ├── wiki_bigram.vec
├── myflask
├── my_threading.py
├── nohup.out
├── pickup.py
├── preprocess_autosummary
│   ├── get_word_frequency.py
│   ├── __init__.py
│   └── sentence_and_doc.py
├── sents2vec.py
├── static
│   ├── backgroup.png
│   ├── css
│   │   ├── 5db11ff4gw1e77d3nqrv8j203b03cweg.jpg
│   │   ├── bower.json
│   │   ├── chat.css
│   │   ├── h.jpg
│   │   ├── hm.js.下载
│   │   ├── host.jpg
│   │   ├── h.png
│   │   ├── hs.jpg
│   │   ├── jquery.js
│   │   ├── layui.all.js
│   │   ├── layui.css
│   │   ├── layui.js
│   │   ├── layui.js.下载
│   │   ├── layui.mobile.css
│   │   ├── modules
│   ├── element.js
│   ├── font
│   │   ├── iconfont.eot
│   │   ├── iconfont.svg
│   │   ├── iconfont.ttf
│   │   ├── iconfont.woff
│   │   └── iconfont.woff2
│   ├── images
│   │   ├── backgroup.png
│   │   ├── bg.jpeg
│   │   ├── h1.png
│   │   ├── h.jpg
│   │   ├── h.png
│   │   ├── hs.jpg
│   │   └── tab.png
│   ├── jquery-3.4.1.js
│   ├── jquery.js
│   ├── lay
│   ├── layui.js.下载
└── templates
    ├── autoSummary.html
    ├── base.html
    ├── demo1.html
    ├── hello.html
    ├── index.html.bak
    ├── jquery.js
    ├── news-extraction.html
    ├── pvuv.html
    ├── qunliao.html
    ├── ss.html
    ├── testform.html
    ├── use_template.html
    └── web_sckone.html

163 directories, 750 files

#### 3.2 展示效果
![提交前]()  
![提交后]()


### 参考文献
[1] A Neural Attention Model for Abstractive Sentence Summarization https://arxiv.org/abs/1509.00685  
[2] A Latent Variable Model Approach to PMI-based Word Embeddings https://arxiv.org/abs/1502.03520  

[3]: Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.

[4]: Hofmann, T. (1999). Probabilistic latent semantic indexing. In Proceedings of the 22nd annual international ACM SIGIR conference on Research and development in information retrieval (pp. 50-57). ACM.

[5]: Li, F., Huang, M., & Zhu, X. (2010). Sentiment Analysis with Global Topics and Local Dependency. In AAAI (Vol. 10, pp. 1371-1376).

[6]: Medhat, W., Hassan, A., & Korashy, H. (2014). Sentiment analysis algorithms and applications: A survey. Ain Shams Engineering Journal, 5(4), 1093-1113.

