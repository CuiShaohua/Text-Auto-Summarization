# AutoSummarizzation
#####  text zuto summarization for news 
### 0 写在前
>>   &ensp;&ensp; 目前不论是大小云厂商还是提供SaaS服务的AI企业，大多推出了文本自动摘要这一功能。华为云去年之前还未做这个功能，但目前已经在EI里推出了这个功能。笔者在根据开课吧学习AI知识之后，将文本自动摘要采用无监督的方式进行实现，并且结合Flask进行项目展示。  
>>    &ensp;&ensp;自有文本摘要的研究以来，研究方法大致分为两大类，一种是生成式的文本摘要，另一种是抽取式文本摘要。对于中文训练集来说，由于尚且没有诸如英语文摘的训练集，所以很少能见到有关学者是采用生成式的方法进行中文的摘要训练；而对于英语生成式文本摘要代表作，可以查看[A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685)，引用原文一句```“Summarization based on text extraction is inherently limited, but generation-style abstractive methods have proven challenging to build. In this work, we propose a fully data-driven approach to abstractive sentence summarization. Our method utilizes a local attention-based model that generates each word of the summary conditioned on the input sentence. While the model is structurally simple, it can easily be trained end-to-end and scales to a large amount of training data. The model shows significant performance gains on the DUC-2004 shared task compared with several strong baselines.”```，可以看出深度学习方法的生成式存在着抽取式无法比拟的众多优势。笔者个人认为，生成式方法虽然目前存在一些尚未改进的地方（a.采用RNN结构，训练速度慢；b.深度学习框架不容易调测；c.深度学习框架依赖于训练集，训练集若集中于某个行业领域，则训练出的模型的普适性不高），但生成式方法一定是文本摘要的未来（a. 生成的句子更灵活，不用写一堆“蹩脚”的规则；b. 引入高层语义，更像人的思考之后说过的话）。  
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







### 参考文献

[1] A Neural Attention Model for Abstractive Sentence Summarization https://arxiv.org/abs/1509.00685

