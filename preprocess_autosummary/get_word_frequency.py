### 文章向量 doc_vec.doc_vector
### sent_set    ---> 一级 句子[] 二级 词汇[]
def get_word_frequency(word_text, looktable):
    if word_text in looktable:
        return looktable[word_text]
    else:
        return 1.0
