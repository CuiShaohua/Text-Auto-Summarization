from textrank4zh import TextRank4Keyword

def keywords_weight(sent_inp):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=sent_inp, lower=True, window=5)
    #print('关键词：')
    keywords = {}
    for item in tr4w.get_keywords(10, word_min_len=1):
        keywords[item['word']] = item['weight']
        # print(item['word'], item['weight']

    return keywords



if __name__== "__main__":
    inp = input("输入一段新闻：")
    keywords_weight(inp)