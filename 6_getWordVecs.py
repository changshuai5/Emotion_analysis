import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
import gensim


#向量x和向量y之间的余弦相似度
def CosineDistance(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# 返回特征词向量
def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')

#改进词向量：返回结果是一个关于文档中所有句子的列表，列表的每个元素是句子中所有词对应的词向量
def getDocVecs(filename,model):
    fileVecs=[]
    word_weight = []
    EmotionMatrix = np.load("data/vecs/EmotionMatrix_new.npy")
    lineNum = 1
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            final_vecs = []
            # DocVecs = []
            print("Start line: " ,lineNum)
            wordList = line.split(' ')
            print(wordList)
            vecs = getWordVecs(wordList,model)
            print("向量长度：",len(vecs))
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                for wordvec in vecs:
                    for emotionVec in EmotionMatrix:
                        distance = CosineDistance(wordvec, emotionVec)
                        word_weight.append(distance)
                        if len(word_weight) == len(EmotionMatrix):
                            maxD = max(word_weight)
                            index = word_weight.index(maxD)
                            vecsArray = (wordvec + EmotionMatrix[index])/ 2
                            final_vecs.append(vecsArray)
                            word_weight.clear()
                    # DocVecs.append(final_vecs)
                fileVecs.append(final_vecs)
            lineNum = lineNum + 1
    return fileVecs

#原始词向量：返回结果是一个关于文档中所有句子的列表，列表的每个元素是句子中所有词对应的词向量
def getDocVecs2(filename,model):
    fileVecs=[]
    word_weight = []
    EmotionMatrix = np.load("data/vecs/EmotionMatrix_new.npy")
    lineNum = 1
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            final_vecs = []
            # DocVecs = []
            print("Start line: " ,lineNum)
            wordList = line.split(' ')
            print(wordList)
            vecs = getWordVecs(wordList,model)
            print("向量长度：",len(vecs))
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                fileVecs.append(vecs)
            lineNum = lineNum + 1
    return fileVecs



def getFileVecs(filename,model):
    FileVecs = []
    word_weight = []
    EmotionMatrix = np.load("data/vecs/EmotionMatrix_new.npy")
    lineNum = 1
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            final_vecs = []
            wordList = line.split(' ')
            # print("line:",lineNum ,"word",wordList)
            vecs = getWordVecs(wordList,model)
            print("向量长度：", len(vecs))
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                for wordvec in vecs:
                    for emotionVec in EmotionMatrix:
                        distance = CosineDistance(wordvec, emotionVec)
                        word_weight.append(distance)
                        if len(word_weight) == len(EmotionMatrix):
                            minD = min(word_weight)
                            index = word_weight.index(minD)
                            vecsArray = (wordvec + EmotionMatrix[index])/ 2
                            final_vecs.append(vecsArray)
                            word_weight.clear()
                docVec = sum(np.array(final_vecs)) / len(final_vecs)
                FileVecs.append(docVec)
            lineNum = lineNum + 1
    return FileVecs

def splitData(data):
    split_rate = 0.3
    p1 = int(len(data) * (1 - split_rate - split_rate))
    p2 = int(len(data) * (1 - split_rate))
    # 将文本进行混杂
    data_a = data[:p1]
    data_b = data[p1:p2]
    data_c = data[p2:]
    return data_a,data_b,data_c

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('data/word2vec/rs200.hy.text.model')
    # b=getDocVecs("data/test.txt",model)
    # print(len(b))
    # print(len(b[0][0]))
    # print(len(b[0][1]))


    # angryFild=getFileVecs("data/yuliao/angry_out_nostop.txt",model)
    # angry_length=len(angryFild)
    #
    # fearFile=getFileVecs("data/yuliao/fear_out_nostop.txt",model)
    # fear_length=len(fearFile)
    #
    # happyFile = getFileVecs("data/yuliao/happy_out_nostop.txt", model)
    # happy_length = len(happyFile)
    #
    # sadFile = getFileVecs("data/yuliao/sad_out_nostop.txt", model)
    # sad_length= len(sadFile)
    #
    # surpriseFile = getFileVecs("data/yuliao/surprise_out_nostop.txt", model)
    # surprise_length= len(surpriseFile)
    #
    # X=np.concatenate([np.array(angryFild),np.array(fearFile),np.array(happyFile),np.array(sadFile),np.array(surpriseFile)])
    # np.save("data/text/text.npy", X)
    #
    # Y = np.concatenate((np.zeros(angry_length), np.ones(fear_length),2* np.ones(happy_length), 3* np.ones(sad_length),4* np.ones(surprise_length)))
    # np.save("data/text/label.npy", Y)



    angryFild=getDocVecs2("data/yuliao/angry_out_nostop.txt",model)
    angry_a,angry_b,angry_c=splitData(angryFild)

    fearFile = getDocVecs2("data/yuliao/fear_out_nostop.txt", model)
    fear_a, fear_b, fear_c = splitData(fearFile)

    sadFile = getDocVecs2("data/yuliao/sad_out_nostop.txt", model)
    sad_a, sad_b, sad_c = splitData(sadFile)

    happyFile = getDocVecs2("data/yuliao/happy_out_nostop.txt", model)
    happy_a, happy_b, happy_c = splitData(happyFile)

    surpriseFile = getDocVecs2("data/yuliao/surprise_out_nostop.txt", model)
    surprise_a, surprise_b, surprise_c = splitData(surpriseFile)

    all_text=[]
    all_text+=angry_a
    all_text+=fear_a
    all_text += sad_a
    all_text += happy_a
    all_text += surprise_a

    Y_a=np.concatenate((np.zeros(len(angry_a)), np.ones(len(fear_a)),2* np.ones(len(sad_a)), 3* np.ones(len(happy_a)),4* np.ones(len(surprise_a))))

    all_text += angry_b
    all_text += fear_b
    all_text += sad_b
    all_text += happy_b
    all_text += surprise_b

    Y_b = np.concatenate((np.zeros(len(angry_b)), np.ones(len(fear_b)), 2 * np.ones(len(sad_b)), 3 * np.ones(len(happy_b)), 4 * np.ones(len(surprise_b))))

    all_text += angry_c
    all_text += fear_c
    all_text += sad_c
    all_text += happy_c
    all_text += surprise_c

    Y_c = np.concatenate((np.zeros(len(angry_c)), np.ones(len(fear_c)), 2 * np.ones(len(sad_c)), 3 * np.ones(len(happy_c)), 4 * np.ones(len(surprise_c))))

    all_label = np.concatenate((Y_a,Y_b,Y_c))

    np.save("data/text/all_label_1.npy", all_label)
    np.save("data/text/all_text_2.npy", all_text)





    # fearFile=getDocVecs("data/yuliao/fear_out_nostop.txt",model)
    # fear_length=len(fearFile)
    #
    # happyFile = getDocVecs("data/yuliao/happy_out_nostop.txt", model)
    # happy_length = len(happyFile)
    #
    # sadFile = getDocVecs("data/yuliao/sad_out_nostop.txt", model)
    # sad_length= len(sadFile)
    #
    # surpriseFile = getDocVecs("data/yuliao/surprise_out_nostop.txt", model)
    # surprise_length= len(surpriseFile)
    #
    #
    #
    # X=np.concatenate([np.array(angryFild),np.array(fearFile),np.array(happyFile),np.array(sadFile),np.array(surpriseFile)])
    # print(len(X))
    # # np.save("data/text/text_a.npy", X)
    #
    #
    # Y = np.concatenate((np.zeros(angry_length), np.ones(fear_length),2* np.ones(happy_length), 3* np.ones(sad_length),4* np.ones(surprise_length)))
    # # np.save("data/text/label_a.npy", Y)
    # print(len(Y))




