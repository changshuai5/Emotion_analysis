import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import gensim
import numpy as np
import logging
import codecs
import xlwt
import xlrd

def getsimilarWordVec(model,word,num):
    finalword=[]
    wordList = model.most_similar(word,topn=num)
    for w in wordList:
        finalword.append(w[0])
    print("------------向量空间中的前",num,"个邻居词为：------------")
    print(finalword)
    return finalword


def getEmotionScore(filename,wordlist):
    words=[]
    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_index(0)
    for row in range(1, sheet.nrows):
        list_temp = sheet.row_values(row)
        word=list_temp[0]
        strength=int(list_temp[5])
        polarity=int(list_temp[6])
        if word in wordlist:
                words.append((word,strength,polarity))
    print("------------情感本体中的词为：------------")
    print(words)
    return words

def filterWord(t,positive=True):
    if positive==True:
        word= [w for w in t if w[2] == 1 or w[2] == 0]
    elif positive==False:
        word= [w for w in t if w[2] == 2 or w[2] == 0]
    print("------------最终词为：------------")
    print(word)
    return word

def  by_sore(w):
    return w[1]

def sortByscoreDec(word):
    return sorted(word,key=by_sore,reverse=True)


def getFinalWord(target_word,positive=True):
    #加载词向量模型
    model = gensim.models.Word2Vec.load('data/wiki.zh.text.model')

    #情感本体文件路径
    File="data/情感词汇本体.xlsx"

    #在向量空间中找到目标词的num个最近邻居
    wordList=getsimilarWordVec(model,target_word,100)

    #根据找到的num个邻居，如果情感本体中有这些词则返回，没有则忽略
    words=getEmotionScore(File,wordList)

    #最后找得到的既是情感本体中的词，又是向量空间中距离目标词比较近的词
    word=filterWord(words,positive)

    #得到的上一步的词，根据情感打分进行倒序排列
    final_words=sortByscoreDec(word)
    return final_words

# def batchGradientDescent(x,y,alpha,beta,m,maxIterations):
#     for i in range(0,maxIterations):
#         for j in range(m):
#             for k in range(len(y)):
#                 x[j] = x[j]- alpha * x[j] - beta * sum(y[k][j])
    # #得到最终所有词的词向量矩阵
    # V=[]
    # for word in final_words:
    #     V.append(model[word])
    #
    # #根据公式迭代求解目标词的词向量
    # T=model(target_word)


if __name__ == '__main__':

    model = gensim.models.Word2Vec.load('data/word2vec/rs200.hy.text.model')
    # “怒”邻居词
    AngerWords = getFinalWord("愤怒", positive=False)

    AngerVec=model["愤怒"]

    np.save("data/vecs/AngerVec.npy", AngerVec)

    AngerMatrix=[]
    for word in AngerWords:
        AngerMatrix.append(model[word[0]])
    np.save("data/vecs/AngerMatrix.npy",np.array(AngerMatrix))


    # # “恶”邻居词
    # DisgustWords = getFinalWord("烦闷", positive=False)
    # DisgustVec = model["烦闷"]
    # np.save("data/DisgustVec.npy", DisgustVec)

    # DisgustMatrix = []
    # for word in DisgustWords:
    #     DisgustMatrix.append(model[word[0]])
    # np.save("data/DisgustMatrix.npy", np.array(DisgustMatrix))

    # “哀”邻居词
    SadnessWords = getFinalWord("悲伤", positive=False)
    SadnessVec = model["悲伤"]
    np.save("data/vecs/SadnessVec.npy", SadnessVec)

    SadnessMatrix = []
    for word in SadnessWords:
        SadnessMatrix.append(model[word[0]])
    np.save("data/vecs/SadnessMatrix.npy", np.array(SadnessMatrix))

    # “乐”邻居词
    HappinessWords = getFinalWord("快乐", positive=True)
    HappinessVec = model["快乐"]
    np.save("data/vecs/HappinessVec.npy", HappinessVec)

    HappinessMatrix = []
    for word in HappinessWords:
        HappinessMatrix.append(model[word[0]])
    np.save("data/vecs/HappinessMatrix.npy", np.array(HappinessMatrix))

    # # “好”邻居词
    # GreatWords = getFinalWord("尊敬", positive=True)
    # GreatVec = model["尊敬"]
    # np.save("data/GreatVec.npy", GreatVec)
    #
    # GreatMatrix = []
    # for word in GreatWords:
    #     GreatMatrix.append(model[word[0]])
    # np.save("data/GreatMatrix.npy", np.array(GreatMatrix))

    # “惧”邻居词
    FearWords = getFinalWord("恐惧", positive=False)
    FearVec = model["恐惧"]
    np.save("data/vecs/FearVec.npy", FearVec)

    FearMatrix = []
    for word in FearWords:
        FearMatrix.append(model[word[0]])
    np.save("data/vecs/FearMatrix.npy", np.array(FearMatrix))

    # “惊”邻居词
    SurpriseWords = getFinalWord("惊愕", positive=True)
    SurpriseVec = model["惊愕"]
    np.save("data/vecs/SurpriseVec.npy", SurpriseVec)

    SurpriseMatrix = []
    for word in SurpriseWords:
        SurpriseMatrix.append(model[word[0]])
    np.save("data/vecs/SurpriseMatrix.npy", np.array(SurpriseMatrix))


