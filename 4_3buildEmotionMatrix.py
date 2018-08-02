import nltk
import nltk.data
from nltk.tokenize import WordPunctTokenizer
import xlwt
import xlrd
import numpy as np
import gensim
import numpy as np
import numpy.matlib
import codecs, sys, string, re

#向量x和向量y之间的余弦相似度
def CosineDistance(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def getAllWords(filename):
    words=[]
    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_index(0)
    for row in range(1, sheet.nrows):
        word = []
        list_temp = sheet.row_values(row)
        if int(list_temp[5]) >=5:
            for i in range(7):
                word.append(list_temp[i])
            words.append(word)
    return words

def getEmotionWords(words):
    # 乐
    pa_list = []
    pe_list = []
    # 怒
    na_list = []
    # 哀
    nb_list = []
    nj_list = []
    # 惧
    ni_list = []
    nc_list = []

    # 惊
    pc_list = []
    for word in words:
        if word[4] == "PA" and (int(word[6]) == 0 or int(word[6]) == 1):
            pa_list.append(word)
        elif word[4] == "PE" and (int(word[6]) == 0 or int(word[6]) == 1):
            pe_list.append(word)
        elif word[4] == "NA" and (int(word[6]) == 0 or int(word[6]) == 2):
            na_list.append(word)
        elif word[4] == "NB" and (int(word[6]) == 0 or int(word[6]) == 2):
            nb_list.append(word)
        elif word[4] == "NJ" and (int(word[6]) == 0 or int(word[6]) == 2):
            nj_list.append(word)
        elif word[4] == "NI" and (int(word[6]) == 0 or int(word[6]) == 2):
            ni_list.append(word)
        elif word[4] == "NC" and (int(word[6]) == 0 or int(word[6]) == 2):
            nc_list.append(word)
        elif word[4] == "PC" and (int(word[6]) == 0 or int(word[6]) == 1):
            pc_list.append(word)
    return pa_list,pe_list,na_list,nb_list,nj_list,ni_list,nc_list,pc_list

def getNeighborVec(wordList,model):
    b = {}
    for word1 in wordList:
        a = []
        for word2 in wordList:
            try:
                distance = CosineDistance(model[word1[0]], model[word2[0]])
                if distance > 0.6 and distance < 0.99:
                    a.append(word2[0])
            except KeyError:
                # wordList.remove(word2)
                continue
            except TypeError:
                continue
        if len(a) < 3:
            continue
        else:
            b[word1[0]] = a
    return b

def dfp(fun,gfun,hess,x0):
    #功能：用DFP算法求解无约束问题：min fun(x)
    #输入：x0式初始点，fun,gfun，hess分别是目标函数和梯度,Hessian矩阵格式
    #输出：x,val分别是近似最优点，最优解，k是迭代次数
    maxk = 1e5
    rho = 0.05
    sigma = 0.4
    epsilon = 1e-12 #迭代停止条件
    k = 0
    n = np.shape(x0)[0]
    #将Hessian矩阵初始化为单位矩阵
    Hk = np.linalg.inv(hess(x0))

    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0*np.dot(Hk,gk)
#         print dk

        m = 0
        mk = 0
        while m < 20:#用Armijo搜索步长
            if fun(x0 + rho**m*dk) < fun(x0) + sigma*rho**m*np.dot(gk,dk):
                mk = m
                break
            m += 1
        #print mk
        #DFP校正
        x = x0 + rho**mk*dk
        print("第"+str(k)+"次的迭代结果为："+str(x))
        sk = x - x0
        yk = gfun(x) - gk

        if np.dot(sk,yk) > 0:
            Hy = np.dot(Hk,yk)
            sy = np.dot(sk,yk) #向量的点积
            yHy = np.dot(np.dot(yk,Hk),yk) #yHy是标量
            Hk = Hk - 1.0*Hy.reshape((n,1))*Hy/yHy + 1.0*sk.reshape((n,1))*sk/sy

        k += 1
        x0 = x
    return x0,fun(x0),k

def getVec(T,V):
    fun = lambda x:  np.sum(np.diag(np.dot(x-V,(x-V).T)))
    gfun = lambda x: 2 *  np.sum(x-V,axis=0)

    dem=V.shape[1]
    a = numpy.matlib.identity(dem)
    hess = lambda x: np.array((2 * V.shape[0]) * a)

    x0, fun0, k = dfp(fun, gfun, hess,T)
    return x0

# 返回各个词类向量的聚类中心
def getWordVecs(wordDict,model):
    T = np.zeros(200)
    vecList=[]
    for wordC,wordlist in wordDict.items():
        vecs = []
        for word in wordlist:
            try:
                vecs.append(model[word])
            except KeyError:
                continue
        vecs.append(model[wordC])
        if len(vecs) > 0:
            vecsArray = getVec(T, np.array(vecs))
        vecList.append(vecsArray)
    return vecList


if __name__ == '__main__':
    # print(splitSentence("我爱你我的家My name is Tom."))
    model = gensim.models.Word2Vec.load('data/word2vec/rs200.hy.text.model')
    File = "data/情感词汇本体.xlsx"
    words=getAllWords(File)

    pa_list, pe_list, na_list, nb_list, nj_list, ni_list, nc_list, pc_list=getEmotionWords(words)

    T=np.zeros(200)
    pa_dict=getNeighborVec(pa_list,model)
    pa_vecList=getWordVecs(pa_dict,model)

    pe_dict = getNeighborVec(pe_list, model)
    pe_vecList = getWordVecs(pe_dict, model)

    na_dict = getNeighborVec(na_list, model)
    na_vecList = getWordVecs(na_dict, model)

    nb_dict = getNeighborVec(nb_list, model)
    nb_vecList = getWordVecs(nb_dict, model)

    nj_dict = getNeighborVec(nj_list, model)
    nj_vecList = getWordVecs(nj_dict, model)

    ni_dict = getNeighborVec(ni_list, model)
    ni_vecList = getWordVecs(ni_dict, model)

    nc_dict = getNeighborVec(nc_list, model)
    nc_vecList = getWordVecs(nc_dict, model)

    pc_dict = getNeighborVec(pc_list, model)
    pc_vecList = getWordVecs(pc_dict, model)

    EmotionMatrix = np.concatenate((pa_vecList, pe_vecList, na_vecList, nb_vecList, nj_vecList, ni_vecList, nc_vecList, pc_vecList), axis=0)
    # print("type:",type(EmotionMatrix))
    # print("len:",len(EmotionMatrix))
    # print("shape:",EmotionMatrix.shape)

    np.save("data/vecs/EmotionMatrix_3.npy", EmotionMatrix)
    print("well done")































