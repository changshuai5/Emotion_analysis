import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import gensim
import numpy as np
import logging
import codecs
import xlwt
import xlrd
import sys
import numpy

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

#向量x和向量y之间的余弦相似度
def CosineDistance(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

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


if __name__ == '__main__':

    model = gensim.models.Word2Vec.load('data/word2vec/rs200.hy.text.model')
    # “怒”邻居词
    AngerWords = getFinalWord("愤怒", positive=False)

    T = np.zeros(200)
    AngerMatrix=[]
    for word in AngerWords:
        AngerMatrix.append(model[word[0]])
    AngerMatrix=np.array(AngerMatrix)
    AngerVec=getVec(T,AngerMatrix)


    # “哀”邻居词
    SadnessWords = getFinalWord("悲伤", positive=False)
    SadnessMatrix = []
    for word in SadnessWords:
        SadnessMatrix.append(model[word[0]])
    SadnessMatrix = np.array(SadnessMatrix)
    SadnessVec = getVec(T, SadnessMatrix)


    # “乐”邻居词
    HappinessWords = getFinalWord("快乐", positive=True)
    HappinessMatrix = []
    for word in HappinessWords:
        HappinessMatrix.append(model[word[0]])
    HappinessMatrix = np.array(HappinessMatrix)
    HappinessVec = getVec(T, HappinessMatrix)


    # “惧”邻居词
    FearWords = getFinalWord("恐惧", positive=False)
    FearMatrix = []
    for word in FearWords:
        FearMatrix.append(model[word[0]])
    FearMatrix = np.array(FearMatrix)
    FearVec = getVec(T, FearMatrix)


    # “惊”邻居词
    SurpriseWords = getFinalWord("惊愕", positive=True)
    SurpriseMatrix = []
    for word in SurpriseWords:
        SurpriseMatrix.append(model[word[0]])
    SurpriseMatrix = np.array(SurpriseMatrix)
    SurpriseVec = getVec(T, SurpriseMatrix)

    EmotionMatrix = np.array([SadnessVec, AngerVec, HappinessVec, FearVec, SurpriseVec])
    np.save("data/vecs/EmotionMatrix_1.npy", EmotionMatrix)
    print("well done")


    #
    #
    #
    # EmotionMatrix = np.array([AngerVec, DisgustVec, SadnessVec, GreatVec, FearVec, SurpriseVec])
    # np.save("data/EmotionMatrix.npy", EmotionMatrix)

    # “怒”情感词向量
    # AngerVec=refinedVec("",positive=False)
    #
    # #“恶”情感词向量
    # DisgustVec=refinedVec("",positive=False)
    #
    # # “哀”情感词向量
    # SadnessVec = refinedVec("", positive=False)
    #
    # # “乐”情感词向量
    # HappinessVec = refinedVec("", positive=True)
    #
    # # “好”情感词向量
    # GreatVec = refinedVec("", positive=True)
    #
    # # “惧”情感词向量
    # FearVec = refinedVec("", positive=False)
    #
    # # “惊”情感词向量
    # SurpriseVec = refinedVec("", positive=True)
    #
    # EmotionMatrix=np.array([AngerVec,DisgustVec,SadnessVec,GreatVec,FearVec,SurpriseVec])
    # np.save("data/EmotionMatrix.npy", EmotionMatrix)



# file=" "
# model = gensim.models.Word2Vec.load('data/wiki.zh.text.model')
# wordList=getsimilarWordVec(model,"开心",100)
# words=pro_output(file,wordList)




# wordList = model.most_similar("开心",topn=20)
# for w in wordList:
#     print(w[0], w[1])

