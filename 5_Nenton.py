import numpy as np
import numpy.matlib
import gensim

#函数表达式
#f(x)=alpha * dist(x,T)+ beta * sum(i,k)dist(x,Vi)
#dist(x,t)=np.dot(x-T,(x-T).T)
#sum(i,k)dist(x,Vi): r=np.dot(x-V,(x-V).T)
#                    final_r=np.sum(np.diag(r))
# fun = lambda x:100*(x[0]**2 - x[1]**2)**2 +(x[0] - 1)**2
#

# a=alpha * np.dot(x-T,(x-T).T)
# b=beta * np.sum(np.diag(np.dot(x-V,(x-V).T)))


#梯度向量
# gfun = lambda x:np.array([400*x[0]*(x[0]**2 - x[1]) + 2*(x[0] - 1),-200*(x[0]**2 - x[1])])
# gfun = lambda x:np.array([2*(x[0]-1),2*(x[1]-2),2*(x[2]-3)])
# gfun = lambda x:np.array([6*x[0]-12,6*x[1]-12])

#Hessian矩阵
# hess = lambda x:np.array([[1200*x[0]**2 - 400*x[1] + 2,-400*x[0]],[-400*x[0],200]])
# hess = lambda x:np.array([[6,0,],[0,6]])

def dfp(fun,gfun,hess,x0):
    #功能：用DFP算法求解无约束问题：min fun(x)
    #输入：x0式初始点，fun,gfun，hess分别是目标函数和梯度,Hessian矩阵格式
    #输出：x,val分别是近似最优点，最优解，k是迭代次数
    maxk = 1e5
    rho = 0.05
    sigma = 0.4
    epsilon = 1e-13 #迭代停止条件
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

        m = 0;
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

def getVec(alpha,beta,T,V):
    fun = lambda x:alpha * np.dot(x-T,(x-T).T) + beta * np.sum(np.diag(np.dot(x-V,(x-V).T)))
    gfun = lambda x: 2 * alpha *(x-T) + 2 * beta * np.sum(x-V,axis=0)

    # dem=T.shape[0]

    dem = V.shape[1]
    a = numpy.matlib.identity(dem)
    hess = lambda x:np.array((2 * alpha + 2 * beta * V.shape[0]) * a)
    # hess = lambda x:np.array([[8,0],[0,8]])

    x0, fun0, k = dfp(fun, gfun, hess, T)

    print("-----------最终解为：",x0,"-----------")
    print("-----------方程的距离和为：", fun0, "-----------")
    print("-----------迭代次数为：",k, "-----------")

    print("-----------我是分隔符-我是分隔符-我是分隔符-我是分隔符--------")
    return x0

if __name__ == '__main__':

    # T=np.array([1,1])
    # V=np.array([[2,2],[3,3]])
    # getVec(1,1,T,V)

    # fdir = 'data/'
    # model = gensim.models.Word2Vec.load(fdir + 'wiki.zh.text.model')
    #
    # kuaile = model["快乐"]
    # xinfu = model["幸福"]
    # yukuai = model["愉快"]
    #
    # a=np.array([xinfu,yukuai])
    #
    # final_test = getVec(1, 1, kuaile, a)
    # np.save("data/final_test.npy", final_test)
    T=np.array([1,1])
    V = np.array([[0,4], [4,4],[0,0], [4,0]])
    x=getVec(0,1,T,V)



    # AngerMatrix = np.load("data/vecs/AngerMatrix.npy")
    # AngerVec = np.load("data/vecs/AngerVec.npy")
    # final_AngerVec=getVec(1, 1, AngerVec, AngerMatrix)
    # np.save("data/vecs/final_AngerVec.npy", final_AngerVec)
    #
    # # DisgustMatrix = np.load("data/DisgustMatrix.npy")
    # # DisgustVec = np.load("data/DisgustVec.npy")
    # # final_DisgustVec = getVec(1, 1, DisgustVec, DisgustMatrix)
    # # np.save("data/final_DisgustVec.npy", final_DisgustVec)
    #
    # SadnessMatrix = np.load("data/vecs/SadnessMatrix.npy")
    # SadnessVec = np.load("data/vecs/SadnessVec.npy")
    # final_SadnessVec = getVec(1, 1, SadnessVec, SadnessMatrix)
    # np.save("data/vecs/final_SadnessVec.npy", final_SadnessVec)
    #
    # HappinessMatrix = np.load("data/vecs/HappinessMatrix.npy")
    # HappinessVec = np.load("data/vecs/HappinessVec.npy")
    # final_HappinessVec = getVec(1, 1, HappinessVec, HappinessMatrix)
    # np.save("data/vecs/final_HappinessVec.npy", final_HappinessVec)
    #
    # # GreatMatrix = np.load("data/GreatMatrix.npy")
    # # GreatVec = np.load("data/GreatVec.npy")
    # # final_GreatVec = getVec(1, 1, GreatVec, GreatMatrix)
    # # np.save("data/final_GreatVec.npy", final_GreatVec)
    #
    # FearMatrix = np.load("data/vecs/FearMatrix.npy")
    # FearVec = np.load("data/vecs/FearVec.npy")
    # final_FearVec = getVec(1, 1, FearVec, FearMatrix)
    # np.save("data/vecs/final_FearVec.npy", final_FearVec)
    #
    # SurpriseMatrix = np.load("data/vecs/SurpriseMatrix.npy")
    # SurpriseVec = np.load("data/vecs/SurpriseVec.npy")
    # final_SurpriseVec = getVec(1, 1, SurpriseVec, SurpriseMatrix)
    # np.save("data/vecs/final_SurpriseVec.npy", final_SurpriseVec)
    #
    # EmotionMatrix = np.array([final_AngerVec,final_SurpriseVec,final_FearVec,final_HappinessVec,final_SadnessVec])
    # np.save("data/vecs/EmotionMatrix.npy", EmotionMatrix)
