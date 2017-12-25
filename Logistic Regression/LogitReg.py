from numpy import *
def sigmoid(x):
    return 1.0/(1.0+exp(-x))
def LogitReg(x,y,tol = 0.001,maxiter = 1000):
    samples,features = x.shape  #分别表示观测样本数量和特征数量
    features += 1
    
    #全部转换为矩阵
    xdata = array(ones((samples,features)))
    xdata[:,0] = 1
    xdata[:,1:] = x
    xdata = mat(xdata)  #sample行，features列的输入
    
    y = mat(y.reshape(samples,1))  #label，一个长度为samples的向量
    
    #首先初始化beta，令所有的系数为1,生成一个长度为features的列向量
    beta = mat(zeros((features,1)))
    
    iternum = 0 #迭代计数器
    
    #计算初始损失
    
    loss0 = float('inf')
    J = []
    while iternum < maxiter:
        try:
            p = sigmoid(xdata*beta) #计算似然概率
            nabla = 1.0/samples*xdata.T*(p-y)   #计算梯度
            H = 1.0/samples*xdata.T*diag(p.getA1())* diag((1-p).getA1())*xdata  #计算黑塞矩阵
            
            loss = 1.0/samples*sum(-y.getA1()*log(p.getA1())-(1-y).getA1()*log((1-p).getA1())) #计算损失
            J.append(loss)
            beta =beta -  H.I * nabla  #更新参数
            iternum += 1 #迭代器加一
            if loss0 - loss < tol:
                break
            loss0 = loss
        except:
            #通常当黑塞矩阵奇异的时候，说明梯度已经非常小了，也可以认为此时已经收敛了
            break
        
    return beta,J

#预测函数
def predictLR(data,beta):
    data = array(data)
    if len(data.shape) == 1:
        length = len(data)
        newdata = tile(0,length+1)
        newdata[0] = 1
        newdata[1:] = data
        newdata = mat(newdata)
        pass
    else:
        shape = data.shape
        newdata = zeros((shape[0],shape[1]+1))
        newdata[:,0] = 1
        newdata[:,1:] = data
        newdata = mat(newdata)
    return sigmoid(newdata*beta)

df = pd.read_csv('df.csv',header=None)
df = array(df)
df.shape
xdata = df[:,:3]
ydata = df[:,3]

beta,J = LogitReg(xdata,ydata) #拟合

testdata = xdata[1:10,]

predictLR(testdata,beta)

matrix([[ 0.4959212 ],
        [ 0.44642627],
        [ 0.47419207],
        [ 0.42209742],
        [ 0.41802565],
        [ 0.51283217],
        [ 0.44833226],
        [ 0.41252982],
        [ 0.47853786]])
