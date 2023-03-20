import numpy as np
def random_split(random_state,X,Y,test_size):
    up = Y.shape[0]#样本个数
    test_num = int(np.round(up*test_size,0))
    train_num = int(up-test_num)
    seed = np.random.seed(random_state)
    train_index = np.random.choice(up,train_num,replace = False)
    train_index_set = set(train_index)
    index_all_set = set(np.arange(up))
    index_mode_set = index_all_set - train_index_set
    index_mode = np.array(list(index_mode_set))
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[index_mode]
    Y_test = Y[index_mode]
    return X_train,X_test,Y_train,Y_test

#x_train,x_test,y_train,y_test = random_split(1,X,Y,0.3)#测试

#step2:定义一个实现分层采样的函数
#定义一个计算每一类别抽多少样本的函数
def sample_each_calss(train_test_num,n_class,counts,length_y):
    sample_should = []
    mode_sample = train_test_num  # 最后一个类别应该抽的数目，防止因为计算精度的问题导致，三类样本个数加总不等于原来的Y样本总数
    for i in range(n_class):
        if i != (n_class - 1):
            sample_should_i = int(counts[i] / length_y*train_test_num)
            sample_should.append(sample_should_i)
            mode_sample = mode_sample - sample_should_i
        else:

            sample_should.append(mode_sample)
    return sample_should

#分别计算训练集和测试集每个类别应该抽多少样本
#train_should,test_should = sample_each_calss(train_num,n_class,counts,length_y),sample_each_calss(test_num,n_class,counts,length_y)
#定义一个分层抽样的函数
#寻找n类样本点的位置
def search_index_eachclass(value,length_y,Y,n_class):
    #temp = []
    index_nclass = []
    for i in range(n_class):
        temp_i = np.array([value[i]]*length_y)
        index_i = np.array(np.where(temp_i == Y)).flatten()
        index_nclass.append(index_i)
    #temp = np.array(temp)
    index_nclass = np.array(index_nclass)
    return index_nclass
#index_nclass = search_index_eachclass(value,length_y,Y)测试
#分别从每一类样本中抽样
#定义一个随机获取三类index的函数
def sample_index_nclass(index_nclass,random_state,n_class,train_should):
    seed = np.random.seed(random_state)
    Train_index = []
    Test_index = []
    for i in range(n_class):
        train_index_i = np.random.choice(index_nclass[i],train_should[i],replace= False)
        Train_index.append(train_index_i)
        train_index_i_set = set(train_index_i)
        index_i_set = set(index_nclass[i])
        test_index_set = index_i_set - train_index_i_set
        test_index = np.array(list(test_index_set))
        Test_index.append(test_index)
    Train_index = np.array(Train_index).flatten()
    Test_index = np.array(Test_index).flatten()
    return Train_index,Test_index

def layered_sample(X,Y,test_size,random_state):
    # 计算标签中一共有多少个类别，每个类别又多少个样本，即频数统计
    value, counts = np.unique(Y, return_counts=True)  # 计算标签类别以及各类别的频数
    n_class = int(len(value))  # 标签类别个数
    # 分别计算训练集和测试集有多少个样本
    length_y = Y.shape[0]
    test_size = test_size
    test_num = int(np.round(length_y * test_size, 0))#总的训练集个数
    train_num = int(length_y - test_num)
    train_should = sample_each_calss(train_num, n_class, counts, length_y)#训练集每个类别应当抽样的个数
    index_nclass = search_index_eachclass(value, length_y, Y,n_class)#每个类别总的index
    Train_index, Test_index = sample_index_nclass(index_nclass, random_state, n_class, train_should)#从每个类别index中该抽取多少样本的index汇总
    # 获取数据
    X_train, X_test, Y_train, Y_test = X[Train_index], X[Test_index], Y[Train_index], Y[Test_index]
    return X_train,X_test,Y_train,Y_test

#X_train,X_test,Y_train,Y_test  = layered_sample(X,Y,0.3,1)

#定义一个总的函数，将分层抽样和不分层抽样结合在一起
def train_test_split_self(X,Y,test_size,random_state,layer=True):
    if layer==True:
        X_train, X_test, Y_train, Y_test = layered_sample(X, Y, test_size,random_state)
    else:
        X_train, X_test, Y_train, Y_test = random_split(random_state, X, Y, test_size)
    return X_train, X_test, Y_train, Y_test

#X_train, X_test, Y_train, Y_test = train_test_split_self(X,Y,0.3,1,False)



