import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets

def initialize_parameters(layers_dims:list,init_type:str) :
    """
    내가 다시 작성하는 코드
    
    Arguments :
    layers_dims -- list 각 레이어의 뉴렬수(입력 샘플수 = 입력 피쳐)
    init_type -- str 초기화 타입(제로, 랜덤, he)
    
    Returns :
    parameters -- dict
    """
    np.random.seed(3) 
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L) :
        layer_shape = (layers_dims[l],layers_dims[l-1])
    
        if init_type == 'zeros' :
            w = np.zeros(layer_shape)

        elif init_type == 'random' :
            w = np.random.randn(layer_shape)*10
        
        elif init_type == 'he' :
            w = np.random.randn(layer_shape)* np.sqrt(2 / layers_dims[l - 1])
        
        else :
            assert False

        b = np.zeros((layers_dims[l],1))
        
        parameters['W'+str(l)] = w
        parameters['b'+str(l)] = b
        
        
    return parameters

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def relu_backward(dA, cache,L):
    """
    dA -- 현재 층의 활성화 값A의 미분(기울기 : ∂L/∂A)
    cache -- 파라미터 저장된 변수
    L -- 현재 레이어 층
    """
    
    Z = cache['Z'+str(L)]
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache,L):
    """
    dA -- 현재 층의 활성화 값A의 미분(기울기 : ∂L/∂A)
    cache -- 파라미터 저장된 변수
    L -- 현재 레이어 층
    """
    
    Z = cache['Z'+str(L)]
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def linear_backward(dZ, cache,L):
    A_prev = cache['A'+str(L-1)]        # 이전층의 A(활성화값)
    W = cache['W'+str(L)]
    m = A_prev.shape[1]                 # 이전층의 A가 입력값, m은 입력 노드수

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)            # dz를 만들기 위한 값. A값 되기 전의 Z값에 활용

    return dA_prev, dW, db

def forward_propagation(X, parameters):
    """
    코드 다시 구성하기
    
    Arguments:
    X -- 입력 데이터
    Y -- 라벨 데이터~
    parameters -- 파라미터(w,b)
    
    Returns:
    parameters -- 입력했던 파라미터에 a,z값추가
    """
        
    # retrieve parameters
    L = len(parameters)//2
    a = X
    
    for i in range(1,L+1):
        w = parameters["W"+str(i)]
        b = parameters["b"+str(i)]

        z = np.dot(w,a) +b
        
        if i == L :
            a = sigmoid(z)
        else :
            a = relu(z)
        
        parameters["A"+str(i)] = a
        parameters["Z"+str(i)] = z

    return parameters

def forward_propagation_for_predict(X, parameters):
    """
    predict할때 활용하는 순전파 코드
    
    Arguments:
    X -- 입력 데이터
    parameters -- 파라미터(w,b)
    
    Returns:
    aL -- 마지막 활성화값(y^)
    """
        
    # retrieve parameters
    L = len(parameters)//2
    a = X
    
    for i in range(1,L):
        w = parameters["W"+str(i)]
        b = parameters["W"+str(i)]
        
        z = np.dot(w,a) +b
        
        if i == L-1 :
            a = sigmoid(z)
        else :
            a = relu(z)
        
        parameters["A"+str(i)] = a
        parameters["Z"+str(i)] = z
    
    # 마지막 값 : 시그모이드한 후의 값
    return a

def backward_propagation_update(AL, Y, cache,learning_rate):
    """
    다시 작성
    역전파, 그리고 파라미터 업데이트까지 같이함.
    
    Arguments:
    AL -- 마지막 활성화 값 from forward_propagation()
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()
    learning_rate - learning rate
    
    Returns:
    parameters -- updated parameters
    """
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)          # 브로드캐스팅 오류 방지
    L = len(cache)//4                # 레이어 층
    gradients = {}
    
    # cross-entropy loss 미분식
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # 마지막 레이어는 sigmoid 활성화함수를 사용함.
    dz = sigmoid_backward(dAL,cache,L)
    da_prev,dw,db = linear_backward(dz,cache,L)
    gradients["dA" + str(L)], gradients["dW" + str(L)], gradients["db" + str(L)] = da_prev,dw,db
    
    for i in range(L,2,-1):
        dz = relu_backward(da_prev,cache,i)         # ReLU 활성화함수 미분~~~~
        da_prev,dw,db = linear_backward(dz,cache,i)
        gradients["dA" + str(L)], gradients["dW" + str(L)], gradients["db" + str(L)] = da_prev,dw,db

    # 파라미터 업데이트
    parameters = {}
    for k in range(L) :
        parameters["W" + str(k+1)] = cache["W" + str(k+1)] - learning_rate * gradients["dW" + str(k+1)]
        parameters["b" + str(k+1)] = cache["b" + str(k+1)] - learning_rate * gradients["db" + str(k+1)]
    
    return parameters

def compute_loss(a3, Y):
    
    """
    Implement the loss function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    loss - value of the loss function
    """
    
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    loss = 1./m * np.nansum(logprobs)
    
    return loss

def load_cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    train_set_x = train_set_x_orig/255
    test_set_x = test_set_x_orig/255

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    # Forward propagation
    a3, caches = forward_propagation(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3>0.5)
    return predictions

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


def model_re(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    costs = [] # to keep track of the loss
    layers_dims = [X.shape[0], 10, 5, 1]
    last_L = len(layers_dims)-1
    
    parameters = initialize_parameters(layers_dims,initialization)
    
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        cache = forward_propagation(X, parameters)
        a = cache['A'+str(last_L)]

        # Loss
        cost = compute_loss(a, Y)

        # Backward propagation and update parameters.
        parameters = backward_propagation_update(a, Y, cache,learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict_re(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)     # 저장할 파라미터
    
    # Forward propagation
    aL = forward_propagation_for_predict(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, aL.shape[1]):
        if aL[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p