import numpy as np


def GD(data, label, iters, eta):
    # data- nXd matrix  n= num of samples, d=dim of each sample
    # labels- nX1 matrix
    # iters= T num of ITERATIONS
    # eta= defines learning rate
    # returns- dXiter matrix

    n, d = data.shape
    w = np.zeros((d, iters))
    mat = np.zeros((d, iters))

    for t in range(iters):
        vt = get_sub_gradient_descent(w[t], data, label)
        w[t] = w[t] - np.dot(eta, vt)
        #mat =

    return mat

def get_sub_gradient_descent(wt, data, y):
    pass


def main():
    GD()



if __name__ == '__main__':
    main()