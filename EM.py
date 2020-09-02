import numpy as np;
import matplotlib.pyplot as pyplt;
from scipy.stats import multivariate_normal
import pylab as plab
from matplotlib.patches import Ellipse
from numpy import linalg as LA


def plot(points, mu, sigma):

    plab.plot(points[:,0], points[:,1], 'x')
    for i in range(3):
        draw_mean_cov(mu[i],sigma[i])

def draw_mean_cov(mu,sigma):

    def sortedEigenValuesAndVectors(sigma):

        eigenValues, eigenVectors = LA.eigh(sigma)
        sorted_indices = np.argsort(eigenValues) #Returns the indices that would sort an array.
        return eigenValues[sorted_indices], eigenVectors[:, sorted_indices]


    ax = plab.gca() #get current axis

    eigenValues, eigenVectors = sortedEigenValuesAndVectors(sigma)
    theta = np.degrees(np.arctan2(*eigenVectors[:, 0][::-1]))


    width,height = 4 * np.sqrt(abs(eigenValues))
    ellip = Ellipse(xy=mu, width=width, height=height, angle=theta,color='r')

    ax.add_artist(ellip)
    return ellip


def initialize_2D(classes,samples):
    w, h = classes, samples
    result = [[0 for x in range(w)] for y in range(h)]
    return result

def data_generation():

    mean = [[-4, 0], [10, 12], [-10, 12]]
    cov = [[[6, 0], [0, 2]], [[5, 0], [0, 20]], [[20, 2], [2, 5]]]

    x, y = np.random.multivariate_normal(mean[0], cov[0], 500).T

    a, b = np.random.multivariate_normal(mean[1], cov[1], 500).T
    x = np.array(list(x) + list(a))
    y = np.array(list(y) + list(b))

    a, b = np.random.multivariate_normal(mean[2], cov[2], 500).T
    x = np.array(list(x) + list(a))
    y = np.array(list(y) + list(b))

    combined = np.vstack((x, y)).T
    return combined

def initializeParameters(points):

    points = np.array(points)
    a = np.random.randint(0,points.shape[0],3)
    mu = initialize_2D(2,3)
    for i in range(3):
        mu[i][0] = points[a[i]][0]
        mu[i][1] = points[a[i]][1]

    sigma = [np.eye(2)]* 3
    w = np.random.randint(1,1000,3);
    sum = w[0]+w[1]+w[2];
    w = w/sum

    mu = np.array(mu)
    sigma = np.array(sigma)
    return mu,sigma,w;


def Pdfs(points,mu,sigma):
    y = initialize_2D(3, points.shape[0])
    for i in range(3):
        y[i] = multivariate_normal.pdf(points, mu[i], sigma[i]);
    return y

def E_Step(n_matrix,w,points):

    pij = np.zeros((points.shape[0], 3))
    for i in range(3):
        pij[:,i] = w[i]*n_matrix[i]
    pij = (pij.T / np.sum(pij, axis=1)).T # running horizontally across columns (axis 1).


    return pij;

def M_Step(pij,mu,sigma,points):
    points_per_distribution = np.sum(pij, axis = 0) #running vertically downwards across rows (axis 0),
    point_minus_mu = np.zeros((3,points.shape[0], 2))

    for i in range (3):
        sum_x = 0
        sum_y = 0
        for j in range (points.shape[0]):
            sum_x= sum_x+ pij[j][i]*points[j][0]
            sum_y = sum_y + pij[j][i] * points[j][1]

        point_minus_mu[i] = points - mu[i]
        mu[i][0] = sum_x/points_per_distribution[i]
        mu[i][1] = sum_y/points_per_distribution[i]


    for i in range(3):
        sigma[i] = np.dot(np.multiply(point_minus_mu[i].T,pij[:,i]),point_minus_mu[i])
        sigma[i] = sigma[i]/points_per_distribution[i]

    w = points_per_distribution/points.shape[0]

    return mu,sigma,w;


def logLikelihood(n_matrix,w,points):
    temp = np.zeros((points.shape[0], 3))
    for i in range(3):
        temp[:, i] = w[i] * n_matrix[i]

    log_likelihood = np.sum(np.log(np.sum(temp, axis=1))) #running horizontally across columns (axis 1).
    return log_likelihood

def main():

    np.random.seed(20)
    points = data_generation()
    np.random.shuffle(points)

    pyplt.plot(points[:,0], points[:,1], 'x')
    pyplt.axis('equal')
    pyplt.show()

    mu,sigma,w=initializeParameters(points)
    print(mu)
    print(sigma)
    print(w)

    prev =-9999.99
    fig = plab.figure(figsize=(10, 6))
    plot(points, mu, sigma)
    plab.show()

    prev = -9999.99
    while(True):

        n_matrix = Pdfs(points,mu,sigma)

        pij=E_Step(n_matrix,w,points)

        mu,sigma,w = M_Step(pij,mu,sigma,points)

        log_L=logLikelihood(n_matrix,w,points)

        fig = plab.figure(figsize=(10, 6))
        plot(points, mu, sigma)
        plab.show()

        if(np.abs(log_L-prev)<0.001):
            break
        prev = log_L

    print(mu)
    print(sigma)
    print(w)



if __name__ == '__main__':
    main()