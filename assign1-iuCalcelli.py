
# Created on: January 31, 2016
# Author: Caleb Elliott
# Purpose: Assignment 1 for CSCI 48400 Data Mining
import sys
import numpy as np
from numpy import linalg as LA
import fileinput

# Q1: Write a function to Compute the sample covariance matrix as inner products
# between the columns of the centered data matrix (see equation 2.31).
# Show that the result from your function matches the one using numpy.cov function.

# function to return the centered data matrix for calculating the sample covariance matrix
# creates a mean vector using numpy function np.mean
# creates an array of ones and then subtracts the prodct of the array of ones
# and the meanVector transposed from the data
def findCenteredDataMatrix(data):
    meanVector = np.mean(data, axis = 0, dtype = float)
    x, y = np.shape(data)
    arrayOfOnes = np.ones((x,y))
    centeredDataMatrix = data - arrayOfOnes * meanVector.T
    return centeredDataMatrix

# finds the covarince matrix by dividing one by the float of x and mupltiplying
# it by the dot prodct of the centeredDataMatrix
def findCovarianceMatrix(data):
    x, y = np.shape(data)
    centeredDataMatrix = findCenteredDataMatrix(data)
    covarianceMatrix = 1/(float(x)) * np.dot(centeredDataMatrix.T, centeredDataMatrix)
    return covarianceMatrix

# Q2: Use linalg.eig to find the first two dominant eigenvectors, and compute the projection of
# data points on the subspace spanned by these two eigenvectors. Now, compute the variance of the
# datapoints in the projected subspace using the subroutine that you wrote for Q
def findTwoDom(data,covarianceMatrix):
    eigVal, eigVec = LA.eig(np.array(covarianceMatrix))
    sortedEigenVal = np.argsort(eigVal)

    index1 = sortedEigenVal[len(sortedEigenVal) - 1]
    index2 = sortedEigenVal[len(sortedEigenVal) - 2]

    vector1 = eigVec.T[index1]
    vector2 = eigVec.T[index2]

    mergedVector_T = np.vstack((vector1, vector2)).T

    projection = data.dot(mergedVector_T)

    return np.around((findCovarianceMatrix(projection)), decimals = 4)

def findDecomp(data,covarianceMatrix):
    eigVal, eigVec = LA.eig(np.array(covarianceMatrix))
    reshapEigVal = np.reshape(eigVal, (1,10))
    arrayOfOnes = np.ones((10, 1))
    newEigVal = np.dot(arrayOfOnes, reshapEigVal)
    diagnalOfEigVal = np.diag(np.diag(newEigVal))
    print diagnalOfEigVal, "\n"
    print eigVec.T, "\n"

    return eigVec


def PCA(covarianceMatrix):
    eigVal, eigVec = LA.eig(np.array(covarianceMatrix))

    totalVariance = np.sum(eigVal)

    variance = 0
    counter = 0

    for i in eigVal:
        variance = variance + i
        counter = counter +1
        if((variance/float(totalVariance)) > float(.9)):
            break

    sortedEigenVal = np.argsort(eigVal)

    index1 = sortedEigenVal[len(sortedEigenVal) - 1]
    index2 = sortedEigenVal[len(sortedEigenVal) - 2]
    index3 = sortedEigenVal[len(sortedEigenVal) - 3]
    index4 = sortedEigenVal[len(sortedEigenVal) - 4]

    vector1 = eigVec.T[index1]
    vector2 = eigVec.T[index2]
    vector3 = eigVec.T[index3]
    vector4 = eigVec.T[index4]

    mergedVector_T = np.vstack((vector1, vector2, vector3, vector4)).T

    centeredDataMatrix = findCenteredDataMatrix(covarianceMatrix)

    reducedMatrix = np.dot(centeredDataMatrix, mergedVector_T)

    firstTen = []

    for i in range(0, 10):
        firstTen.append(reducedMatrix[i])
    return np.asarray(firstTen)

def main():


    #inputFile = np.loadtxt("magic04data.txt", delimiter = ",", usecols = (0,1,2,3,4,5,6,7,8,9))
    commandArg = sys.argv[-1]
    print commandArg
    inputFile = np.loadtxt(commandArg, delimiter = ",",usecols = (0,1,2,3,4,5,6,7,8,9))
    data = inputFile

    dataWidth = len(data[0])

    print "\n_____Question 1_____\n"
    print "The custom covarience matrix is: \n"
    covarienceMatrix = findCovarianceMatrix(data)
    print covarienceMatrix, "\n"
    print "The difference matrix of the custom covariance matrix and numpys: \n"
    differenceMatrix = np.around(covarienceMatrix - np.cov(data.T, bias = 1), decimals = 4)
    print differenceMatrix, "\n"

    print "\n_____Question 2_____\n"
    print "The variance of the datapoints in the projected subspace is: \n"
    print findTwoDom(data,covarienceMatrix), "\n"

    print "\n_____Question 3_____\n"
    print "covariance matrix in its eigendecompositionform: \n"
    eigVec = findDecomp(data,covarienceMatrix)
    print "The eigenvectors are: \n"
    print eigVec
    print "\n\n_____Question 5_____\n"
    firstTen = PCA(covarienceMatrix)
    print "The first ten points of the reduced Matrix are:\n"
    print firstTen

if __name__ == "__main__":
    main()
