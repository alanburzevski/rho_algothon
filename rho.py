#!/usr/bin/env python

# this file contains the implementation of our team's getMyPosition function

import numpy as np

# global constants / variables
nDays = 250
nInst = 100

currentPos = np.zeros(nInst)

# Dummy algorithm to demonstrate function format.
def getMyPosition (prcSoFar):
    global currentPos
    (nins,nt) = prcSoFar.shape

    # Important Variables
    givenPrices = readTraining()

    # Day 1, set initial positions?
    if nt == 1:
        return initialPos(prcSoFar)

    rpos = np.array([int(x) for x in 1000 * np.random.randn(nins)])
    currentPos += rpos
    # The algorithm must return a vector of integers, indicating the position of each stock.
    # Position = number of shares, and can be positve or negative depending on long/short position.
    return currentPos



# Helper functions

# Function to determine the starting position
# input: matrix of shape (100, 1) containing instruments' prices on day 1
# output: integer vector of shape (100, 1); i'th index contains start position of instrument i
def initialPos (prcSoFar):
    # TODO: return optimal portfolio obtained from training data + predictions
    return 0


# mooooore helpers
# Reads in first 250 days' data into numpy array givenPrices
# Input: N/A
# Output: matrix of price history from training data
def readTraining ():
    f = open("prices250.txt", "r")
    givenPrices = []
    for line in f:
        row = [float(num) for num in line.split()]
        givenPrices.append(row)
    givenPrices = np.array(givenPrices)
    # print(givenPrices.shape)
    return givenPrices

# Given the price history, output daily percentage price change matrix
# Input: givenPrices (price history)
# Output: matrix of daily percentage change; dailyReturns[i-1][j-1] = percentage change of instrument j between days i and i-1
def dailyReturns (givenPrices):
    logChange = []
    # loop through each instrument
    for inst in givenPrices.T:
        logChangeInst = np.empty((nDays, 1))
        # logChangeInst = np.zeros(nDays)
        for day in range(0, nDays - 1):
            logChangeInst[day] = np.log(inst[day + 1] / inst[day])
        logChange.append(logChangeInst)
    logChange = np.array(logChange)
    # print(logChange.T.shape)
    return logChange.T

# Given the daily price changes, output the avg. daily return, SD, variance for each instrument
def returnMeasures (dailyReturns):
    measures = np.empty((3, nInst))
    measures[0] = [np.average(inst) for inst in dailyReturns.T]
    measures[1] = [np.std(inst, ddof=1) for inst in dailyReturns.T]
    measures[2] = [np.var(inst, ddof=1) for inst in dailyReturns.T]
    return measures

# Given the price history, output the excess returns matrix
# Input: givenPrices (price history)
# Output: matrix of excess returns; excessReturns[i-1][j-1] = excess returns of instrument j on day i
def excessReturns (givenPrices):
    excessReturns = []
    # get the required matrices for further computation
    Returns = dailyReturns(givenPrices)
    measures = returnMeasures(Returns)

    for inst in range(nInst):
        instExcess = np.empty(nDays - 1) 
        instReturns = (Returns.T)[inst]
        for i in range (0, nDays - 1):
            instExcess[i] = instReturns[i] - measures[0][inst]
        excessReturns.append(instExcess)

    excessReturns = np.array(excessReturns)
    print(excessReturns.T.shape)
    return excessReturns.T

# Calculate the variance covariance matrix
# Input: excess returns
# Output: variance covariance matrix
def varCov (givenExcess):
    varCovMat = np.matmul(givenExcess.T, givenExcess)/(249 - 1) # -1 because sample std
    return varCovMat

# Calculate the scaled variance covariance matrix
# Input: variance covariance matrix
# Output: scaled variance covariance matrix
def sigma (varCovMat):
    sigmaMat = varCovMat*250
    return sigmaMat

# Calculate weights
# Input: 
# Output: weights
def getWeights(returns, inverseSigma, targetReturn):
    ones = np.ones(nInst)
    A = np.matmul(np.matmul(ones, inverseSigma), ones.T)
    B = np.matmul(np.matmul(ones, inverseSigma), returns.T)
    C = np.matmul(np.matmul(returns, inverseSigma), returns.T)
    delta = A * C - B**2
    lam = (C - targetReturn*B)/delta
    gam = (targetReturn*A - B)/delta

    weightsTerm1 = lam * np.matmul(inverseSigma, ones.T)
    weightsTerm2 = gam * np.matmul(inverseSigma, returns.T)
    weights = weightsTerm1 + weightsTerm2
    return(weights)

excessReturns(readTraining())

# TEST CODE
##########################################################################
givenPrices = readTraining()

print(excessReturns(givenPrices))
# how to separate by spaces and put everything into a matrix


##########################################################################