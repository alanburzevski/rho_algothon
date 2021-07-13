#!/usr/bin/env python

# this file contains the implementation of our team's getMyPosition function

import numpy as np
import pandas as pd
import itertools

from statsmodels.tsa.arima.model import ARIMA

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


# Get formatted data (necessary for ARIMA)
# Input: N/A
# Output: Pandas dataframe of formatted data
def getFormattedData():
    ## Reading data
    dataRaw = pd.read_csv('prices250.txt', delimiter = '\s+', header = None).T

    ## Adding column names (for each timepoint)
    dataRaw.columns = ['price' + str(x) for x in range(1, dataRaw.shape[1]+1, 1)]

    ## Adding column representing which instrument
    dataRaw['instrument'] = ['instrument' + str(x) for x in range(1, dataRaw.shape[0]+1, 1)]

    ## Converting to long format
    data = pd.wide_to_long(dataRaw, stubnames = 'price', i = 'instrument', j = 'time')
    data['instrument'] = [x[0] for x in data.index]
    data['time'] = [x[1] for x in data.index]
    data.reset_index(drop = True, inplace = True)

    return data


# Long Term Trading Stratgey
##########################################################################


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
# Input: average returns, inverse scaled variance covariance matrix, and target returns
# Output: weights for optimal portfolio given target returns
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

def minmaxTransform(OldValue, NewMin, NewMax):
    OldMax = max(OldValue)
    OldMin = min(OldValue)

    newValue = np.zeros(len(OldValue))
    for index, value in enumerate(OldValue):
        newValue[index] = (((value - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin

    return newValue

def longTermTrading(weights, window, curPrices):
    transformWeights = minmaxTransform(weights, -1, 1)
    longHoldingsDol = transformWeights*5000
    longHoldings = round(longHoldingsDol/curPrices)

    return longHoldings



# Short Term Trading Stratgey
##########################################################################


# Get holdings from short term trading
# Input: historical data from individual instruments, parameters for ARIMA, window to trade, previous instrument position
# Output: recommended position
def shortTermTrading(instrumentData, params, window, prevPosition):
    
    # Fitting ARIMA and predicting specified number of days
    arimaModel = ARIMA(instrumentData, order = params)
    arimaModelFit = arimaModel.fit()
    predictions = arimaModelFit.forecast(steps = window)
    
    # Obtaining prices and days within window with minimum and maximum days
    predictionsList = [x for x in predictions]
    dayMinBool = [x == min(predictions) for x in predictions]
    dayMaxBool = [x == max(predictions) for x in predictions]
    days = [x for x in range(0, window, 1)]
    dayMin = list(itertools.compress(days, dayMinBool))[0]
    dayMax = list(itertools.compress(days, dayMaxBool))[0]
    priceMin = list(itertools.compress(predictionsList, dayMinBool))[0]
    priceMax = list(itertools.compress(predictionsList, dayMaxBool))[0]

    # Obtaining lists of minimum and maximum days, prices, and non-minimum or maximum days
    minMaxDays = [dayMin, dayMax]
    minMaxPrice = [priceMin, priceMax]
    otherDays = np.setdiff1d(days, minMaxDays)

    # Obtaining position
    position = np.full(shape = (window, 1), fill_value = prevPosition[-1]) # initial position all 0
    position[minMaxDays[0]] = np.floor(5000/minMaxPrice[1])
    position[minMaxDays[1]] = np.ceil(-5000/minMaxPrice[0])
    
    for i in otherDays: # all other days reflect previous day's position
        try:
            position[i] = position[i-1]
        except:
            pass

    return position

# Iteratively obtain holdings
# Input: N/A
# Output: An array of final holdings
def shortTermTradingFinal():
    
    finalPosition = np.empty(shape = (25, 0))
    formattedData = getFormattedData()

    for i in range(1, 101, 1):
        instrumentData = formattedData[formattedData['instrument'] == str("instrument" + str(i))]
        position = np.zeros(shape = (1,1))
        i = 1
        
        while i <= 10:
            newPosition = shortTermTrading(instrumentData = instrumentData, params = (20, 2, 0), window = 25, prevPosition = position)
            position = np.concatenate((position, newPosition), axis=0)
            i = i+1

        finalPosition = np.concatenate((finalPosition, position[1:]), axis = 1) # Removing the first 0

    return finalPosition


# TEST CODE
##########################################################################
givenPrices = readTraining()
returns = returnMeasures(dailyReturns(givenPrices))[0]*250
inverseSigma = np.linalg.inv(sigma(varCov(excessReturns(givenPrices))))
targetReturn = 0.05
print(getWeights(returns, sigma, targetReturn))
# how to separate by spaces and put everything into a matrix


##########################################################################