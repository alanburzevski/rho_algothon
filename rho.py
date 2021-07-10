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
def readTraining ():
    f = open("prices250.txt", "r")
    givenPrices = []
    for line in f:
        row = [float(num) for num in line.split()]
        givenPrices.append(row)
    givenPrices = np.array(givenPrices)
    # print(givenPrices.shape)
    return givenPrices

# Given the price history, output the excess returns matrix
def excessReturns (givenPrices):
    pass

# Given the price history, output daily percentage price change matrix
# Sorry for bad code I am noob at python
def dailyPChange (givenPrices):
    pChange = []
    # loop through each instrument
    for inst in givenPrices.T:
        pChangeInst = np.zeros(nDays)
        for day in range(1, nDays):
            pChangeInst[day] = inst[day] / inst[day - 1] - 1
        pChange.append(pChangeInst)
    pChange = np.array(pChange)
    # print(pChange.T.shape)
    return pChange.T

# TEST CODE
##########################################################################
givenPrices = readTraining()
print(dailyPChange(givenPrices))

# how to separate by spaces and put everything into a matrix


##########################################################################