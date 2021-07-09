# SIG x FinTechSoc Algothon 2021
Contains evaluation program, relevant price data and an example algorithm/submission file for the Algothon. We'd highly recommend cloning this repo to use a base for your algorithm development, and to make it easier for submission.

**Good luck!**

# Case Brief
*Find this case brief as well as more information about our Algothon on our Information Hub, located [here](https://www.notion.so/Algothon-2021-Information-Hub-3b65973a4a694d07952ff55706a6b1aa).*

## Task

Implement a function *getMyPosition()* which

- Takes as input a NumPy array of the shape *nInst* x *nt*.
    - nInst = 100 is the number of instruments.
    - nt is the number of days for which the prices have been provided.
- Returns a vector of desired positions.
    - i.e. This function returns a NumPy vector of integers. This integer denotes your daily position per instrument in the universe. With 100 instruments, we can expect this function to return 100 integers each time it is called.

## Data

All required data has been generated by us, and is available in [this GitHub repository](https://github.com/oniewankenobi/algothon21). 

- Our simulated trading universe consists of several years of daily price data, spanning 100 instruments.
- The instruments are numbered from 0 to 99, and days go chronologically from 0 onwards such that p[inst, t] indicates the price of the instrument *inst* on day *t*.
- The price data file contains a NumPy array of the shape *nInst x nt.*
    - *nInst* = number of instruments, *nt* = number of days.

In the preliminary round, teams will be provided the first 250 days of price data to be used as training data. This can be found in *prices250.txt.*

- Preliminary round algorithms will be assessed on data from days 251 - 500.
- Successful teams will then receive price data and results from preliminary evaluation.
- Final round algorithms will be assessed on remaining future price data.

## The Algorithm

### Format

Algorithms must be contained in a file titled *[teamName].py.* 

- This file must contain a function *getMyPosition().*
- *getMyPosition()* must take in the daily price data, and output a vector of integer positions - the numbers of shares desired for each stock as the total final position after the last day.
- *getMyPosition()* must be in the global scope of the file called *[teamName].py* and have the appropriate signature.
    - When *getMyPosition()* is called, we will trade position differences from the previous position **at the most recent price, buying or selling.** 
    - Consider the case where your last position was +30, and the new stock price is $20. If your new position is +100, *eval* will register this as buying 70 **extra** shares at $20 a share. If your new position is -200, *eval* will sell 230 shares also at $20 a share.

### **Considerations**

- A commission rate of 50 bps (0.0050) can be assumed, meaning you will be charged commission equating 0.0050 * *totalDollarVolumeTraded*. This will be deducted from your PL.
- Positions can be long or short (i.e. the integer positions can be either positive or negative).
- Teams are limited to a $10k position limit per stock, positive or negative. The $10k limit cannot be breached at the time of the trade.
    - This position limit may technically be exceeded in the case that exactly $10k worth of a stock is bought, and stock goes up the next day - this is fine.
    - However, given this occurs, the position must be slightly reduced to be no greater than $10k by the new day's price.
    - Note: *eval.py* contains a function to clip positions to a maximum of $10k. This means that if the price is $10 and the algorithm returns a position of 1500 shares, *eval.py* will assume a request of 1000 shares.
- Buying and selling is always done at the same price, and liquidity is unlimited (except by the $10k position limit).

### **Assessment Benchmarks**

The program we will use to evaluate your algorithm is provided in *eval.py.* Ensure your code runs against this file. 

Metrics used to quantitatively assess your submission will include:

- PL (daily and mean),
- Return (PL / *dollarVolumeTraded*),
- Sharpe Ratio, and
- Trading volume.

Your algorithms will be assessed against *unseen, future* price data of the same 100 instruments within the provided simulated trading universe.


