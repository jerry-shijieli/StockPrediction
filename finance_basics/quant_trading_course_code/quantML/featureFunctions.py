# This module will contain all the functions
# required to construct the features that we want to use
# using the rawdata

from setup import *

def getCalFeatures(tickerDataRaw,start_date):
    return tickerDataRaw[tickerDataRaw.index>=start_date][["Month","Day","DayofWeek","tDaysleftMonth"]]
# This will add the month, day , day of the week and trading days left in the month to the
# features that will be used to represent at datapoint

def getHistory(returnSeries,start_date,history=3):

    # This will add the last 3 periods returns as features, given the return series of the
    # datapoints in question

    dates = np.array(returnSeries.index)
    start_idx = np.where(dates>=start_date)[0][-1]
    # This fetches us the position of the start date in the series, the series will contain
    # some trailing data behind the start date
    # the start_date input to this function is inherently assumed to be of the type datetime.date
    dateIndex=returnSeries[0:start_idx+1].index
    # This creates an index which only contains the dates we care about, the trailing
    # dates are removed

    historydict={}
    # This will be used to keep track of the series for t-1 , t-2, t-3. Each will become
    # a column in a dataframe that will be returned to the calling function

    for i in range(history):
        returnHistory=pd.Series(returnSeries[i+1:start_idx+2])
        returnHistory.index=dateIndex
        # The above 2 steps assigned to date t , the returns lagged by i periods
        rname="Hist"+str(i+1)
        historydict[rname]=returnHistory
    return pd.DataFrame(historydict)


def getMomentum(tickerDataRawDaily,period=25):

    # This function will return the momentum in 2 different ways
    # The sum of returns over the recent period
    # The sum of returns / standard deviation
    # The period here will be decided by whether we are computing momentum for
    # a daily , weekly , monthly strategy

    returnSeries=tickerDataRawDaily["Return"]

    logReturns=pd.DataFrame({"logRtn":np.log(1+returnSeries)})

    # Shift the series by 1 day, because on a given day, we want to use the
    # returns data till the previous day
    rtnIndex=logReturns.index[0:-1]
    logReturns=logReturns[1:]
    logReturns.index=rtnIndex

    # We'll use a rolling_apply function to compute the sum of returns
    # This function will apply on a window whose right edge is the cell under consideration
    # For this function to work properly we need to apply it on a series sorted in ascending
    # order of date

    logReturns.sort_index(ascending=True,inplace=True)

    logReturns["Mom_1"]=pd.rolling_apply(logReturns["logRtn"],period,np.nansum,min_periods=2)
    logReturns["Mom_2"]=logReturns["Mom_1"]/pd.rolling_apply(logReturns["logRtn"],period,np.nanstd,min_periods=2)

    # sort the dataframe back into descending order
    logReturns.sort_index(ascending=False,inplace=True)
    del logReturns["logRtn"]

    return logReturns


def getValue(tickerDataRaw,period=256):
    return -1*getMomentum(tickerDataRaw,period)
# This is the reversal function. It is exactly the reverse of momentum, except that it is
# computed over a longer time period


def getJump(tickerDataRaw,period=[3,25]):

    # For jump we compare the very short term recent past with a longer period

    returnSeries=tickerDataRaw["Return"]
    # The first few steps are exactly the same as for momentum
    logReturns = pd.DataFrame({"logRtn": np.log(1 + returnSeries)})
    # Shift the series by 1 day, because on a given day, we want to use the
    # returns data till the previous day
    rtnIndex = logReturns.index[0:-1]
    logReturns = logReturns[1:]
    logReturns.index = rtnIndex

    # We'll use a rolling_apply function to compute the sum of returns
    # This function will apply on a window whose right edge is the cell under consideration
    # For this function to work properly we need to apply it on a series sorted in ascending
    # order of date

    logReturns.sort_index(ascending=True, inplace=True)

    logReturns["Jmp_1"]=pd.rolling_apply(logReturns["logRtn"],period[0],np.nanmean,min_periods=2)-\
                        pd.rolling_apply(logReturns["logRtn"],period[1],np.nanmean,min_periods=2)
    logReturns["Jmp_2"]=logReturns["Jmp_1"]/pd.rolling_apply(logReturns["logRtn"],period[1],np.nanstd,min_periods=2)

    # We have 2 jump measures here, using the difference of returns in the 2 periods,
    # and in the second we normalize this measure

    del logReturns["logRtn"]
    logReturns.sort_index(ascending=False, inplace=True)
    return logReturns


# We'll write a function to compute a categorical variable

def getPrevWeeks(tickerDataRaw):

    period=24 # This specifies that we are using the last 4 weeks . We take each "week" to be a period of
    # 6 trading days

    returnSeries =tickerDataRaw["Return"]
    # The first few steps are exactly the same as for momentum
    logReturns = pd.DataFrame({"logRtn": np.log(1 + returnSeries)})
    # Shift the series by 1 day, because on a given day, we want to use the
    # returns data till the previous day
    rtnIndex = logReturns.index[0:-1]
    logReturns = logReturns[1:]
    logReturns.index = rtnIndex

    # We'll use a rolling_apply function to compute the sum of returns
    # This function will apply on a window whose right edge is the cell under consideration
    # For this function to work properly we need to apply it on a series sorted in ascending
    # order of date

    logReturns.sort_index(ascending=True, inplace=True)

    # WE have the same set of steps to prepare the returns as for momentum and jump

    logReturns["prevWeeks"]=pd.rolling_apply(logReturns["logRtn"],period,daysToWeeks,min_periods=2)

    del logReturns["logRtn"]

    return logReturns

   # The function daysToWeeks is where the trends in the past 4 weeks are identified and coded into
   # a 4 digit integer
   # The rolling apply will only work on functions which take a series and return an integer
def daysToWeeks(dailyRtns):

    weekSignals="0"
    for i in range(dailyRtns.size/6):
        weeklyRtn=np.exp(np.sum(dailyRtns[(i*6):i*6+6]))-1
        weekSignals=weekSignals+str(weektype(weeklyRtn))
    return int(weekSignals)

def weektype(weeklyRtn):
    if abs(weeklyRtn)<0.015:
        return 1
    elif weeklyRtn>0.015:
        return 2
    else:
        return 3




























