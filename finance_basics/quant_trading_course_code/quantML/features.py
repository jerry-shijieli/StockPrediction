
# We have written functions to construct features
# and assign labels to data points

# Let's now see how to put this all together

# The features will be constructed based on the options passed in by the user
from setup import *
from featureFunctions import *
from labelFunctions import *
from fetchData import  *

def getTrainData(ticker,start_date,end_date,options,assignLabels=labels3):

    start_date=datetime.strptime(start_date,"%Y-%m-%d").date()
    end_date=datetime.strptime(end_date,"%Y-%m-%d").date()

    tickerDataRaw=getRawData(ticker,start_date,end_date,options)
    tickerDataRaw["Return"]=getReturn(tickerDataRaw["Price"])

    tickerDataDaily = tickerDataRaw

    momentumPeriod=25
    jumpPeriod=[3,25]
    valuePeriod=256

    # We store the daily data separately so that it can be used for
    # momentum calculations even if the final datapoints are weekly/monthly

    # Let's compute the returns given the price column
    # Now we setup the data at the right frequency
    if options['freq']>0:
        offset=options['offset']
        if options['freq']==1:

            if offset>0:
                tickerDataRaw=tickerDataRaw[tickerDataRaw["tDayinMonth"]==int(offset)]
            else:
                tickerDataRaw=tickerDataRaw[tickerDataRaw["tDaysleftMonth"]==int(abs(offset))]

            momentumPeriod=75
            jumpPeriod=[2,25]
            valuePeriod=25


        if options['freq']==2:
            tickerDataRaw=tickerDataRaw[tickerDataRaw["tDayinWeek"]==int(abs(offset))]

            momentumPeriod=75
            jumpPeriod=[8,96]
            valuePeriod=25
        # We've filtered the data to the right frequency. For each frequency,
        # The trailing periods used for momentum, jump calculations are different
        # if the frequency is not daily, then recompute the returns

        tickerDataRaw["Return"]=getReturn(tickerDataRaw["Price"])

    # We can compute the labels for these returns
    labels = tickerDataRaw[tickerDataRaw.index>=start_date]["Return"].apply(assignLabels)

    # We are finally ready to construct the features based on the options specified
    # We have an initial dataframe for the features with a dummy column

    featureIndex=tickerDataRaw.index[tickerDataRaw.index>=start_date]
    features = pd.DataFrame(np.empty(featureIndex.size),index=featureIndex,columns=["Dummy"])

    if options['pure']==1:
        # This is simply the price on the previous day
        pure=tickerDataRaw["Price"][1:]
        pureIndex=tickerDataRaw.index[0:-1]
        pure.index=pureIndex
        features=pd.concat([features,pure],axis=1,join="inner")

    if options["cal"]==1:
        calFeatures=getCalFeatures(tickerDataRaw,start_date)
        features = pd.concat([features,calFeatures],axis=1,join="inner")

    if options["history"]==1:
        tickerHistory=getHistory(tickerDataRaw["Return"],start_date)
        features = pd.concat([features,tickerHistory],axis=1,join="inner")

    if options["momentum"]==1:
        momentum=getMomentum(tickerDataDaily,momentumPeriod)

        # This momentum needs to be shifted so that it only uses data from the
        # previous period

        momfeatures=pd.concat([tickerDataRaw,momentum],axis=1,join='inner')
        # This will filter out to the right frequency
        momIndex=momfeatures.index[0:-1]
        momfeatures=momfeatures[1:][momentum.columns]
        momfeatures.index=momIndex

        features=pd.concat([features,momfeatures],axis=1,join="inner")

    if options["value"]==1:
        value=getValue(tickerDataDaily,valuePeriod)

        valFeatures=pd.concat([tickerDataRaw, value], axis=1, join='inner')
        # This will filter out to the right frequency
        valIndex = valFeatures.index[0:-1]
        valFeatures = valFeatures[1:][value.columns]
        valFeatures.index = valIndex

        features = pd.concat([features, valFeatures], axis=1, join="inner")

    if options["jump"]==1:
        jump=getJump(tickerDataRaw,jumpPeriod)
        features = pd.concat([features, jump], axis=1, join="inner")

    if options["prevWeeks"]==1:
        prevWeeks=getPrevWeeks(tickerDataDaily)

        prevWeeksFeatures = pd.concat([tickerDataRaw, prevWeeks], axis=1, join='inner')
        prevWeeksFeatures.sort_index(ascending=False,inplace=True)
        # This will filter out to the right frequency
        prevWeeksIndex = prevWeeksFeatures.index[0:-1]

        prevWeeksFeatures = prevWeeksFeatures[1:][prevWeeks.columns]
        prevWeeksFeatures.index = prevWeeksIndex

        features = pd.concat([features, prevWeeksFeatures], axis=1, join="inner")




    del features["Dummy"]
    # We are done constructing the features, we can now make sure the features and
    # labels are in the same order and return them to the calling function

    features.sort_index(ascending=False,inplace=True)
    labels.sort_index(ascending=False,inplace=True)
    return features,labels

def  getFeatures(ticker,start_date,end_date,options,supportTickers,assignLabels=labels3):

    (features,labels)=getTrainData(ticker,start_date,end_date,options,assignLabels=assignLabels)

    # First we get the features and labels for our main ticker

    if supportTickers is None:
        return features,labels
    # For each of our support tickers
    for (supTicker,supOptions) in supportTickers:
        supOptions["cal"]=0
        supOptions["history"]=0

        tempOptions=deepcopy(options)
        tempOptions.update(supOptions)
        # We update the options based of the features of the support ticker we want
        (tempFeatures,tempLabels) = getTrainData(supTicker,start_date,end_date,tempOptions,assignLabels=assignLabels)
        tempFeatures.columns=supTicker+"_"+tempFeatures.columns
        # We get the features
        features=pd.merge(features,tempFeatures,left_index=True,right_index=True,how="left")
        # Merge the features of the support ticker with those of the original ticker
    features.sort_index(ascending=False,inplace=True)
    labels.sort_index(ascending=False,inplace=True)

    return features,labels


def getReturn(priceSeries):

    returns=np.array(priceSeries[:-1],np.float)/np.array(priceSeries[1:],np.float)-1
    return np.append(returns,np.nan)