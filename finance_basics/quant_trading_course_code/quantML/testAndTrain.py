

from labelFunctions import  *
from setup import  *
from features import *
from fetchData import *

# This function gives us a prediction between [-3,-2,-1,0,1,2,3]
# WE could instead use the probabilities that the classifier provides for each of these
# labels and computed a weighted average signal
def getPredictionsProb(ticker,trainStart,testPeriod,options,supportTickers):
    # This function will take in a test period, for each data point in that
    # period it will give us a prediction in [-3,-2,-1,0,1,2,3]


    # It will use the algorithm that is chosen by the user
    algo = options['algo']
    algo_params=options['algo_params']
    testData=getFeatures(ticker,testPeriod[0],testPeriod[1],options,supportTickers)[0]
    # We get the test data points represented using the features

    # For each test data point a classifier will be trained using the
    # training data from trainStart to testDate-1,

    start_date=trainStart
    dates=pd.Series(testData.index)

    predictions=np.empty(dates.size)

    for i in range(dates.size):
        end_date=str(testData.iloc[i,].name)
        testPoint = testData.iloc[i,]
        print start_date, end_date
        # This is the training period used for the current testPoint

        clf=algo(**algo_params)
        (features, labels)=getFeatures(ticker,start_date,end_date,options,supportTickers)

        # We use an imputer to fill in missing values if any
        imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp=imp.fit(features)

        features_imp=imp.transform(features)
        clf=clf.fit(features_imp, labels)
        classes=clf.classes_ # This gives the class labels
        prediction_proba=clf.predict_proba(imp.transform(testPoint.reshape(1,-1))) # This is the list of probabilities for each class label
        predictions[i]=np.sum(np.multiply(classes,prediction_proba))

    return predictions



def getPredictions(ticker,trainStart,testPeriod,options,supportTickers):
    # This function will take in a test period, for each data point in that
    # period it will give us a prediction in [-3,-2,-1,0,1,2,3]


    # It will use the algorithm that is chosen by the user
    algo = options['algo']
    algo_params=options['algo_params']
    testData=getFeatures(ticker,testPeriod[0],testPeriod[1],options,supportTickers)[0]
    # We get the test data points represented using the features

    # For each test data point a classifier will be trained using the
    # training data from trainStart to testDate-1,

    start_date=trainStart
    dates=pd.Series(testData.index)

    predictions=np.empty(dates.size)

    for i in range(dates.size):
        end_date=str(testData.iloc[i,].name)
        testPoint = testData.iloc[i,]
        print start_date, end_date
        # This is the training period used for the current testPoint

        clf=algo(**algo_params)

        (features, labels)=getFeatures(ticker,start_date,end_date,options,supportTickers)

        # We use an imputer to fill in missing values if any
        imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp=imp.fit(features)

        features_imp=imp.transform(features)
        clf=clf.fit(features_imp, labels)
        predictions[i]=(clf.predict(imp.transform(testPoint.reshape(1,-1))))

    return predictions


# We have the predictions by using this algorithm. WE can now run a backtest, ie
# See what the returns and Sharpe Ratio would have been if we used these predictions to
# execute trades

def backtestResults(ticker,trainStart,testPeriod,options,supportTickers,predictFn=getPredictions):

    predictions=predictFn(ticker,trainStart,testPeriod,options,supportTickers)

    # We also need to get the actual returns
    returns = getAssetReturns(ticker,testPeriod[0],testPeriod[1],options)

    anl_factor=252
    if options["freq"]==1:
        anl_factor=12
    if options["freq"]==2:
        anl_factor=52


    strat_returns=pd.Series(np.multiply(predictions,returns))
    sharpe=np.sqrt(anl_factor)*np.nanmean(strat_returns)/np.nanstd(strat_returns)
    returnSeries=pd.concat([returns,strat_returns],axis=1)
    returnSeries.columns=["Asset_Returns","Strat_Returns"]
    returnSeries.to_csv("/Users/swethakolalapudi/pytest/Returns.csv")
    result=[]
    result.append(sharpe)
    result.append(np.nanmean(strat_returns))
    result.append(np.nanstd(strat_returns))

    result.append(scipy.stats.skew(strat_returns,nan_policy="omit"))
    result.append(float(strat_returns[strat_returns>0].size)/float(strat_returns.size))
    result.append(float(strat_returns[strat_returns < 0].size) / float(strat_returns.size))

    result=pd.DataFrame(result,index=["Sharpe","Mean","Risk","Skew","%up","%Down"])
    print result

    return result









# We need this function that can give us the returns for a ticker between 2 dates
# at the specified frequency

def getAssetReturns(ticker,start_date,end_date,options):

    # This is just like the features function where we construct all the features
    # We just want to return the Return series instead of all the features

    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    tickerDataRaw = getRawData(ticker, start_date, end_date, options)


    # Now we setup the data at the right frequency
    if options['freq'] > 0:
        offset = options['offset']
        if options['freq'] == 1:

            if offset > 0:
                tickerDataRaw = tickerDataRaw[tickerDataRaw["tDayinMonth"] == int(offset)]
            else:
                tickerDataRaw = tickerDataRaw[tickerDataRaw["tDaysleftMonth"] == int(abs(offset))]


        if options['freq'] == 2:
            tickerDataRaw = tickerDataRaw[tickerDataRaw["tDayinWeek"] == int(abs(offset))]

        # We've filtered the data to the right frequency. For each frequency,
        # The trailing periods used for momentum, jump calculations are different
        # if the frequency is not daily, then recompute the returns

    tickerDataRaw["Return"] = getReturn(tickerDataRaw["Price"])

    return tickerDataRaw["Return"][tickerDataRaw.index>=start_date]





def getPredictionsNN(ticker,trainStart,testPeriod,options,supportTickers):
    # This function will take in a test period, for each data point in that
    # period it will give us a prediction in [-3,-2,-1,0,1,2,3]


    # It will use the algorithm that is chosen by the user
    algo = options['algo']
    algo_params=options['algo_params']
    testData=getFeatures(ticker,testPeriod[0],testPeriod[1],options,supportTickers)[0]
    # We get the test data points represented using the features

    # For each test data point a classifier will be trained using the
    # training data from trainStart to testDate-1,

    start_date=trainStart
    dates=pd.Series(testData.index)

    predictions=np.empty(dates.size)

    for i in range(dates.size):
        end_date=str(testData.iloc[i,].name)
        testPoint = testData.iloc[i,]
        print start_date, end_date
        # This is the training period used for the current testPoint

        clf=algo(**algo_params)
        (features, labels)=getFeatures(ticker,start_date,end_date,options,supportTickers,assignLabels=labelId)

        # We use an imputer to fill in missing values if any
        imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp=imp.fit(features)

        features_imp=imp.transform(features)
        clf=clf.fit(features_imp, labels)
        predictions[i] = labels3(clf.predict(imp.transform(testPoint.reshape(1, -1))))
        # With the K-Nearest neighbors the prediction will be the average return of the the
        # nearest neighbours. We convert it to a signal in [-3,-2...]

    return predictions














