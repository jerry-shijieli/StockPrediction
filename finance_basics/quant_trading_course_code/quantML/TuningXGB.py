from setup import *

from xgboost.sklearn import XGBClassifier

from hyperopt import hp

from hyperopt import fmin, tpe, hp, STATUS_OK,Trials


# We want to tune the Gradient boosted classifier, so that we find the optimum set of parameters

# For that there are 2 things we need to define

# A function that will be minimized by hyperopt

from testAndTrain import *
def score(params):
    # We'll write a function that will return the Sharpe ratio for a strategy
    # using the NIFTY

   # The below code is what we normally use to test a strategy
   # We want the classifier to be XGBClassifier and the params to be what we pass in

    params['n_estimators']=int(params['n_estimators'])

    options = {'qtype': 'close',
               'tables': ["cm_adjPrice", "tradingDays"],
               'freq': 0,  # The frequency of trading, daily=0, monthly=1,weekly=2
               'offset': 1,
               # The offset if the period > 1day, ie which trading day in the month/week the strategy will be executed
               'pure': 0,  # from here we have the features , the returns as is
               'cal': 0,  # Calendar features
               'history': 0,  # last 3 periods returns
               'momentum': 0,  # momentum features
               'jump': 0,  # jump features
               'value': 0,  # long term reversal features
               'prevWeeks': 1,  # Now by turning this to 1 we can run a model which includes previous weeks
               'algo': XGBClassifier,
               'algo_params': params
               }

    supportTickers = None

    ticker = "NIFTY"
    trainStart = "2009-01-01"
    testPeriod = ["2013-06-01", "2016-04-01"]
    print "Training with params"
    result = backtestResults(ticker, trainStart, testPeriod, options, supportTickers)
    score = result[0]
    print params

    print "\t Score {0}".format(score)
    return {'loss':-score,'status':STATUS_OK}
 # The function returns -score as we want to maximize Sharpe but hyperopt will try to minimize



# A Search space with all the combinations over which the function will be minimized
space ={
    'n_estimators':hp.quniform('n_estimators',100,1000,1),
    'learning_rate':hp.quniform('learning_rate',0.025,0.5,0.025),
    'max_depth':hp.quniform('max_depth',1,13,1),
    'min_child_weight': hp.quniform('min_child_weight',1,6,1),
    'subsample': hp.quniform('subsample',0.5,1,0.05),
    'gamma':hp.quniform('gamma',0.5,1,0.05),
    'colsample_bytree':hp.quniform('colsample_bytree',0.5,1,0.05),
    'nthread':6,
    'silent':1
}

trials=Trials()

best=fmin(score,space,algo=tpe.suggest,trials=trials,max_evals=250)

print best




















