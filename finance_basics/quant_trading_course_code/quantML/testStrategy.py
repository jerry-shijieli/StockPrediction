
# Here's what our code should do
# Given a ticker,
# training period, test period
# it should train a classifier using the training period and
# run a backtest on the test period
# The result should be the Sharpe Ratio of our strategy in the test period
# and the other metrics we might need to evaluate the strategy


# We should be able to run this backtest with different options
# The frequency (daily/weekly/monthly) , the features to be used
# which kind of ticker - NSE vs international
# which price is used - open , close, high or low etc

# all these options are captured in a dictionary that will be passed
# from our main function down to any function that needs it

# We'll keep adding to this dict as needed, the first set of options is the
# one that specifies from which tables and how the raw data we need will be fetched.
from setup import *
from testAndTrain import *
options = {'qtype':'close',
           'tables':["cm_adjPrice","tradingDays"],
           'freq':1, # The frequency of trading, daily=0, monthly=1,weekly=2
           'offset':1, # The offset if the period > 1day, ie which trading day in the month/week the strategy will be executed
           'pure':0, # from here we have the features , the returns as is
           'cal':0, # Calendar features
           'history':0, # last 3 periods returns
           'momentum':0, # momentum features
           'jump':0, # jump features
           'value':0, # long term reversal features
           'prevWeeks':1,# Now by turning this to 1 we can run a model which includes previous weeks
           'algo':KNeighborsRegressor,
           'algo_params':{'n_neighbors':5}
           }
# qtype specifies the price type we are running our model on, and the tables are
# those which hold the data for our ticker


# We've written a function that can construct the features for our datapoints,
# But we also might want to construct similar features for tickers that might
# be related and use them as input features

supportTickers= None
   # [("BANKNIFTY",{'pure':0,'momentum':1,'jump':0,'prevWeeks':0})]

ticker="NIFTY"
trainStart="2006-06-01"
testPeriod=["2013-06-01","2016-04-01"]

result=backtestResults(ticker,trainStart,testPeriod,options,supportTickers,predictFn=getPredictionsNN)













