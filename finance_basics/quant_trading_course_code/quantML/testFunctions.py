ticker ="NIFTY"

start_date="2009-01-01"
end_date="2010-01-01"

options={'qtype':'close',
         'tables':['cm_adjPrice','tradingDays']}
from fetchData import getRawData

tickerDataRaw=getRawData(ticker,start_date,end_date,options)

print tickerDataRaw.head()