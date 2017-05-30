# This module will hold a function to fetch the data for
# a ticker given the options and a start_date, end_date
from setup import *

def getRawData(ticker,start_date,end_date,options,buf=300):

    # The buf variable will be the extra number of days behind the
    # start date that need to be fetched. This is required to get trailing data that
    # might be required to compute momentum, reversal, jump etc

    qtype=options['qtype']
    stockTable=options['tables'][0]
    daysTable=options['tables'][1]

    # We can construct the query required now

    query='select c.timestamp, c.'+qtype+', t.month, t.day, t.dayOfWeek, ' \
          't.tDaysLeftMonth, t.tDayinMonth, t.tDayinWeek from ' +stockTable+' c left join '+daysTable+' t on c.timestamp=t.tDay' \
           ' where c.symbol=%s and c.timestamp<%s and '\
           't.id>=(select min(id) from tradingDays where tDay>=%s)-%s order by timestamp desc'

    params=(ticker,end_date,start_date,buf)

    rawData=getQuery(query,params)
    # getQuery is a function required to connect to the database and execute the query
    # We'll store the returned raw data in a pandas data frame
    tickerDataRaw = pd.DataFrame(rawData,columns=["Timestamp","Price","Month","Day","DayofWeek","tDaysleftMonth","tDayinMonth","tDayinWeek"])

    # Let's make the index of the data frame the date, this will help easily sort, filter  by date
    tickerDataRaw.index=tickerDataRaw["Timestamp"]
    del tickerDataRaw["Timestamp"]

    return tickerDataRaw


def getQuery(query,params):
    config={
        'user':'',
        'password':'',
        'host':'127.0.0.1',
        'database':''
    }

    conn =mysql.connector.connect(**config)
    # We need to import the mysql connector here. Let's import that in a separate module
    # where we can have all the import statements required in common across all modules

    c=conn.cursor()
    c.execute('set autocommit=1')
    c.execute('set global max_allowed_packet=1073741824;')
    c.execute(query,params)
    data= c.fetchall()
    conn.close()
    return data















