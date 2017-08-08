import datetime as dt
import pandas_datareader.data as wb

quote_code = 'NI225'
start_date = dt.datetime(2013, 1, 1)
end_date = dt.datetime(2015, 12, 31)
ts_data = wb.DataReader(quote_code, data_source='google', start=start_date, end=end_date)

ts_data.to_csv(quote_code+'.csv')