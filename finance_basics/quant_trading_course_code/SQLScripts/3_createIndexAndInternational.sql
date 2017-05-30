# We are almost done with setting up our database
# As a final step we need to also insert the historical prices for
# various indices that trade on the NSE 

# Let's create a table for storing the index prices 

create table indices (
name varchar(256),
timestamp date,
open float,
high float,
low float,
close float);

# We'll use 2 data sources for each index. Let's download the data
# for 1 index the NIFTY, and we can then use the same process to download
#and insert other indices 


# the first datasource is Yahoo Finance
# We can bulk load the data from yahoo finance 

Load data local infile
'/Users/swethakolalapudi/Movies/Loonycorn Videos/quantTrading/Nifty/from2008.csv'
into table indices
fields terminated by ','
optionally enclosed by '"'
ignore 1 lines
(@Date,open,high,low,close,@trded,@turnover)
set timestamp=STR_TO_DATE(@Date,"%Y-%m-%d"),name="NIFTY";

# The second data source is from NSE india website
# The NSE allows you to only download only 1 year worth of movements at a 
# time


Load data local infile 
'/Users/swethakolalapudi/Movies/Loonycorn Videos/quantTrading/Nifty/2006.csv'
into table indices
fields terminated by ',' IGNORE 1 lines
(@Date,open,high,low,close,@trded,@turnover)
set timestamp=STR_TO_DATE(@Date,"%d-%b-%Y"), name ="NIFTY";
# Insert the index data also into cmAdjPrice
insert ignore into cmAdjPrice
(select ind,timestamp,close from indices where timestamp not in 
(select distinct timestamp from indices where timestamp not in 
 (select  tDay from tradingDays) and year(timestamp) not in (2006,2016))
);

# Create a table for holding data for international stocls 

create table intStocks (symbol varchar(256), timestamp date, open float, high float, low float, close float);

Load data local infile 
'/Users/swethakolalapudi/Movies/Loonycorn Videos/quantTrading/SNP/amzn.csv'
into table intStocks
fields terminated by ',' IGNORE 1 lines
(@Date,open,high,low,@close,@vol,close)
set timestamp=STR_TO_DATE(@Date,"%Y-%m-%d"), symbol="AMZN";
