
# We have created a database and dumped NSE historical price
# data into that database 

select * from cmStaging limit 10;
# This table contains the cm Historical prices 

select * from foStaging limit 10; 
# This table contains the historical prices for futures and options traded
# on the NSE 

# Let's now do some data cleaning and preparation 
# First order of business is to remove duplicates if any in the data 

# For this we'll create a new set of tables for the cm and fo historical prices
# These tables will have a unique index on the set of columns that 
# uniquely define a row 

# These tables will have the exact same schema though, as our original
# tables 

create table cmProd (
symbol varchar(256),
series varchar(256),
open float,
high float,
low float,
close float,
last float,
prevclose float,
tottrdqty float,
tottrdval float,
timestamp date,
totaltrades float,
isin varchar(256)
);

create table foProd (
instrument varchar(256),
symbol varchar(256),
expiry_dt date,
strike_pr float,
option_type varchar(256),
open float,
high float,
low float,
close float,
settle_pr float,
contracts float,
val_inlakh float,
open_int float,
chg_in_oi float,
timestamp date);


alter table cmProd add constraint symbol_day unique(symbol,series,timestamp);

# This will make sure that the combination of these 3 values 
# is unique within the rows of the table 

insert ignore into cmProd(select * from cmStaging); 
# This will insert all rows from cmStaging into cmProd. 
# However, whenever it encounters duplicates, it will ignore the 
# duplicate row 

# Similarly we can create an index on foProd and insert the data 

alter table foProd add unique index id (instrument, symbol, expiry_dt,strike_pr,option_type,timestamp);

# This is another way to add a unique index on a table 

insert ignore into foProd (select * from foStaging);






































