# We can create a table to hold the NSE bhav copy files 

# The table will need to have columns which map to the file

create database nse;
use nse;

create table cmStaging (
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

# Similarly for the FO bhav copy files 

create table foStaging (
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


















