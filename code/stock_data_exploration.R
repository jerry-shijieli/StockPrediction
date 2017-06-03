library(quantmod)
library(tseries)
library(timeSeries)
library(forecast)
library(xts)
library(readr)
library(ggplot2)

raw_data <- read_csv('../data/IBEX35.csv', col_names=TRUE)
tsprice <- xts(x=raw_data$Close, order.by = as.Date(raw_data$Date,"%m/%d/%y"), frequency = 255)
summary(tsprice)

plot(tsprice, type='l', main='Close price of stock', xlab='Date', ylab='Close Price')