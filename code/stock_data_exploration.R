library(quantmod)
library(tseries)
library(timeSeries)
library(forecast)
library(xts)
library(readr)
library(ggplot2)

raw_data <- read_csv('../data/IBEX35.csv', col_names=TRUE)
tsprice <- xts(x=raw_data$Price, order.by = as.Date(raw_data$Date,"%m/%d/%y"), frequency = 255)
summary(tsprice)

plot(tsprice, type='l', main='Price of stock', xlab='Date', ylab='Price')

tsprice_train <- tsprice[index(stock_log_return)<as.Date("2015-01-01")]
tsprice_test <- tsprice[index(stock_log_return)>=as.Date("2015-01-01")]

fit_model <- auto.arima(tsprice_train)

arima_model <- arima(tsprice_train, order=c(0,1,0))

test_pred <- forecast.Arima(arima_model, h=length(tsprice_test))

plot.forecast(test_pred)