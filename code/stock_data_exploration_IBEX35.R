library(quantmod)
library(tseries)
library(timeSeries)
library(forecast)
library(xts)
library(readr)
library(ggplot2)

data_dir = '../data/'
data_name= 'IBEX35'
raw_data <- read_csv(paste0(data_dir, data_name, '.csv'), col_names=TRUE)
tsprice <- xts(x=raw_data$Price, order.by = as.Date(raw_data$Date,"%m/%d/%y"), frequency = 255)
summary(tsprice)

plot(tsprice, type='l', main='Price of stock', xlab='Date', ylab='Price')

tsprice_train <- tsprice[index(tsprice)<as.Date("2015-01-01")]
tsprice_test <- tsprice[index(tsprice)>=as.Date("2015-01-01")]

# regular ARIMA model forecasting
fit_model <- auto.arima(tsprice_train)
arima_model <- arima(tsprice_train, order=arimaorder(fit_model))
test_fc_arima <- forecast.Arima(arima_model, h=length(tsprice_test))
plot.forecast(test_fc_arima)

# one-step ahead ARIMA model forecasting
test_dates <- index(tsprice_test)
test_pred <- vector()
for (dt in test_dates) {
  tsprice_train <- tsprice[index(tsprice)<as.Date(dt)]
  arima_model <- arima(tsprice_train, order=arimaorder(auto.arima(tsprice_train)))
  test_pred <- c(test_pred, forecast.Arima(arima_model, h=1)$mean[1])
}
test_fc <- xts(test_pred, order.by=test_dates)
test_mad <- mean(abs(test_fc - tsprice_test)) # Mean Absolute Deviation
test_rmse <- sqrt(mean((test_fc - tsprice_test)^2)) # Root-Mean-Square Error
test_mep <- mean(abs(test_fc - tsprice_test)/tsprice_test*100) # Mean Error Percentage
print("Mean Forecasting Errors:")
print(paste("MAD:", test_mad))
print(paste("RMSE:", test_rmse))
print(paste("MEP:", test_mep, '%'))

par(col='green')
plot(test_fc, main="")
par(new=TRUE)
par(col='black')
plot(tsprice_test, xlab='Date', ylab='Price', main=paste(data_name, "Stock Price Forecasting"))


par(col='green')
plot(c(tsprice_train, test_fc), main="")
par(new=TRUE)
par(col='black')
plot(tsprice, xlab='Date', ylab='Price', main=paste(data_name, "Stock Price Forecasting"))

