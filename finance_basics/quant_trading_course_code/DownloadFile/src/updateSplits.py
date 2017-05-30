# Now we need to do one more thing

# Adjust the stock prices based on any splits that might have occurred.

#  Let's say there was a stock split of 1:2
# This means that 1 of the old stocks = 2 of the new stocks
# In our historical data, we would see a drop of 50% in the stock price
# but this is not due to any market movements.
# In order to adjust for this split, we need to divide all the prices before the
# split date by 2

# So in order to adjust for splits , we need 3 data points
# the old face value of the stock, and the new face value : this will give us the ratio
# by which the old prices need to be adjusted. And the date on which this split occurred.

# We have collected all the splits till date manually from moneycontrol.com
# and put them in an html file

# We'll use BeautifulSoup to parse this html file , and the links to the companies on moneycontrol.cm
# will be used to fetch the NSE symbol for each stock

splitData ="file:///Users/swethakolalapudi/Downloads/qtRough/Splits.html"

from bs4 import  BeautifulSoup
import urllib2

page = urllib2.urlopen(splitData).read()
soup = BeautifulSoup(page)

cells=soup.findAll('td') # Let's understand this one a bit more.
# This will be a list in which all the rows have been concatenated , so every 4th element
# a new row starts

def getSymbol(url):
    page = urllib2.urlopen(url).read()
    soup = BeautifulSoup(page)
    return soup.findAll("div",{"class":"FL gry10"})[0].text.split("|")[1].split(":")[1]
                                                    # The first element with this div class
                                                    # The second symbol in the row is for NSE
                                                    # Then we parse the symbol text which appears after
                                                    # NSE:
# This function will fetch the NSE symbol from a money control company page

# We'll create a list of lists now
# Each list will be the Company name, old fair value, new fair value, split date and symbol

rows =[]
for i in range(len(cells)/4):
    if i==0:
        continue
        # The first row , we can skip over that
    else:
        currentRow=[]
        try:
            for j in range(4):
                currentRow.append(cells[4*i+j].text)
                # Now we have the first 4 elements of our row.
                # The last element is the symbol
            symbol=getSymbol(cells[4*i].a["href"]).strip()
            if symbol=="":
                continue
            currentRow.append(symbol)
            rows.append(currentRow)
        except:
            continue
        print(currentRow)


# Now we have the list of splits
# WE need to iterate through them and compute the adjusted prices

import mysql.connector

config = {
    'user':'qt',
    'password':'password',
    'host':'127.0.0.1',
    'database':'nse'
}

conn= mysql.connector.connect(**config)

c=conn.cursor()

# this will get us a cursor to connect to the database

c.execute('create table cmSplits(company varchar(256), old_fv float,new_fv float, split_date date, symbol varchar(256));')
conn.commit()

from datetime import datetime
for row in rows:
    rowtupe=(row[0],float(row[1]),float(row[2]),datetime.strptime(row[3],"%d-%m-%Y"),row[4])
    c.execute('insert into cmSplits values(%s,%s,%s,%s,%s)',rowtupe)
    conn.commit()


# This has now stored all the splits into our database. We'll read that table, symbol
# by symbol and compute the adjusted prices

# Since we are going to be connecting very often to the database, some
# connection settings might have to be tweaked, so we don't get limited by
# the mySql server
c.execute('set autocommit=1')
c.execute('set global max_allowed_packet=1073741824')
c.execute('select distinct symbol from cmSplits;')
symbolsWithSplits=c.fetchall()

adjustedPricesAll=[]
# This will contain lists, each representing symbol, date and adjusted price
for symbol in symbolsWithSplits:
    symbol=list(symbol)[0]
    # the cursor returns rows in the form of tuples
    print symbol
    c.execute('select symbol,split_date,old_fv,new_fv from cmSplits where symbol=%s;',(symbol,))
    splits=list(c.fetchall())

    c.execute('select symbol, timestamp,close from cmProd where symbol = %s;',(symbol,))
    prices=list(c.fetchall())


    # Now we have a list of splits and all the unadjusted prices for a symbol
    # We'll go through the prices, if the price is for a date before a split
    # We will adjust the price


    for price in prices:
        price =list(price)
        for split in splits:
            if split[1]>=price[1]:
                price[2]=price[2]*split[3]/split[2]
        adjustedPricesAll.append(price)


# Once we are done computing adjusted prices we need to put them back into the database

# Let's create a table which only contains the adjusted closing prices for the CM Stocks


c.execute('create table cmAdjPrice (symbol varchar(256),timestamp date, close float);')
conn.commit()
c.execute('alter table cmAdjPrice add unique index id (symbol,timestamp)')
conn.commit()
# WE have created table to insert the adjusted prices into
c.execute('insert ignore into cmAdjPrice (select symbol, timestamp, close from cmProd;')
conn.commit()
# This is a table with unadjusted prices. Now lets replace the rows where prices need to be
# adjusted


for price in adjustedPricesAll:
    rowTuple=(price[0],price[1],price[2])
    c.execute('replace into cmAdjPrice values(%s,%s,%s)',rowTuple)
    conn.commit()

# Let's run all of this!

































