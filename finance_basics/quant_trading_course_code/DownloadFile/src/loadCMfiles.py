# Now that we have all our data from the NSE,
# We can put it into a database so that it's nicely queriable

# We've already created tables for each type of file in a MySQL database.
# We'll need to install and use a mysql connector
# To do so you can run

# $ conda install -c https://conda.anaconda.org/anaconda mysql-connector-python

# The above needs to be run at your command prompt

import mysql.connector

config = {
    'user':'qt',
    'password':'password',
    'host':'127.0.0.1',
    'database':'nse'
}

conn= mysql.connector.connect(**config)

c=conn.cursor()


def insertRows(fileName,c):
    # Given a cursor like th one we created above and a filename this function will insert
    # the data from the file into a table in our database

    delimiter=r','
    # The files need to be csv files

    dateString=r'%d-%b-%Y'
    # This is the format of the dates in the NSE files, we'll use this to convert
    # strings to dates. This format represents a date like 02-JAN-2006

    file=fileName.split("/")[-1]


    if file.startswith("cm"):
        c.execute("Load data local infile %s into table cmStaging fields terminated by %s ignore 1 lines(symbol,series,open,high,low,close,last,prevclose,tottrdqty,tottrdval,@timestamp,totaltrades,isin) SET timestamp = STR_TO_DATE(@timestamp, %s)",(fileName,delimiter,dateString))
    if file.startswith("fo"):
        c.execute("Load data local infile %s into table foStaging fields terminated by %s ignore 1 lines(instrument,symbol,@expiry_dt,strike_pr,option_type,open,high,low,close,settle_pr,contracts,val_inlakh,open_int,chg_in_oi,@timestamp) SET timestamp=STR_TO_DATE(@timestamp,%s), expiry_dt=STR_TO_DATE(@expiry_dt,%s);",(fileName,delimiter,dateString,dateString))

    # That was just to insert into CM table, what if the file was an FO file?

localExtractFilePath="/Users/swethakolalapudi/pytest"
# This is the path where you've saved all your files extracted from zip files

import os

for file in os.listdir(localExtractFilePath):
    if file.endswith(".csv"):
        insertRows(localExtractFilePath+"/"+file,c)
        print "Loaded file "+file+" into database"
        conn.commit()
c.close()
conn.close()




















