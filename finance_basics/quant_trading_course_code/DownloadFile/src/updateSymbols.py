# LEt's now write a script that can read the symbols
# from the csv file that have changed and update them in our
# prod tables

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
fileName = "/Users/swethakolalapudi/Downloads/symbolchange.csv"

# Since this is a csv file, we'll need to read it line by line for the changed symbols

import csv

lineNum=0
# This variable will help us keep track of whether we are reading the header row or a
# normal row

with open(fileName,'rb') as csvfile:
    # the file is opened now. We can get a csv handler from the csv library
    # and use it to read our file

    lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")

    for row in lineReader:
        lineNum = lineNum+1
        if lineNum==1:
            continue

        oldSymbol = row[1]
        newSymbol = row[2]

        print (oldSymbol,newSymbol)

        try:
            c.execute("Update cmProd set symbol=%s where symbol = %s;",(newSymbol,oldSymbol))
        except:
            continue
        finally:
            conn.commit()

c.close()
conn.close()



























