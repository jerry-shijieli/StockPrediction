# We are almost done with setting up our database.
# AS a final step we need to also insert the historical prices for various
# indices that trade on the NSE

# NIFTY, BANKNIFTY etc.
# Just like the cm and fo files, there is a daily file published by the NSE with
# index open, low, high ,close for all these indices.

# Let's first define a function to construct the url for this file
# https://www1.nseindia.com/content/indices/ind_close_all_03052016.csv

from download import *

list_of_years=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]

localDir='/Users/swethakolalapudi/pytest/'
for year in list_of_years:
    for month in range(12):
        for day in range(31):
            url=constructIndexURL(day+1,month+1,year)
            fileName=url.split("/")[-1]
            download(localDir+fileName,url)
