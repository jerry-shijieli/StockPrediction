# Let's now write a script to download and unzip all the files from 2006 onwards

from download import *
import time

listOfMonths = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
listOfYears= [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]

for year in listOfYears:
    for month in listOfMonths:
        for dayOfMonth in range(31):

            day=dayOfMonth+1
            # range(31) will create a sequence starting from 0 but dates start from 1
            # so let's add 1

            nseURL=constructNSEurl("CM",day,month,year)
            fileName="cm" + str(day) + month +str(year)+"bhav.csv.zip"

            localFilePath = "/Users/swethakolalapudi/pytest/"

            download(localFilePath+fileName,nseURL)
            unzip(localFilePath+fileName,localFilePath)

            # This will first download and then unzip our file in the given location

            time.sleep(10)
            # We give it some time between each download request so we don't inadvertently
            # overwhelm the NSE website


































































