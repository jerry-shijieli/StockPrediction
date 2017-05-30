

# We are going to write some code that can download data for a ticker
# We will be able to specify the start date, end date and the frequency
# Then the corresponding URL where the price data for this ticker is available
# Using the URL , we'll be able to download a file that contains the prices for our
# ticker.

# We'll do this exercise first for Yahoo Finance.
# We need to understand the URL structure given the start date , end date and frequency.

# Let's study a few different URLs to understand this structure

# ^GSPC ; Daily ; Jan 3 1950 - May 3 2016
# http://real-chart.finance.yahoo.com/table.csv?s=%5EGSPC&a=00&b=3&c=1950&d=04&e=3&f=2016&g=d&ignore=.csv


# ^GSPC ; Weekly ; Jan 3 1950 - May 3 2016
# http://real-chart.finance.yahoo.com/table.csv?s=%5EGSPC&a=00&b=3&c=1950&d=04&e=3&f=2016&g=w&ignore=.csv

# ^GSPC ; Monthly ; Jan 3 1950 - May 3 2016
# http://real-chart.finance.yahoo.com/table.csv?s=%5EGSPC&a=00&b=3&c=1950&d=04&e=3&f=2016&g=m&ignore=.csv

# AAPL ; Daily ; Dec 12 1980 - May 3 2016
# http://real-chart.finance.yahoo.com/table.csv?s=AAPL&a=11&b=12&c=1980&d=04&e=3&f=2016&g=d&ignore=.csv


# Let's now write a function to construct a url for downloading data from Yahoo Finance
from datetime import datetime


def constructYFURL(ticker,start_date,end_date,freq):
    start_date = datetime.strptime(start_date,"%Y-%m-%d").date()
    end_date = datetime.strptime(end_date,"%Y-%m-%d").date()

    s=ticker.replace("^","%5E")

    if start_date.month-1<10:
        a="0"+str(start_date.month-1)
    else:
        a=str(start_date.month-1)
    # a represents the month portion - however the month count starts from 0
    # Also the month always has 2 digits
    b=str(start_date.day)

    c=str(start_date.year)
    # b and c represent the day and year parts of the start date
    if end_date.month - 1 < 10:
        d = "0" + str(end_date.month - 1)
    else:
        d = str(end_date.month - 1)
    # similarly we have to set up the month part for the end date
    e=str(end_date.day)

    f=str(end_date.year)
    # e and f represent the day and year parts of the end date
    g=freq
    # g represents the frequency d = daily, w= weekly, m=monthly

    # Finally let's set up the URL

    yfURL = "http://real-chart.finance.yahoo.com/table.csv?s="+s+"&a="+a+"&b="+b+"&c="+c+"&d="+d+"&e="+e+"&f="+f+"&g="+g+"&ignore=.csv"
    return yfURL


# We'll set up another function to download a file and save it to a local path

def download(filePath,urlOfFile):
    import urllib2

    # We can just use a function from urllib2 to download a url, and save its contents to a
    # local path
    hdr = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.112 Safari/537.36',
           'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset':'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Language':'en-US,en;q=0.8',
           'Accept-Encoding':'none',
           'Connection':'keep-alive'}


    webRequest  = urllib2.Request(urlOfFile,headers=hdr)
    #  We'll pass in a header attribute to the webRequest

    # The rest of our code will be enclosed within a try:/except: pair
    # This will act as a safety net in case we encounter some errors when
    # accessing the web urls or working with the files

    try:
        page=urllib2.urlopen(webRequest)
        # save the contents of the web request in a variable called 'content'
        # These are literally the file form the URL (i.e. what you'd get if you
        # downloaded the URL manually

        content=page.read()

        with open(filePath,"wb") as output:
            output.write(bytearray(content))

    # We are simply reading the bytes in content and writing them to our local file.
    # This way we are agnostic to what kind of file we are trying to download ie zip files , csvs,
    # excel etc

    except urllib2.HTTPError, e:
        # Let's print out the error , if any resulted
        print e.fp.read()


# Let's now write a similar URL constructor but for downloading historical prices from
# the NSE

# The NSE publishes daily price movements at the end of each trading day for the
# cash markets and for the futures markets
# So we'll need to construct a URL for each of these types of files given a trading date

# Let's study the URL for each of these types of files

# Cash Markets :
# https://www1.nseindia.com/content/historical/EQUITIES/2016/MAY/cm02MAY2016bhav.csv.zip

# Futures Markets:
# https://www1.nseindia.com/content/historical/DERIVATIVES/2016/MAY/fo02MAY2016bhav.csv.zip


def constructNSEurl(sectype,day,month,year):
    # This function will expect the day to be an integer
    # The year should be an integer and the month will be a string representing
    # the first 3 letters of the month

    # We need to convert the day to a string with 2 digits
    if day<10:
        day="0"+str(day)
    else:
        day=str(day)

    year = str(year)

    # sectype can either be "CM" or "FO"
    if sectype=="CM":
        nseURL="https://www1.nseindia.com/content/historical/EQUITIES/"+year+"/"+month+"/"+"cm"+day+month+year+"bhav.csv.zip"
    elif sectype=="FO":
        nseURL="https://www1.nseindia.com/content/historical/DERIVATIVES/"+year+"/"+month+"/"+"fo"+day+month+year+"bhav.csv.zip"
    else :
        nseURL=""

    return nseURL

def unzip(localFilePath,localExtractFilePath):

    # First we are checking if the file that's being requested to be unzipped even exists
    import os

    if os.path.exists(localFilePath):
        listOfFiles=[]
        # The zip file might contain more than 1 file, so we maintain a list
        # of all the files we are extracting
        with open(localFilePath,"rb") as fh:
            import zipfile
            zipFileHandler = zipfile.ZipFile(fh)
            # This zipfilehandler from the library zipfile will be able
            # to access and do stuff with files inside our zip file
            for name in zipFileHandler.namelist():
                # We are now iterating through each file in the zip file
                zipFileHandler.extract(name,localExtractFilePath)
                listOfFiles.append(localExtractFilePath+name)
        print "Extracted "+ str(len(listOfFiles)) + " from "+localFilePath




# https://www1.nseindia.com/content/indices/ind_close_all_02052016.csv


def constructIndexURL(day,month,year):
    if day<10:
        day='0'+str(day)
    if month<10:
        month='0'+str(month)
    return 'https://www1.nseindia.com/content/indices/ind_close_all_'+str(day)+str(month)+str(year)+'.csv'




























