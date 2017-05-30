
from download import constructYFURL
from download import  download, unzip
from download import constructNSEurl
# Let's take our function for a spin

ticker = "^GSPC"

start_date="2015-01-01"

end_date = "2016-01-01"

freq = "d"

yfURL = constructYFURL(ticker,start_date,end_date,freq)

print yfURL

# Let's try out our download function

localFilePath="/Users/swethakolalapudi/pytest/gspc.csv"

download(localFilePath,yfURL)

nseURL= constructNSEurl("CM",2,"MAY",2016)

print nseURL

nseFilePath="/Users/swethakolalapudi/pytest/nsebhav.csv.zip"

download(nseFilePath,nseURL)


# the download function gave an error. The NSE doesn't allow any program
# to download it's files directly, but we can get around this. We just need to
# make the NSE feel that it's a human and not a machine trying to do the download.

# Now let's try to download thatfile again


# The NSE file is a zip file so it needs to be unzipped and the contents written to csv

localExtractFilePath="/Users/swethakolalapudi/pytest/"

unzip(nseFilePath,localExtractFilePath)




















