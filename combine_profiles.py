import os

import numpy as np
import glob
import argparse

import psrchive

parser = argparse.ArgumentParser()
parser.add_argument("date", help="date of observation in yyyymmdd format")
parser.add_argument("folder", help="subfolder of data/*/Timing/yyyymmdd/\
									that contains folded profiles")											
parser.add_argument("-sband", help="start band number", default="1", type=int)	
parser.add_argument("-eband", help="end band number", default="16", type=int)
parser.add_argument("-o", help="name of output file name", default="all.ar")
args = parser.parse_args()

date, folder = args.date, args.folder
sband, eband, outnamee = args.sband, args.eband, args.o 

def combine_in_time(sband=1, eband=16):

	for band in range(sband, eband+1):
                band = "%02d"%band
		fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)
		print "Processing %s" % band

		file_list = glob.glob(fullpath + "/*_1*.ar")

		data_time_average = []

		for fn in file_list:
			arch = psrchive.Archive_load(fn)
			data = arch.get_data()
			data_time_average.append(data)

		data_time_average = np.concatenate(data_time_average, axis=0).sum(axis=0)

def combine_in_time_(band, date, folder, subint='', outfile='band'):
	fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)

	print outfile + band
	print '%s/*_%s*.ar' % (fullpath, subint)
	os.system("(nice psradd -P %s/*_%s*.ar \
				-o %s.ar; psredit -m -c bw=18.75 %s.ar)" 
		 		% (fullpath, subint, outfile + band, outfile + band))

def combine_allbands(sband=1, eband=16):

	for band in range(sband, eband+1):
		band = "%02d"%band

		for xx in ['10', '11', '12', '13', '14', '15',
				   '16', '17', '18', '19']:
			print xx
			combine_in_time_(band, date, folder, subint=xx, outfile=xx + 'band')
		combine_in_time_(band, date, folder)
        print "Done %s" % band

combine_allbands(sband, eband)
#combine_in_time(sband, eband)

#fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)








