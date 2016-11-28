import os
import time

import numpy as np
import glob
import argparse

import psrchive


def combine_in_time_(filepath, band, date, 
					subint='', outfile='band', background=False):

	if background is False:
		os.system("(nice psradd -P %s -o %s.ar; psredit -m -c bw=18.75 %s.ar)" 
		 		% (filepath, outfile + band, outfile + band))
	else:
		os.system("(nice psradd -P %s -o %s.ar; psredit -m -c bw=18.75 %s.ar) &" 
		 		% (filepath, outfile + band, outfile + band))

def combine_subints(sband=1, eband=16):

	# Take subints to be the outerloop 
	for xx in range(10):
		xx = str(xx)
		xx='' 
		for band in range(sband, eband+1):
			band = "%02d"%band
			print "subint %s and band %s" % (xx, band)
			fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)
			filepath = '%s/*%s*.ar' % (fullpath, '_'+xx)
			combine_in_time_(filepath, band, date, 
				subint='_'+xx, outfile=xx, background=True)

	# Wait for the remaining processes to finish
	while True:
		if os.system('ps -e | grep psradd') == 0:
			print "Waiting for psradd to finish"
			time.sleep(10)
		else:
			break


	for band in range(sband, eband+1):
		print "collecting %s" % band
		band = "%02d"%band
		subintfiles = './*band%s.ar' % band
		outfile = 'time_averaged_%s_%s' % (date, band)

		combine_in_time_(subintfiles, band, date, outfile=outfile)

     

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


if __name__=='__main__':
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
	combine_subints(sband, eband)
#combine_in_time(sband, eband)

#fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)








