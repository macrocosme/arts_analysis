# 
#
#
#
#
#
#
#
#

import os
import time

import numpy as np
import glob
import argparse

import psrchive


def combine_in_time(filepath, subint='', outfile='band', background=False):

	if background is False:
		os.system("(nice psradd -P %s -o %s.ar; psredit -m -c bw=18.75 %s.ar)" 
		 		% (filepath, outfile, outfile))
	else:
		os.system("(nice psradd -P %s -o %s.ar; psredit -m -c bw=18000.75 %s.ar) &" 
		 		% (filepath, outfile, outfile))

def combine_subints(sband=1, eband=16, 
				outfile='time_averaged', subints='', loop_subints=False):

	# # Take subints to be the outerloop 
	# for xx in range(4):
	# 	xx = str(xx)

		# for band in range(sband, eband+1):
		# 	band = "%02d"%band
		# 	print "subint %s and band %s" % (xx, band)
		# 	fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)
		# 	filepath = '%s/*%s*.ar' % (fullpath, '_0'+xx)
		# 	combine_in_time_(filepath, band, date, 
		# 		subint='_'+xx, outfile=xx+'band'+band, background=True)

	for band in range(sband, eband+1):
		band = "%02d"%band
		print "subint %s and band %s" % (subints, band)
		fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)
		filepath = '%s/*%s*.ar' % (fullpath, '_'+subints)
		
		flist = glob.glob(filepath)
		print "Processing total of %d files\n" % len(flist)

		combine_in_time(filepath,
			subint='_'+subints, outfile=outfile+subints+'band'+band, background=True)

	# Wait for the remaining processes to finish
	while True:
		if os.system('ps -e | grep psradd') == 0:
			print "Waiting for psradd to finish"
			time.sleep(2)
		else:
			break

	if loop_subints is False:
		return

	for band in range(sband, eband+1):
		print "collecting %s" % band
		band = "%02d"%band

		subintfiles = './*band%s.ar' % band
		outfile_full = '%s_%s_%s' % (outfile, date, band)

		combine_in_time(subintfiles, outfile=outfile_full, background=True)

def combine_freq(fnames, outfile='all.ar'):
	print "Combining in frequency"
	os.system('nice psradd -P -m phase -R %s*.ar -o %s' % (fnames, outfile))

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("date", help="date of observation in yyyymmdd format")
	parser.add_argument("folder", help="subfolder of data/*/Timing/yyyymmdd/\
										that contains folded profiles")											
	parser.add_argument("-sband", help="start band number", default="1", type=int)	
	parser.add_argument("-eband", help="end band number", default="16", type=int)
	parser.add_argument("-subints", 
	       help="only process subints starting with parameter. e.g. 012\
	       would analyze only *_012*.ar files", 
	       default="")
	parser.add_argument("-o", help="name of output file name", default="all")
	args = parser.parse_args()

	# Unpack arguments
	date, folder = args.date, args.folder
	sband, eband, outname, subints = args.sband, args.eband, args.o, args.subints

	combine_subints(sband, eband, subints=subints, outfile='time_averaged'+folder)
	combine_freq(fnames='time_averaged'+folder, outfile=outname+folder+'.ar')









