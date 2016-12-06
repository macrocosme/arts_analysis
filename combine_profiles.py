#!/usr/bin/python

# Liam Connor 28 November 2016 
# Code to combine data using psrchiv 

import os
import time

import numpy as np
import glob
import argparse

import psrchive

def dedisperse_folded_spec(fname):
	arch = psrchive.Archive_load(fname+'.ar')
#	arch.set_bandwidth(500000.0)
#	arch.set_centre_frequency(500000.0)
#	arch.set_dispersion_measure(10000.0)
	arch.dedisperse()
#	arch.unload(fname+'dedispersed.ar')
	arch.unload(fname+'.ar')


def dedisperse_manually(fname, dm, p0):
	arch = psrchive.Archive_load(fname+'.ar')
	data = arch.get_data()
	freq_ref = 1390.62 # MHz
	bw = 131.25 # MHz
	nchan = data.shape[-2]
	dt = p0 / data.shape[-1]

	freq = np.linspace(freq_ref - bw/2., freq_ref + bw/2., nchan)

	dm_del = 4.148808e3 * dm * (freq**(-2) - freq_ref**(-2))

	for ii, ff in enumerate(freq):
		dmd = int(round(dm_del[ii] / dt))
		data[:, :, ii] = np.roll(data[:, :, ii], -dmd, axis=-1)

	return data

def plot_me_up(data):
	import matplotlib.pylab as plt 

	data -= np.median(data, axis=-1)[..., None]

	plt.imshow(data, aspect='auto', interpolation='nearest')
	plt.colorbar()
	plt.show()


def combine_in_time(filepath, outfile='band', background=False):
	""" 

	Parameters
	----------
	filepath   : str
		Path to .ar files, should include *

	outfile    : str 
		Name of file to write to, not including '.ar'

	background : bool
		Determines if process is run in the background with '&'

	Returns
	-------

	"""
	if background is False:
		os.system("(nice psradd -P %s -o %s.ar; psredit -m -c bw=18.75 %s.ar)" 
		 		% (filepath, outfile, outfile))
	else:
		os.system("(nice psradd -P %s -o %s.ar; psredit -m -c bw=18.75 %s.ar) &" 
		 		% (filepath, outfile, outfile))

def combine_subints(sband=1, eband=16, 
				outfile='time_averaged', subints='', loop_subints=False):
	"""

	Parameters
	----------
	sband : int 
		start band

	eband : int 
		end band

	outfile : str
		file to write to

	subints : str 
		prefix of file numbers to use, e.g. '01' would use only 
		files '/01*.ar'

	loop_subints : bool
		if True, divide up files into smaller subint chunks and 
		loop over them
	"""
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

		combine_in_time(filepath, outfile=outfile+subints+'band'+band, background=True)

	# Wait for the remaining processes to finish
	while True:
		if os.system('ps -e | grep psradd') == 0:
			print "Waiting for psradd to finish"
			time.sleep(5)
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
	"""
	Parameters
	----------
	fnames : str 
		filenames to use (will actually use fnames+'*.ar')

	outfile : str 
		outfile name
	"""
	print "Combining in frequency"

#	if os.path.exists(outname+folder+'.ar'):

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
	parser.add_argument("-dm", help="dm for manual dedispersion", type=int, default=0)
	parser.add_argument("-o", help="name of output file name", default="all")
	args = parser.parse_args()

	# Unpack arguments
	date, folder = args.date, args.folder
	sband, eband, outname, subints = args.sband, args.eband, args.o, args.subints

	combine_subints(sband, eband, subints=subints, outfile='time_averaged'+folder)

	# for band in range(sband, eband+1):
	# 	print "collecting %s" % band

	# 	band = "%02d"%band
	# 	dedisperse_folded_spec('time_averaged'+folder+subints+'band'+band)

	combine_freq(fnames='time_averaged'+folder, outfile=outname+folder+'.ar')
	#dedisperse_folded_spec(outname+folder)
	p0 = 2.787565229026**-1
	dm = args.dm
	data = dedisperse_manually(outname+folder, dm, p0)
	print data.shape
	data = data.mean(0).mean(0)
	data = data[:len(data)//4*4].reshape(len(data)//4*4, 4, -1)
	plot_me_up(data.mean(1))






