#!/usr/bin/python

# Liam Connor 28 November 2016 
# Code to combine data using psrchiv 
#
# Example usage: 
# To combine only files *01*ar in directory 2016.11.16-18:35:55.B0329+54
# python combine_profiles.py 20161116 2016.11.16-18:35:55.B0329+54 -sband 5 -eband 12 -subints 01 
#
# To manually dedisperse 1933 with twice its dm, do
# python combine_profiles.py 20161116 2016.11.16-12:25:13.B1933+16 -sband 5 -eband 11 -manual_dd 1 -dm 316.0 -F0 2.787565229026

import os
import time

import numpy as np
import glob
import argparse

import psrchive

def dedisperse_psrchive(fname):
	""" Dedisperse data using psrchive tools
	"""
	arch = psrchive.Archive_load(fname+'.ar')
	arch.dedisperse()
	arch.unload(fname+'dedispersed.ar')


def dedisperse_manually(fname, dm, p0):
	""" Dedisperse data manually, assuming 
	bw and central frequency.
	"""
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

def plot_spectra(data):
	import matplotlib.pylab as plt 

	fig = plt.figure()
	data -= np.median(data, axis=-1)[..., None]

	fig.add_subplot(211)
	plt.imshow(data, aspect='auto', interpolation='nearest', 
			extent=[0, p0, 0, 1])
	#plt.colorbar()

	fig.add_subplot(212)
	plt.plot(data.mean(0))

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
	parser.add_argument("-manual_dd", help="dedisperse manually", type=int, default=0)
	parser.add_argument("-dm", help="dm for manual dedispersion", type=float, default=0)
	parser.add_argument("-F0", help="pulsar rotation frequency in Hz", type=float, default=1)
	parser.add_argument("-o", help="name of output file name", default="all")
	args = parser.parse_args()

	# Unpack arguments
	date, folder = args.date, args.folder
	sband, eband, outname, subints = args.sband, args.eband, args.o, args.subints

	combine_subints(sband, eband, subints=subints, outfile='time_averaged'+folder)

	combine_freq(fnames='time_averaged'+folder, outfile=outname+folder+'.ar')

	if args.manual_dd == 1:
		p0 = args.F0**-1
		dm = args.dm

		# data = dedisperse_manually(outname+folder, dm, p0)
		# data = data.mean(0).mean(0)
		# data = data[:len(data)//4*4].reshape(len(data)//4, 4, -1).mean(1)
		# nph = data.shape[-1]
		# data = data.reshape(-1, nph/4, 4).mean(-1)
		# plot_spectra(data)

	else:
		dedisperse_psrchive(outname+folder)

	trickery=True

	if trickery:
		freq_ref = 1390.62 # MHz
		bw = 131.25 # MHz
		ftop = freq_ref + bw/2.
		fbot = freq_ref - bw/2.
		import matplotlib.pylab as plt
		fig = plt.figure()

		frq_rb = 2

		data_dm0 = dedisperse_manually(outname+folder, 0.0, p0)
		data = data_dm0.mean(0).mean(0)
		data = data[:len(data)//frq_rb*frq_rb].reshape(len(data)//frq_rb, frq_rb, -1).mean(1)
		nph = data.shape[-1]
		data = data.reshape(-1, nph/4, 4).mean(-1)
		data -= np.median(data, axis=-1)[..., None]
		prof_0dm = data.mean(0)

		ax1=fig.add_subplot(411)
		plt.imshow(data, aspect='auto', interpolation='nearest', 
			extent=[0, p0, ftop, fbot], cmap='Greys')
		plt.ylabel('freq [MHz]')

		data_dm = dedisperse_manually(outname+folder, dm, p0)
		data = data_dm.mean(0).mean(0)
		data = data[:len(data)//frq_rb*frq_rb].reshape(len(data)//frq_rb, frq_rb, -1).mean(1)
		nph = data.shape[-1]
		data = data.reshape(-1, nph/4, 4).mean(-1)
		data -= np.median(data, axis=-1)[..., None]
		prof_dm = data.mean(0)

		ax2=fig.add_subplot(412)
		plt.imshow(data, aspect='auto', interpolation='nearest', 
			extent=[0, p0, ftop, fbot], cmap='Greys')
		plt.ylabel('freq [MHz]')

		data_2dm = dedisperse_manually(outname+folder, 2*dm, p0)
		data = data_2dm.mean(0).mean(0)
		data = data[:len(data)//frq_rb*frq_rb].reshape(len(data)//frq_rb, frq_rb, -1).mean(1)
		nph = data.shape[-1]
		data = data.reshape(-1, nph/4, 4).mean(-1)
		data -= np.median(data, axis=-1)[..., None]
		prof_2dm = data.mean(0)

		ax3 = fig.add_subplot(413)
		plt.imshow(data, aspect='auto', interpolation='nearest', 
			extent=[0, p0, ftop, fbot], cmap='Greys')
		plt.ylabel('freq [MHz]')

		ax4 = fig.add_subplot(414)
		ph = np.linspace(0, p0, len(prof_2dm))
		plt.plot(ph, prof_0dm, color='steelblue')
		plt.plot(ph, prof_dm, color='grey')
		plt.plot(ph, prof_2dm, color='salmon')
		plt.legend(["DM=0", 
					"DM=expected", 
					"DM=2*expected"],
					loc=2,
					fontsize=12)
		plt.xlim(0, p0)
		plt.xlabel('pulse phase [s]')

		plt.text(0.1, .75, 'DM=0', fontsize=14, color='red')
		#plt.text(0.1, .75, 'DM=0', fontsize=14, color='')
		#plt.text(0.1, .75, 'DM=0', fontsize=14, color='')

		plt.show()


