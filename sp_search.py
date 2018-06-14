import os

import numpy as np
import optparse

import tools 

SNRTools = tools.SNR_Tools()

parser = optparse.OptionParser(prog="sp_search.py", \
                               version="", \
                               usage="%prog FN_FILTERBANK [OPTIONS]", \
                               description="Search for single pulses with presto, amber then compare")

parser.add_option('--fn_sp', dest='fn_true', type='str', \
                  help="Text file with true single pulses"
                  "(Default: 8.0)", default=None)

parser.add_option('--dm', dest='dm', type='float', \
                  help="Search at this DM"
                  "(Default: 8.0)", default=250.0)

parser.add_option('--ncpu', dest='ncpu', type='int', \
                  help="Number of CPUs to use",\
                  default=10)

parser.add_option('--outfile', dest='outfile', type='str', \
                  help="file to write to",\
                  default='test')

parser.add_option('--figname', dest='figname', type='str', \
                  help="figure name",\
                  default=None)

options, args = parser.parse_args()
fn_fil = args[0]

dm = options.dm
outfile = options.outfile
ncpu = options.ncpu

amber='/home/arts/test/amber.sh'
nbatch=100

os.system('prepdata -dm %d -o %s -ncpus %d -nobary %s' % (dm, outfile, ncpu, fn_fil)) 
os.system('single_pulse_search.py %s.dat -b -p' % outfile)

print("===============\nStarting AMBER\n===============")
os.system('%s %s %d %s' % (amber, fn_fil, nbatch, outfile))

fn_true = options.fn_true
fn_presto = outfile+'.singlepulse' 
fn_amber = outfile+'.trigger'

dm_min, dm_max = 0.9*dm, 1.1*dm

if fn_true is not None:
    os.system('python ./tools.py %s %s %d %d %s' % (fn_true, fn_presto, dm_min, dm_max, 'true_presto.pdf')) # true vs. presto
    os.system('python ./tools.py %s %s %d %d %s' % (fn_true, fn_amber, dm_min, dm_max, 'true_amber.pdf')) # true vs. amber

os.system('python ./tools.py %s %s %d %d %s' % (fn_presto, fn_amber, dm_min, dm_max, 'presto_amber.pdf')) # presto vs. amber   

# if fn_true is not None:
#     par_true_presto, par_presto_true, par_match_true_presto, ind_missed_true_presto = SNRTools.compare_snr(
#                                         fn_true, fn_presto, dm_min=dm_min, 
#                                         dm_max=dm_max, save_data=False,
#                                         sig_thresh=5.0, t_window=0.13)

#     par_true_amber, par_amber_true, par_match_true_amber, ind_missed_true_amber = SNRTools.compare_snr(
#                                         fn_true, fn_amber, dm_min=dm_min, 
#                                         dm_max=dm_max, save_data=False,
#                                         sig_thresh=5.0, t_window=0.13)

# par_presto_amber, par_amber_true, par_match_presto_amber, ind_missed_presto_amber = SNRTools.compare_snr(
#                                         fn_presto, fn_amber, dm_min=dm_min, 
#                                         dm_max=dm_max, save_data=False,
#                                         sig_thresh=5.0, t_window=0.13)

# print('\nFound %d common trigger(s)' % par_match_arr.shape[1])

# snr_1 = par_match_arr[0, :, 0]
# snr_2 = par_match_arr[0, :, 1]

# print(snr_1)
# print(snr_2)

# print('File 1 has %f times higher S/N than file 2' % np.mean(snr_1/snr_2))

# if figname is not None:
#     import matplotlib.pyplot as plt
#     SNRTools.plot_comparison(par_1, par_2, par_match_arr, ind_missed, figname)













