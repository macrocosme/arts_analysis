import os
import time

import glob

N_FRB = 100
SNR_MIN = 7
backend = 'PRESTO'
AMBER_PATH = '~/test/amber_arg.sh'

outdir = '/data/03/Triggers/injection/%s' % time.strftime("%Y%m%d")
infile = '/data/03/Triggers/injection/sky_data_nofrb.fil'
#infile = '/data2/output/20180402/2018-04-02-09:40:11.M31/filterbank/CB22.fil'
#infile = '/data2/output/20180328/2018-03-28-13:01:20.J0248+6021/filterbank/CB21.fil'
#infile = '/data1/output/20180425/2018-04-25-03\:02\:05.RA20DEC57/filterbank/CB21.fil'
#infile = '/data2/output/snr_tests_liam/CB21.fil'
#outdir = '/data2/snr_tests_liam'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

timestr = time.strftime("%Y%m%d-%H%M")
os.system('python inject_frb.py %s %s --nfrb %d --dm_list 300.0,400.0,600.0,800.0 --calc_snr True'% (infile, outdir, N_FRB))

# note this assumes tstr is the same in both inject_frb and glob
fil_list = glob.glob('%s/*%s.fil' % (outdir, timestr))

for fn_fil in fil_list:
    DM = float(fn_fil.split('dm')[1].split('_')[0])
    fn_base = fn_fil.strip('.fil')

    if backend is 'AMBER':
        os.system('%s %s' % (AMBER_PATH, fn))
        fn_trigger = '%s.trigger' % fn_base
    elif backend is 'PRESTO':
        os.system('prepdata -start 0 -dm %d -o %s -ncpus 5 %s' % (DM, fn_base, fn_fil))        
        os.system('single_pulse_search.py %s.dat -t %d -b' % (fn_base, SNR_MIN))
        fn_trigger = '%s.singlepulse' % fn_base
        os.system('python tools.py %s.txt %s.singlepulse' % (fn_base, fn_base))

    else:
        print("Incorrect backend. Must be either PRESTO or AMBER")
        pass

#    os.system('python triggers.py %s %s --ntrig 500 \
#               --ndm 1 --save_data 0 --ntime_plot 250 \
#               --sig_thresh 8.' % (fn_fil, fn_trigger))
exit()
