import os
import time

import glob

N_FRB = 100
SNR_MIN = 7
backend = 'PRESTO'
AMBER_PATH = '~/test/amber_arg.sh'

outdir = '/data/03/Triggers/injection/%s' % time.strftime("%Y%m%d")
infile = '/data/03/Triggers/injection/sky_data_nofrb.fil'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

timestr = time.strftime("%Y%m%d-%H%M")
os.system('python inject_frb.py %s %s --nfrb %d --dm_list 25.0,50.0,75.0,125.0,150.0,200.0,250.0,300.0,325.0,350.0,375.0 --calc_snr True'% (infile, outdir, N_FRB))
exit()

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
