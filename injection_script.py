import os

import glob

outdir = '/data/03/Triggers/'
infile = '/data/03/Triggers/CB_random.fil'
#infile = '/data2/output/20180402/2018-04-02-09:40:11.M31/filterbank/CB22.fil'
#outdir = '/data2/'
NFRB = 20

fn250 = 'dm250_%sfrbs' % NFRB
fn500 = 'dm500_%sfrbs' % NFRB
fn1000 = 'dm1000_%sfrbs' % NFRB
fn2500 = 'dm2500_%sfrbs' % NFRB

#os.system('python inject_frb.py %s %s/%s --nfrb %d --dm_high 250.0 --calc_snr True' % (infile, outdir, fn250, NFRB))
#os.system('python inject_frb.py %s %s/%s --nfrb %d --dm_high 500.0' % (infile, outdir, fn500, NFRB))
os.system('python inject_frb.py %s %s/%s --nfrb %d --dm_high 1000.0 --calc_snr True' % (infile, outdir, fn1000, NFRB))
#os.system('python inject_frb.py %s %s/%s --nfrb %d --dm_high 2500.0' % (infile, outdir, fn2500, NFRB))

try:
    outfile_250 = glob.glob('%s/%s*fil' % (outdir, fn250))[-1]
    outfile_250_dat = outfile_250.strip('.fil')
    os.system('prepdata -start 0 -dm 250.0 -o %s -ncpus 10 %s' % (outfile_250_dat, outfile_250))
    os.system('single_pulse_search.py %s.dat -t 8 -b' % outfile_250_dat)
    os.system('python triggers.py %s %s.singlepulse \
          --ntrig 500 --ndm 1 --save_data 0' 
          % (outfile_250, outfile_250_dat))
except:
    pass
try:
    outfile_500 = glob.glob('%s/dm500_100frbs* XX.fil' % outdir)[-1]
    outfile_500_dat = outfile_500.strip('.fil')
    os.system('prepdata -start 0 -dm 500.0 -o %s -ncpus 10 %s' % (outfile_500_dat, outfile_500))
    os.system('single_pulse_search.py %s.dat -t 8 -b' % outfile_500_dat)
except:
    pass
try:
    outfile_1000 = glob.glob('%s/%s*fil' % (outdir, fn1000))[-1]
    outfile_1000_dat = outfile_1000.strip('.fil')
    os.system('prepdata -start 0 -dm 1000.0 -o %s -ncpus 10 %s' % (outfile_1000_dat, outfile_1000))
    os.system('single_pulse_search.py %s.dat -t 8 -b' % outfile_1000_dat)
    os.system('python triggers.py %s %s.singlepulse \
          --ntrig 500 --ndm 1 --save_data 0' 
          % (outfile_1000, outfile_1000_dat))    
except:
    pass
try:
    outfile_2500 = glob.glob('%s/dm2500_100frbs*.fil' % outdir)[-1]
    outfile_2500_dat = outfile_2500.strip('.fil')
    os.system('prepdata -start 0 -dm 2500.0 -o %s -ncpus 10 %s' % (outfile_2500_dat, outfile_2500))
    os.system('single_pulse_search.py %s.dat -t 8 -b' % outfile_2500_dat)
except:
    pass

