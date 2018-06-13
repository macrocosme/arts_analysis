import os

import numpy as np
import optparse

parser = optparse.OptionParser(prog="sp_search.py", \
                               version="", \
                               usage="%prog FN_FILTERBANK [OPTIONS]", \
                               description="Search for single pulses with presto, amber then compare")

parser.add_option('--fn_sp', dest='fn_true', type='float', \
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

options, args = parser.parse_args()
fn_fil = args[0]

dm_min, dm_max = 200, 300
fn_presto, fn_amber = options.outfile+'.singlepulse', options.outfile+'.trigger'
os.system('python tools.py %s %s %d %d' % (fn_presto, fn_amber, dm_min, dm_max)) # presto vs. amber

dm = options.dm
outfile = options.outfile
ncpu = options.ncpu

amber='/home/arts/test/amber.sh'
nbatch=50

os.system('prepdata -dm %d -o %s -ncpus %d -nobary %s' % (dm, outfile, ncpu, fn_fil)) 
os.system('single_pulse_search.py %s.dat -b -p' % outfile)

print("===============\nStarting AMBER\n===============")
os.system('%s %s %d %s' % (amber, fn_fil, nbatch, outfile))

fn_true = options.fn_true
fn_presto = outfile+'.singlepulse' 
fn_amber = outfile+'.trigger'

dm_min, dm_max = 0.9*dm, 1.1*dm

if fn_true is not None:
    os.system('python %s %s %d %d' % (fn_true, fn_presto, dm_min, dm_max)) # true vs. presto
    os.system('python %s %s %d %d' % (fn_true, fn_amber, dm_min, dm_max)) # true vs. amber

os.system('python ./tools.py %s %s %d %d' % (fn_presto, fn_amber, dm_min, dm_max)) # presto vs. amber
