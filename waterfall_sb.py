import optparse
import numpy as np

import triggers_liam as triggers

if __name__=='__main__':
    parser = optparse.OptionParser(prog="waterfall_sb.py",
                                   version="",
                                   usage="%prog FN_FILTERBANK_PREFIX [OPTIONS]",
                                   description="Create diagnostic plots for individual triggers")

    parser.add_option('--ndm', dest='ndm', type='int',
                      help="Number of DMs to use in DM transform (Default: 50).",
                      default=1)

    parser.add_option('--mask', dest='maskfile', type='string',
                      help="Mask file produced by rfifind. (Default: No Mask).",
                      default=None)

    parser.add_option('--save_data', dest='save_data', action='store_true',
                      help="save trigger data (default: false)")

    parser.add_option('--rficlean', dest='rficlean', action='store_true',
                      help="use rficlean if True (default False)", default=False)

    parser.add_option('--threshold_time', dest='threshold_time', action='store_true',
                      help="If rficlean is True, defines threshold for time-domain clean (default 3.25)",
                      default=3.25)

    parser.add_option('--threshold_frequency', dest='threshold_frequency', type=float,
                      help="If rficlean is True, defines threshold for freqency-domain clean (default 2.5)",
                      default=2.75)

    parser.add_option('--bin_size', dest='bin_size', action='store_true',
                      help="If rficlean is True, defines bin size for bandpass removal (default 32)",
                      default=32)

    parser.add_option('--n_iter_time', dest='n_iter_time', action='store_true',
                      help="If rficlean is True, defines number of iteration for time-domain clean (default 3)",
                      default=3)

    parser.add_option('--n_iter_frequency', dest='n_iter_frequency', action='store_true',
                      help="If rficlean is True, defines number of iteration for frequency-domain clean (default 3)",
                      default=3)

    parser.add_option('--clean_type', dest='clean_type',
                      help="If rficlean is True, defines type of clean (default 'time')",
                      choices=['time', 'freqency', 'both'], default='time')

    parser.add_option('--subtract_zerodm', dest='subtract_zerodm', action='store_true',
                      help="use DM=0 timestream subtraction if True (default False)", default=False)

    parser.add_option('--nfreq_plot', dest='nfreq_plot', type='int',
                      help="make plot with this number of freq channels",
                      default=32)

    parser.add_option('--ntime_plot', dest='ntime_plot', type='int',
                      help="make plot with this number of time samples",
                      default=250)

    parser.add_option('--cmap', dest='cmap', type='str',
                      help="imshow colourmap",
                      default='RdBu')

    parser.add_option('--dm', dest='dm', type='float',
                      help="Dispersion measure of pulse (default 0)",
                      default=1.)

    parser.add_option('--t', dest='t', type='float',
                      help="Arrival time of pulse in seconds (default 10)",
                      default=10.0)

    parser.add_option('--downsamp', dest='downsamp', type='int',
                      help="Downsample data in time factor (default 1)",
                      default=1)

    parser.add_option('--outdir', dest='outdir', type='str',
                      help="directory to write data to",
                      default='./data/')

    parser.add_option('--beamno', dest='beamno', type='str',
                      help="Beam number of input data",
                      default='')

    parser.add_option('--tab', dest='tab', type=int,
                      help="TAB to process (0 for IAB) (default: 0)", default=0)

    parser.add_option('--sb', dest='sb', type=str, default='36',
                      help="Process synthesized beams")

    parser.add_option('--central_freq', dest='freq', type=int, default=1370, 
                      help="Central frequency in zapped channels filename (Default: 1370)")

    options, args = parser.parse_args()
    fn_fil = args[0]

    t0 = options.t
    downsamp = options.downsamp

    sb_generator = triggers.SBGenerator.from_science_case(science_case=4)
    sb_generator.reversed = True

    if options.sb in ['all', 'ALL', 'All']:
        sbs = range(70)
        triggers.mpl.use('Agg', warn=False)
        import matplotlib.pyplot as plt
    else:
        reload(triggers.mpl)
        triggers.mpl.use('TkAgg', warn=False)
        import matplotlib.pyplot as plt
        reload(plt)
        sbs = [int(options.sb)]
    
    for sb in sbs:
        print('Plotting SB %d' % sb)
        '''x = triggers.proc_trigger(fn_fil, options.dm, t0, -1,
                 ndm=options.ndm, mk_plot=True, downsamp=downsamp,
                 beamno=options.beamno, fn_mask=None, nfreq_plot=options.nfreq_plot,
                 ntime_plot=options.ntime_plot,
                 cmap='RdBu', cand_no=1, multiproc=False,
                 rficlean=False, snr_comparison=-1,
                 outdir=options.outdir, sig_thresh_local=0.0,
                 subtract_zerodm=False,
                 threshold_time=3.25, threshold_frequency=2.75, bin_size=32,
                 n_iter_time=3, n_iter_frequency=3, clean_type='time', freq=1370,
                 sb_generator=sb_generator, sb=sb)'''
        x = triggers.proc_trigger(fn_fil, options.dm, t0, -1,
                 ndm=options.ndm, mk_plot=True, downsamp=downsamp,
                 beamno=options.beamno, fn_mask=None, nfreq_plot=options.nfreq_plot,
                 ntime_plot=options.ntime_plot,
                 cmap='RdBu', cand_no=1, multiproc=False,
                 rficlean=False, snr_comparison=-1,
                 outdir=options.outdir, sig_thresh_local=0.0,
                 subtract_zerodm=False,
                 threshold_time=options.threshold_time, threshold_frequency=options.threshold_frequency,
                 bin_size=options.bin_size, n_iter_time=options.n_iter_time, n_iter_frequency=options.n_iter_frequency,
                 clean_type=options.clean_type, freq=1370, sb_generator=sb_generator, sb=sb)
        data = x[1]
        if options.save_data:
            np.save('./test_data', data)
