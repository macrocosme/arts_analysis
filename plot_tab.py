import sys

import tools

fn = sys.argv[1]

try:
    suptitle = sys.argv[2]
except:
    suptitle = ''

tools.plot_tab_summary(fn, suptitle=suptitle)
