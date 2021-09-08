from pathlib import Path
import sys
import numpy as np
from scipy import stats
from chainsaddiction import poishmm


PARAMS_PATH = 'params/earthquakes/3/quantile.p'
DATA_PATH = 'data/earthquakes'


def main(argv=None):
    if argv is None:
        argv = sys.argv

    data = np.fromfile(DATA_PATH, sep='\n')
    prs = poishmm.read_params(PARAMS_PATH)
    lsdp = stats.poisson.logpmf(data[:, None], prs['lambda'])
    params = prs['lambda'], prs['gamma'], prs['delta']
    hyper = (data.size, prs['m_states'], 300, 1e-6)
    hmm = poishmm.fit(*hyper, *params, data)

    decoding = poishmm.local_decoding(data.size, prs['m_states'], lsdp)
    print(decoding)
    return 0


if __name__ == '__main__':
    sys.exit(main())
