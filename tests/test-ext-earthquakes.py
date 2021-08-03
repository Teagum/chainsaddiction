from pathlib import Path
import sys
import numpy as np
from chainsaddiction import poishmm


PARAMS_PATH = 'params/earthquakes/'
DATA_PATH = 'data/earthquakes'


def main(argv=None):
    if argv is None:
        argv = sys.argv

    data = np.fromfile(DATA_PATH, sep='\n')

    for file in Path(PARAMS_PATH).glob('**/*.p'):
        print(file)
        prs = poishmm.read_params(str(file))
        hyper = (data.size, prs['m_states'], 300, 1e-6)
        params = prs['lambda'], prs['gamma'], prs['delta']
        res = poishmm.fit_em(*hyper, *params, data)
        print(res.err, res.n_iter, res.llk, res.aic, res.bic, "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
