# spectrograph functions
import numpy as np


def load_order_bounds(spec='YJ',order=110):
    """
    spec: spectrograph name ("YJ" or "HK")
    order: order number
        if YJ: 110 - 149
        if HK: 59 - 97
    """
    path = '/Users/ashbake/Documents/Research/Projects/HISPEC/RV_simulations/rvsim/data/HISPEC/'
    if spec=='YJ':
        filename = 'orders_20220608C_HISPEC_SPECTRO_YJ_pyechelle.csv'
    elif spec=='HK':
        filename = 'orders_20220608C_HISPEC_SPECTRO_HK_pyechelle.csv'
    else:
        raise NameError('Fix spec name! HK or YJ')

    f = np.loadtxt(path + filename, delimiter=',')
    orders, wav_starts, wav_ends = f[:,0], f[:,1], f[:,2]
    iorder = np.where(orders==order)[0]
    
    return [wav_starts[iorder][0], wav_ends[iorder][0]]




if __name__=='__main__':
	pass




