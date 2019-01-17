from mne_extras import write_edf
from mne.datasets import testing
from mne.io import read_raw_edf
import mne
import numpy as np
import os.path as op

def _get_zero_crossing(data):
    assert data.ndim == 1, 'data has to be 1d not' % data.ndim
    return np.where(np.diff(np.sign(data)))[0]

data_path = testing.data_path(download=False)
fname = op.join(data_path, 'EDF', 'test_generator_2.edf')

raw = read_raw_edf(input_fname=fname, preload=True)
sine = raw.copy().pick_channels(['sine 1 Hz']).get_data()[0]

ch_names = ['stAtUs', 'tRigGer', 'sine 1 Hz']
ch_types = ['stim', 'stim', 'misc']
my_data = np.stack([np.zeros_like(sine),
                    np.zeros_like(sine),
                    sine])
my_data[:2,_get_zero_crossing(sine)] = 1

my_raw = mne.io.RawArray(data=my_data,
                         info=mne.create_info(ch_names=ch_names,
                                              sfreq=raw.info['sfreq'],
                                              ch_types=ch_types))
# my_raw.plot(scalings='auto')

write_edf(my_raw, '/tmp/test_stim_channel.edf', overwrite=True)
