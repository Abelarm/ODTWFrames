from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from dataset.files import TimeSeries, RefPattern


class BasePattern:

    length = None
    min_value = None
    max_value = None
    values = {}

    def __init__(self, length, min_value, max_value, noise_val=5):
        self.length = length
        self.min_value = min_value
        self.max_value = max_value
        self.noise_val = noise_val
        self.noise = np.random.normal(loc=0, scale=(max_value - min_value) * noise_val / 100.0, size=length)
        self.pattern_names = ''

    # A
    def _mean(self):
        val = (self.min_value + self.max_value)/2
        noise = np.random.normal(loc=0, scale=self.noise_val/50, size=self.length)
        tmp_array = np.full(self.length, val)
        self.values['mean'] = tmp_array + noise
        self.pattern_names += 'A'

    # B
    def _increase(self):
        xp = [0, self.length]
        fp = [self.min_value, self.max_value]

        tmp_array = np.interp(range(0, self.length), xp, fp)
        self.values['increase'] = tmp_array + self.noise
        self.pattern_names += 'B'

    # C
    def _one_peak(self):
        xp = [0, self.length//2, self.length]
        fp = [self.min_value, self.max_value, self.min_value]

        tmp_array = np.interp(range(0, self.length), xp, fp)
        self.values['one_peak'] = tmp_array + self.noise
        self.pattern_names += 'C'

    def _peak_down(self):
        third = self.length//3
        xp = [0, third, third*2, self.length]
        fp = [self.min_value, self.max_value, self.min_value, self.max_value]

        tmp_array = np.interp(range(0, self.length), xp, fp)
        self.values['peak_down'] = tmp_array + self.noise

    def _two_peak(self):
        forth = self.length//4
        xp = [0, forth, forth*2, forth*3, self.length]
        fp = [self.min_value, self.max_value, self.min_value, self.max_value, self.min_value]

        tmp_array = np.interp(range(0, self.length), xp, fp)
        self.values['two_peak'] = tmp_array + self.noise

    def _one_step(self):
        mid = self.length//2
        tmp_array = np.empty(self.length)

        tmp_array[:mid] = self.min_value
        tmp_array[mid:] = self.max_value
        self.values['one_step'] = tmp_array + self.noise

    # D
    def _multiple_steps(self):
        third = self.length//3
        tmp_array = np.empty(self.length)

        tmp_array[:third] = self.min_value
        tmp_array[third:third*2] = (self.min_value + self.max_value)/2
        tmp_array[third*2:] = self.max_value
        self.values['multiple_steps'] = tmp_array + self.noise
        self.pattern_names += 'D'

    # E
    def _teeth(self):
        third = self.length // 3
        tmp_array = np.empty(self.length)
        tmp_array[:third] = self.min_value
        tmp_array[third:third * 2] = self.max_value
        tmp_array[third * 2:] = self.min_value
        self.values['teeth'] = tmp_array + self.noise
        self.pattern_names += 'E'

    def compute_pattern(self):

        # Never used
        # self._peak_down()

        self._mean()  # A
        self._increase()  # B
        # self._one_peak()  # C
        # self._two_peak()
        # self._one_step()
        # self._multiple_steps()   # D
        self._teeth()  # E

    def save(self, save_path):

        num_patter = len(self.values)

        to_save = np.empty((num_patter, self.length+1))
        for i, k in enumerate(self.values):

            v = self.values[k]
            plt.plot(v, label=k)
            to_save[i] = np.insert(v, 0, i)

        plt.legend()
        plt.show()

        if len(to_save) > 4:
            self.pattern_names = ''
        if len(self.pattern_names) > 0:
            filename = f'BASE_REF_len-{self.length}_noise-{self.noise_val}_num-1_{self.pattern_names}'
        else:
            filename = f'BASE_REF_len-{self.length}_noise-{self.noise_val}_num-1'

        path = join(save_path, filename)
        print(f'Saving: {path}')
        np.save(path, to_save)


dataset = 'cbf'
if dataset == 'gunpoint':
    ref_name = 'REF_num-5.npy'
    stream_name = 'STREAM_cycles-per-label-20_set-test_id-0.npy'
else:
    ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'
    stream_name = 'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-test_id-0.npy'


t = TimeSeries(
    f'../../data/{dataset}/{stream_name}')
timeseries = t.timeseries

ref = RefPattern(f'../../data/{dataset}/{ref_name}')
len_ref = ref.lab_patterns[0]['pattern'].shape[0]

base = BasePattern(len_ref, timeseries.min(), timeseries.max())
base.compute_pattern()

base.save(f'../../data/{dataset}')
