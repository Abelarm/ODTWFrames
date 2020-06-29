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

        self.methods_dic = {
            'A': self._mean,
            'B': self._increase,
            'C': self._one_peak,
            'D': self._multiple_steps,
            'E': self._teeth
        }

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

    def compute_pattern(self, pattern_name):

        for pattern in pattern_name:

            print(f'Creating base_pattern: {pattern}')
            self.methods_dic[pattern]()

    def plot_patterns(self):

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(1, len(self.values))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, k in enumerate(self.values):
            v = self.values[k]

            f_axi1 = fig.add_subplot(gs[0, i])
            f_axi1.plot(v, color=colors[i])
            # f_axi1.axis('equal', ymin=min(v), ymax=max(v))
            # f_axi1.axes.get_yaxis().set_visible(False)
            # f_axi1.axes.get_xaxis().set_visible(False)
            f_axi1.set_title(f'Universal Pattern: {self.pattern_names[i]}')
            if i == 0:
                f_axi1.axis(ymin=self.min_value, ymax=self.max_value)

        # plt.show()

    def save(self, save_path):

        num_patter = len(self.values)

        to_save = np.empty((num_patter, self.length+1))
        for i, k in enumerate(self.values):

            v = self.values[k]
            plt.plot(v, label=k)
            to_save[i] = np.insert(v, 0, i)

        if len(to_save) > 5:
            self.pattern_names = ''
        if self.pattern_names == 'ABCDE':
            self.pattern_names = 'FULL'
        if len(self.pattern_names) > 0:
            filename = f'BASE_REF_len-{self.length}_noise-{self.noise_val}_num-1_{self.pattern_names}'
        else:
            filename = f'BASE_REF_len-{self.length}_noise-{self.noise_val}_num-1'

        path = join(save_path, filename)
        print(f'Saving: {path}')
        np.save(path, to_save)