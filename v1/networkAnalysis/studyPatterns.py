from itertools import product
from os.path import join

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm


def pattern_study(model, generator, classes, save_dir):

    max_number_image = 1000

    df = pd.DataFrame()
    for class_num in classes:
        selected_samples = []
        i = 0
        for x, y in generator:

            if y.argmax() == class_num:
                selected_samples.append((x, y))
                i += 1

            if i == max_number_image:
                break

        accuracy_dict = {}
        channels = range(x.shape[-1])
        for sample, chan in tqdm(product(selected_samples, channels)):

            x = sample[0]
            y = sample[1]

            x[0, :, :, chan] = x[0, :, :, chan] * 0

            y_pred = model.predict(x=x,batch_size=1)

            acc = 1 - cosine(y, y_pred)

            if chan in accuracy_dict:
                accuracy_dict[chan].append(acc)
            else:
                accuracy_dict[chan] = [acc]

        accuracy_dict = dict((k, np.mean(v)) for k, v in accuracy_dict.items())
        df = df.append({'Class': int(class_num), **accuracy_dict},  ignore_index=True)

    df = df.sort_values(by=['Class'])
    df = df.set_index('Class')
    print(f'Accuracy result for pattern study')
    print(df)
    with open(join(save_dir, 'pattern_study.txt'), 'w') as f:
        f.write(f'Accuracy result for pattern study\n')
        f.write(df.to_string())
