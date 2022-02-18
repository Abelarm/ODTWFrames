import subprocess

for name in ['Chinatown', 'Coffee', 'Fungi', 'Ham', 'Plane', 'PowerCons', 'Gunpoint']:
    for mode in ['dtw', 'sts']:
        for arch in ['resnet', 'cnn', 'rnn']:
            for window in [5, 15, 25]:

                if arch == 'rnn' and mode == 'dtw':
                    print('RNN not present for DTW')
                    continue
                if mode == 'dtw' and window != 5:
                    print('skipping dtw  ws > 5')
                    continue
                subprocess.call(f"python v2/train.py --dataset_name {name} --mode {mode} --architecture {arch} --window_size {window} --num_workers 15", shell=True)
