import re


def get_id_interval(filename):
    # print(filename)

    if '.png' in filename:
        filename = filename.replace('.png', '')

        ts_id = re.search('[0-9]+/(.+)_', filename, re.IGNORECASE).group(1)
        interval = re.search('[0-9]+_(.*)', filename, re.IGNORECASE).group(1)

    else:
        filename = filename.replace('.npy', '')

        ts_id = re.search(':(.*)_', filename, re.IGNORECASE).group(1)
        interval = re.search('_(.*)\\|', filename, re.IGNORECASE).group(1)

    interval = list(map(int, interval.split('-')))

    return ts_id, interval