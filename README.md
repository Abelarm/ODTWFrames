# DTW v2

### Install requirements
Run `pip install -r requirements.txt`

#### Pull submodule
For pulling submodules `git submodule update --init --recursive`

### Download the dataset

Select a dataset TS from the [UCR_UEA database](https://www.timeseriesclassification.com/index.php)

`python v2/download_data.py --dataset_name <dataset_name>`

If you want to calculate more medoids for each class you can run the command adding those parameters:
`--distance_type cluster  --num_cluster <x>`

it will calculate `x` medoids for each class 


### Create the DTWs

Create the DTWs from an already downloaded dataset as:

```
python v2/create_DTWs.py \
--dataset_name <dataset_name> \
--num_reference <x> \
--tot_sts <y> \
--rho <r> \
```

default values for `num_reference` is `60` for `tot_sts` is `30` and `rho` is `0.1`

### Visualize the dataset

Visualize an already created dataset

`python v2/visualize_dataset.py --dataset_name <dataset_name>`

use `c` for cotinuing visualizing another example and `b` for stopping

### Train the network

For training the network use:

`python v2/train.py --dataset_name <dataset_name> --mode ['dtw', 'sts'] --architecture ['resnet', 'cnn', 'rnn']`

for `DTW` mode only the `cnn` and `resnet` are available.

all the results and checkpoints will be created under the folder `data/<dataset_name>`

you can modify the following parameters:
- `window_size` default value `5`
- `batch_size` default value `128`
- `lr` default value `5e-03`
- `max_epochs` default value `15`
- `num_workers` default value `6`
- `PATH` default value `None` if a path to a checkpoint is provided the network won't be trained and only evaluated


