from itertools import product

from dataset.generator import generate
from utils.functions import Paths
from utils.specification import specs

dataset = ['rational']
rho = ['0.100']
window_size = [5]
base_pattern = False
pattern_name = ''
mat_type = 'RP'

for ds, r,  w_s in product(dataset, rho, window_size):

    classes = specs[ds]['y_dim']
    ds_name = ds if not base_pattern else ds+'_base'
    max_stream_id = specs[ds_name]['max_stream_id']

    paths = Paths(ds, mat_type, r, w_s, base_pattern, pattern_name, core_path='..')
    beginning_path = paths.get_beginning_path()

    print(f'Creating dataset for: {ds} - {beginning_path} - {r} - {w_s} - {pattern_name}')

    generate(mat_type, beginning_path, r, w_s, classes,
             max_stream_id=max_stream_id,
             base_pattern=base_pattern,
             pattern_name=pattern_name,
             path_class=paths)
