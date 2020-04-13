from keras_preprocessing.image import ImageDataGenerator

from dataset.dataset import Dataset

dataset_name = 'cbf'
beggining_path = f'../data/{dataset_name}/'
rho = '0.500'
window_size = 15


ds = Dataset(f'{beggining_path}REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy',
             f'{beggining_path}STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-train_id-*.npy',
             'train',
             f'{beggining_path}/rho {rho}',
             rho,
             window_size=window_size,
             classes=[1, 2, 3])

ds.create_image_dataset(f'{beggining_path}/rho {rho}/NN_DTW_dataset_{window_size}')
ds.create_series_dataset(f'{beggining_path}/rho {rho}/NN_TS_dataset_{window_size}/train')

ds = Dataset(f'{beggining_path}/REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy',
             f'{beggining_path}/STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-validation_id-*.npy',
             'validation',
             f'{beggining_path}/rho {rho}',
             rho,
             window_size=window_size,
             classes=[1, 2, 3])


ds.create_image_dataset(f'{beggining_path}/rho {rho}/NN_DTW_dataset_{window_size}')
ds.create_series_dataset(f'{beggining_path}/rho {rho}/NN_TS_dataset_{window_size}/validation')

ds = Dataset(f'{beggining_path}/REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy',
             f'{beggining_path}/STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-test_id-*.npy',
             'test',
             f'{beggining_path}/rho {rho}',
             rho,
             window_size=window_size,
             classes=[1, 2, 3])


ds.create_image_dataset(f'{beggining_path}/rho {rho}/NN_DTW_dataset_{window_size}', ref_ids=[8, 13, 25])
ds.create_series_dataset(f'{beggining_path}/rho {rho}/NN_TS_dataset_{window_size}/test')


datagen = ImageDataGenerator(rescale=1./255,
                             featurewise_center=True,
                             featurewise_std_normalization=True)


train_generator = datagen.flow_from_directory(
    directory=f'{beggining_path}/rho {rho}/NN_DTW_dataset_{window_size}/train',
    target_size=(100, window_size),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

x, y = train_generator.next()
print(x[0].shape)
print(y[0].shape)
#
#
# val_generator = datagen.flow_from_directory(
#     directory=r"../data/cbf/rho 0.100/NN_dataset/validation",
#     target_size=(100, 25),
#     color_mode="rgb",
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=True,
#     seed=42
# )
#
# test_generator = datagen.flow_from_directory(
#     directory=r"../data/cbf/rho 0.100/NN_dataset/test",
#     target_size=(100, 25),
#     color_mode="rgb",
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=True,
#     seed=42
# )


# datagen = DataGenerator('../data/cbf/rho 0.100/NN_TSDTW_dataset/train',
#                         dim=(3, 100, 25, 3),
#                         n_classes=3,
#                         to_fit=True,
#                         shuffle=True,
#                         batch_size=8)
# print(datagen.__len__())
#
# sample = datagen.__getitem__(datagen.__len__()-2)
#
# y_dim = sample[1][0].shape[0]
#
# shapes = sample[0][0].shape
#
# print(f'TotShape: {sample[0].shape}, timeseries shape: {shapes}, Y shape: {y_dim}')
#
# sample = datagen.__getitem__(datagen.__len__()-1)
#
# y_dim = sample[1][0].shape[0]
#
# shapes = sample[0][0].shape
#
# print(f'TotShape: {sample[0].shape}, timeseries shape: {shapes}, Y shape: {y_dim}')

# for i, (x, y) in enumerate(datagen):
#     print(i, x.shape)