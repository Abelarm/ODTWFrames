from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm

from models.dataGenerator import DataGenerator


def create_generator(root_dir, x_dim, y_dim, batch_size,
                     preprocessing=True, reload_images=False, base_pattern=False, always_custom=False):

    if len(x_dim) < 3 or x_dim[2] > 4 or x_dim[2] < 3 or base_pattern or always_custom:

        train_generator = DataGenerator(f'{root_dir}/train',
                                        dim=x_dim,
                                        n_classes=y_dim,
                                        to_fit=True,
                                        batch_size=batch_size,
                                        preprocessing=preprocessing)
        if preprocessing:
            scaler = train_generator.scaler
        else:
            scaler = None

        validation_generator = DataGenerator(f'{root_dir}/validation',
                                             dim=x_dim,
                                             n_classes=y_dim,
                                             to_fit=True,
                                             batch_size=batch_size,
                                             scaler=scaler,
                                             preprocessing=preprocessing)

        test_generator = DataGenerator(f'{root_dir}/test',
                                       dim=x_dim,
                                       n_classes=y_dim,
                                       to_fit=True,
                                       batch_size=batch_size,
                                       scaler=scaler,
                                       preprocessing=preprocessing)

        test_generator_analysis = DataGenerator(f'{root_dir}/test',
                                                dim=x_dim,
                                                n_classes=y_dim,
                                                to_fit=True,
                                                shuffle=False,
                                                batch_size=1,
                                                scaler=scaler,
                                                preprocessing=preprocessing)

    else:
        if preprocessing:
            print('Preprocessing the images')
            datagen = ImageDataGenerator(rescale=1. / 255,
                                         featurewise_center=True,
                                         featurewise_std_normalization=True)

            if not reload_images:
                try:
                    np.load(f'{root_dir}/all_images.npy')
                except IOError:
                    print(f'DIDN\'T FIND FILE {root_dir}/all_images.npy ==> reload all the images')
                    reload_images = True

            if reload_images:
                images = None
                print('Loading all the train images for fitting the data generator')

                train_generator = datagen.flow_from_directory(
                    directory=f'{root_dir}/train',
                    target_size=x_dim[:-1],
                    color_mode="rgb" if x_dim[2] == 3 else "rgba",
                    batch_size=128,
                    class_mode="categorical",
                    shuffle=False
                )

                len_generator = train_generator.__len__()
                for i in tqdm(range(len_generator)):
                    img, _ = train_generator.__getitem__(i)
                    if images is None:
                        images = img
                    images = np.vstack((images, img))
                # print(f'saving images in {root_dir}/all_images.np')
                # np.save(f'{root_dir}/all_images', images)
            else:
                print(f'loading all images to fit from {root_dir}/all_images.np')
                images = np.load(f'{root_dir}/all_images.npy')

            print(f'image shape {images.shape[1:]}')

            datagen.fit(images)

        else:
            print('NOT preprocessing the images - only scaling')
            datagen = ImageDataGenerator(rescale=1. / 255,
                                         featurewise_center=False,
                                         featurewise_std_normalization=False)

        train_generator = datagen.flow_from_directory(
                directory=f'{root_dir}/train',
                target_size=x_dim[:-1],
                color_mode="rgb" if x_dim[2] == 3 else "rgba",
                batch_size=batch_size,
                class_mode="categorical",
                seed=42,
                shuffle=True
            )

        validation_generator = datagen.flow_from_directory(
                directory=f'{root_dir}/validation',
                target_size=x_dim[:-1],
                color_mode="rgb" if x_dim[2] == 3 else "rgba",
                batch_size=batch_size,
                class_mode="categorical",
                seed=42,
                shuffle=True
        )

        test_generator = datagen.flow_from_directory(
            directory=f'{root_dir}/test',
            target_size=x_dim[:-1],
            color_mode="rgb" if x_dim[2] == 3 else "rgba",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True
        )

        test_generator_analysis = datagen.flow_from_directory(
            directory=f'{root_dir}/test',
            target_size=x_dim[:-1],
            color_mode="rgb" if x_dim[2] == 3 else "rgba",
            batch_size=1,
            class_mode="categorical",
            shuffle=False
        )

    return train_generator, validation_generator, test_generator, test_generator_analysis
