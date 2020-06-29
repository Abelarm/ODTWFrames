"""
sts_generator is a Python adaptation of Usue Mori's Synthetic time series database
generator.

The original code, in R, was made by Usue Mori. The Documentantion and
the original code is available in the following Github repository:
    https://github.com/Usue/GenerationSyntheticDatabases
--------------------------------------------------------------------------------
Python adaptation by Izaskun Oregui. 2018-11-30
"""

import sys
import numpy as np
from dataset.generate_files.DataGenerator.generators import arma, synthetic_control, sines, kohlerlorenz, cbf, two_patterns, rational, seasonal, two_patterns2


class sts_generator:
    """
    This class allows to create 8 different synthetic database.
    The available databases are:
            1) arma              ----> 8 classes
            2) synthetic_control ----> 6 classes
            3) sines             ----> 5 classes
            4) kohlerlorenz      ----> 5 classes
            5) cbf               ----> 3 classes
            6) two_patterns      ----> 4 classes
            7) rational          ----> 4 classes
            8) seasonal          ----> 4 classes

    Parameters
    ----------
            * name: name of the database (must be one of the mentioned)
            * num_series: # of series in the database (positive integer).
            * length_series: length of series in the database (positive integer).
            * noise_level: percentage of noise introduced in database (positive real between 0 and 100).
            * outlier_level: proportion of outliers introduced in database  (positive integer between 0 and 100).
            * shift_level (sh in documentation): mean level of shift introduced in the database (positive integer between 0 and 100).
            * warp_level: percentage of warp introduce in the database (optional positive integer between 0 and 100. default None).
            * classpercentages: proportion of series in each clusters. (vector of length total classes, must sums 1)
    """

    def __init__(self, num_series, seed=123):

        if num_series < 1:
            sys.exit(
                'Initialization Error: number of time series must be largen than 0.')

        self.num_series = num_series
        self.names_list = ['two_patterns2', 'arma', 'synthetic_control', 'sines',
                           'kohlerlorenz', 'cbf', 'two_patterns', 'rational', 'seasonal']
        self.num_cluster = [2, 8, 6, 5, 5, 3, 4, 4, 4]
        self.seed = seed

    def generate(self, name, length_series, noise_level, outlier_level, shift_level, classpercentages, warp_level=None):
        """
        generate() creates the synthetic database specified in name.

        Parameters
        ----------
                * name: name of the database. Possible values:
                        arma, synthetic_control, sines, kohlerlorenz, CBF, two_patterns, rational and seasonal
                * length_series: length of series in the database (positive integer).
                * noise_level: percentage of noise introduced in database (positive real between 0 ans 100).
                * outlier_level: proportion of outliers introduced in database  (positive integer between 0 ans 100).
                * shift_level (sh in documentation): mean level of shift introduced in the database (positive integer between 0 ans 100).
                * warp_level: percentage of warp introduce in the database (positive integer between 0 ans 100).
                * classpercentages: proportion of series in each clusters. (vector of length total classes, must sums 1)
        """

        # Initialization check
        self._InitialCheck(name, length_series, noise_level,
                           outlier_level, shift_level, warp_level, classpercentages)

        # Create the list where the synthetic series will be saved.
        database = np.zeros((self.num_series, length_series))

        # Create a vector where the cluster number of each series will be saved.
        classes = np.zeros((self.num_series,))

        # The shift level is normalized based on the series length.
        shift_level = (shift_level * length_series) / 100.0

        # The maximum value to which the series may be shifted is calculated.
        max_shift = np.round(
            (3 * shift_level - 2 + np.sqrt(9 * shift_level**2 + 4)) / 4).astype(np.uint8)

        # The number of series in each cluster is calculated.
        classfrequencies = np.round(
            self.num_series * np.array(classpercentages)).astype(np.uint8)
        total_clusters = self.num_cluster[self.names_list.index(name)]
        if np.sum(classfrequencies) != self.num_series:
            self.num_series = np.sum(classfrequencies)
            print('Warning: num series has been modified. num_series = %d' %
                  (self.num_series))

        # The warping interval is defined.
        if name in ['two_patterns', 'two_patterns2', 'kohlerlorenz', 'cbf']:
            max_warp = (np.floor(warp_level * length_series / 100)
                        ).astype(np.uint8)

            if name == 'two_patterns':
                if (0.1 * length_series - max_warp) <= 0:
                    sys.exit(
                        'Initialization Error: in generate(...) warp level is too high for %s dataset' % name)
                if (np.floor(2.0 / 3.0 * length_series) + max_shift + 0.1 * length_series + max_warp) > length_series:
                    sys.exit(
                        'Initialization Error: in generate(...) shift and/or warp levels are to high for %s dataset' % name)

            elif (name == 'cbf') and (max_shift + max_warp + np.floor(length_series / 3.0) > length_series):
                sys.exit(
                    'Initialization Error: in generate(...) shift and/or warp levels are to large for cbf.')

            elif name == 'kohlerlorenz':
                if (np.round(length_series / 3.0) + 1.1 * (max_shift + max_warp)) > length_series:
                    sys.exit(
                        'Initialization Error: in generate(...) shift and warp levels are too large for kohlerlorenz database.')
                if (np.round(length_series / 2.0) + 1.1 * max_shift) > length_series:
                    sys.exit(
                        'Initialization Error: in generate(...) shift level is too large for kohlerlorenz database.')
                if (np.round(length_series / 4.0) - 1.1 * max_shift) < 1:
                    sys.exit(
                        'Initialization Error: in generate(...) shift level is to large for kohlerlorenz database.')

        # ----------------------------------------------------------------------------------------------
        # --------------------------- GENERATE TIME SERIES FOR EACH CLUSTER ----------------------------
        # ----------------------------------------------------------------------------------------------
        if name == 'arma':
            for cluster in range(total_clusters):
                if classfrequencies[cluster] > 0:
                    for i in range(classfrequencies[cluster]):
                        # A specific shift value is sampled from the defined interval.
                        np.random.seed(int(cluster + i) * self.seed)
                        shift = [np.random.randint(
                            low=0, high=2 * max_shift, size=1)[0] - max_shift if max_shift > 0 else 0][0]
                        # The series is generated and saved into the list.
                        if cluster == 0:
                            database[i, :] = arma(
                                length_series, max_shift, shift, cluster, seed=cluster * 1234)
                            classes[i] = cluster + 1
                        else:
                            database[int(np.sum(classfrequencies[:cluster]) + i), :] = arma(
                                length_series, max_shift, shift, cluster, seed=np.sum(
                                    classfrequencies[:cluster])
                            )
                            classes[int(
                                np.sum(classfrequencies[:cluster]) + i)] = cluster + 1

        elif name == 'synthetic_control':
            for cluster in range(total_clusters):
                if classfrequencies[cluster] != 0:
                    for i in range(classfrequencies[cluster]):
                        # A specific shift value is sampled from the defined interval.
                        np.random.seed(int(cluster + i) * self.seed)
                        shift = [np.random.randint(
                            low=0, high=2 * max_shift, size=1)[0] - max_shift if max_shift > 0 else 0][0]
                        if cluster == 0:
                            # The series is generated and saved into the database
                            database[i, :] = synthetic_control(
                                length_series, shift, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[i] = cluster + 1
                        else:
                            # The series is generated and saved into the database
                            database[int(np.sum(classfrequencies[:cluster])) + i,
                                     :] = synthetic_control(length_series, shift, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[int(
                                np.sum(classfrequencies[:cluster])) + i] = cluster + 1

        elif name == 'sines':
            for cluster in range(total_clusters):
                if classfrequencies[cluster] != 0:
                    for i in range(classfrequencies[cluster]):
                        # A specific shift value is sampled from the defined interval.
                        np.random.seed(int(cluster + i) * self.seed)
                        shift = [np.random.randint(
                            low=0, high=2 * max_shift, size=1)[0] - max_shift if max_shift > 0 else 0][0]
                        if cluster == 0:
                            # The series is generated and saved into the database
                            database[i, :] = sines(
                                length_series, shift, noise_level, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[i] = cluster + 1
                        else:
                            # The series is generated and saved into the database
                            database[int(np.sum(classfrequencies[:cluster])) + i, :] = sines(
                                length_series, shift, noise_level, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[int(
                                np.sum(classfrequencies[:cluster])) + i] = cluster + 1

        elif name == 'kohlerlorenz':
            for cluster in range(total_clusters):
                if classfrequencies[cluster] != 0:
                    for i in range(classfrequencies[cluster]):
                        # A specific shift value is sampled from the defined interval.
                        np.random.seed(int(cluster + i) * self.seed)
                        shift = [np.random.randint(
                            low=0, high=2 * max_shift, size=1)[0] - max_shift if max_shift > 0 else 0][0]
                        # A specific warp is sampled from the defined interval.
                        warp = np.random.randint(
                            low=0, high=2 * max_warp, size=1)[0] - max_warp
                        if cluster == 0:
                            # The series is generated and saved into the database
                            database[i, :] = kohlerlorenz(
                                length_series, max_shift, shift, warp, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[i] = cluster + 1
                        else:
                            # The series is generated and saved into the database
                            database[int(np.sum(classfrequencies[:cluster])) + i, :] = kohlerlorenz(
                                length_series, max_shift, shift, warp, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[int(
                                np.sum(classfrequencies[:cluster])) + i] = cluster + 1
        elif name == 'cbf':
            cluster_type = ['cylinder', 'bell', 'funnel']
            for cluster in range(total_clusters):
                if classfrequencies[cluster] != 0:
                    for i in range(classfrequencies[cluster]):
                        np.random.seed(int(cluster + i) * self.seed)
                        # A specific shift value is sampled from the defined interval.
                        shift = [np.random.randint(
                            low=0, high=2 * max_shift, size=1)[0] - max_shift if max_shift > 0 else 0][0]
                        # A specific warp is sampled from the defined interval.
                        warp = [np.random.randint(
                            low=0, high=2 * max_warp, size=1)[0] - max_warp if max_warp > 0 else 0][0]
                        if cluster == 0:
                            # The series is generated and saved into the database
                            database[i, :] = cbf(
                                length_series, shift, warp, cluster_type[cluster], seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[i] = cluster + 1
                        else:
                            # The series is generated and saved into the database
                            database[int(np.sum(classfrequencies[:cluster])) + i, :] = cbf(
                                length_series, shift, warp, cluster_type[cluster], seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[int(
                                np.sum(classfrequencies[:cluster])) + i] = cluster + 1

        elif name == 'two_patterns':
            for cluster in range(total_clusters):
                if classfrequencies[cluster] != 0:
                    for i in range(classfrequencies[cluster]):
                        np.random.seed(int(cluster + i) * self.seed)
                        # A specific shift value is sampled from the defined interval.
                        shift = [np.random.randint(
                            low=0, high=2 * max_shift, size=1)[0] - max_shift if max_shift > 0 else 0][0]
                        # A specific warp  value is sampled from the defined interval
                        warp = [np.random.randint(
                            low=0, high=2 * max_warp, size=1)[0] - max_warp if max_warp > 0 else 0][0]
                        if cluster == 0:
                            # The series is generated and saved into the database
                            database[i, :] = two_patterns(
                                length_series, shift, warp, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[i] = cluster + 1
                        else:
                            # The series is generated and saved into the database
                            database[int(np.sum(classfrequencies[:cluster])) + i, :] = two_patterns(
                                length_series, shift, warp, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[int(
                                np.sum(classfrequencies[:cluster])) + i] = cluster + 1

        elif name == 'two_patterns2':
            for cluster in range(total_clusters):
                if classfrequencies[cluster] != 0:
                    for i in range(classfrequencies[cluster]):
                        np.random.seed(int(cluster + i) * self.seed)
                        # A specific shift value is sampled from the defined interval.
                        shift = [np.random.randint(
                            low=0, high=2 * max_shift, size=1)[0] - max_shift if max_shift > 0 else 0][0]
                        # A specific warp  value is sampled from the defined interval
                        warp = [np.random.randint(
                            low=0, high=2 * max_warp, size=1)[0] - max_warp if max_warp > 0 else 0][0]
                        if cluster == 0:
                            # The series is generated and saved into the database
                            database[i, :] = two_patterns2(
                                length_series, shift, warp, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[i] = cluster + 1
                        else:
                            # The series is generated and saved into the database
                            database[int(np.sum(classfrequencies[:cluster])) + i, :] = two_patterns2(
                                length_series, shift, warp, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[int(
                                np.sum(classfrequencies[:cluster])) + i] = cluster + 1

        elif name == 'rational':
            for cluster in range(total_clusters):
                if classfrequencies[cluster] != 0:
                    for i in range(classfrequencies[cluster]):
                        np.random.seed(int(cluster + i) * self.seed)
                        # A specific shift value is sampled from the defined interval.
                        shift = [np.random.randint(
                            low=0, high=2 * max_shift, size=1)[0] - max_shift if max_shift > 0 else 0][0]
                        if cluster == 0:
                            # The series is generated and saved into the database
                            database[i, :] = rational(
                                length_series, shift, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[i] = cluster + 1
                        else:
                            # The series is generated and saved into the database
                            database[int(np.sum(classfrequencies[:cluster])) + i,
                                     :] = rational(length_series, shift, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[int(
                                np.sum(classfrequencies[:cluster])) + i] = cluster + 1

        elif name == 'seasonal':
            for cluster in range(total_clusters):
                if classfrequencies[cluster] != 0:
                    for i in range(classfrequencies[cluster]):
                        np.random.seed(int(cluster + i) * self.seed)
                        # A specific shift value is sampled from the defined interval.
                        shift = [np.random.randint(
                            low=0, high=2 * max_shift, size=1)[0] - max_shift if max_shift > 0 else 0][0]
                        if cluster == 0:
                            # The series is generated and saved into the database
                            database[i, :] = seasonal(
                                length_series, shift, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[i] = cluster + 1
                        else:
                            # The series is generated and saved into the database
                            database[int(np.sum(classfrequencies[:cluster])) + i,
                                     :] = seasonal(length_series, shift, cluster + 1, seed=self.seed * (cluster + 1) * (i + 1))
                            # The cluster number is saved
                            classes[int(
                                np.sum(classfrequencies[:cluster])) + i] = cluster + 1

        # ----------------------------------------------------------------------------------------------
        # ------------------------------- ADD NOISE TO EACH TIME SERIES --------------------------------
        # ----------------------------------------------------------------------------------------------
        if name != 'sines':
            if noise_level != 0:
                for i in range(self.num_series):
                    np.random.seed(int(1 + i) * self.seed)
                    noise = np.random.normal(loc=0, scale=(np.max(
                        database[i, :]) - np.min(database[i, :])) * noise_level / 100.0, size=length_series)
                    database[i, :] += noise

        # ----------------------------------------------------------------------------------------------
        # ----------------------------- ADD OUTLIERS TO EACH TIME SERIES -------------------------------
        # ----------------------------------------------------------------------------------------------
        if (outlier_level != 0):
            # The number of points that will become outliers are selected.
            num_points = np.round(
                float(length_series * outlier_level) / 100).astype(np.uint8)
            # For each series in the database:
            if num_points > 0:
                for i in range(self.num_series):
                    np.random.seed(int(num_points + i) * self.seed)
                    # The exact points that will be modified are selected.
                    positions = np.random.randint(
                        low=0, high=length_series - 1, size=num_points)
                    for j in positions:
                        a = np.random.randint(
                            low=0, high=self.num_series - 1, size=1)
                        b = np.random.randint(
                            low=0, high=length_series - 1, size=1)
                        # The value is interchanged.
                        database[i, j] = database[a, b]

        return database, classes

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def _InitialCheck(self, name, length_series, noise_level, outlier_level, shift_level, warp_level, classpercentages):
        """
        This function checks for possible errors in the introduced parameters.
        """

        if name not in self.names_list:
            sys.exit(
                'Initialization Error: In generate(name, ...), %s is not an acceptable value' % name)
        elif len(classpercentages) != self.num_cluster[self.names_list.index(name)]:
            sys.exit(
                'Initialization Error: In generate(name=%s), classpercentages must be a %d length vector, but it has %d elements'
                % (name, len(classpercentages), self.num_cluster[self.names_list.index(name)]))

        if (abs(np.sum(classpercentages) - 1) > 1e-10):
            sys.exit(
                'Initialization Error: In generate(...), classpercentages must sum 1')

        if length_series < 1:
            sys.exit(
                'Initialization Error: the length of the time series in generate(...) must be largen than 0.')

        if (noise_level < 0) or (noise_level > 100):
            sys.exit(
                'Initialization Error: in generate(...) the noise level must be a positive integer between 0 and 100.')

        if (outlier_level < 0) or (outlier_level > 100):
            sys.exit(
                'Initialization Error: in generate(...) the level of outliers must be a positive integer between 0 and 100.')

        if (shift_level < 0) or (shift_level > 100):
            sys.exit(
                'Initialization Error: in generate(...) the shift level must be a positive integer between 0 and 100.')

        if (warp_level is None) and (name in ['two_patterns', 'kohlerlorenz', 'cbf']):
            sys.exit(
                'Initialization Error: in generate(...) warp_level must be specified to generate %s dataset' % name)
        elif warp_level:
            if ((warp_level < 0) or (warp_level > 100)) and (name in ['two_patterns', 'kohlerlorenz', 'cbf']):
                sys.exit(
                    'Initialization Error: In generate(...) the warp level must be positive between 0 and 100')
