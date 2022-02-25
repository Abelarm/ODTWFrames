import os.path

import numpy as np
from sklearn.metrics import classification_report

from networkAnalysis.summary import plot_roc_auc, plot_confusion_matrix

data_dir = 'iza/data/PowerCons'

with np.load(os.path.join(data_dir, 'DTWs_test.npz'), 'r') as data:
    labels = data['STS_labels']
    dtws = data['DTWs']

y_true = []
y_pred = []
for i in range(labels.shape[0]):

    y_true.extend(labels[i])
    tmp = []
    for j in range(dtws.shape[-1]):
        tmp.append(np.min(dtws[i, :, :, j], axis=0))
    y_pred.extend(np.argmin(np.array(tmp), axis=0))


y_pred = np.array(y_pred)
y_true = np.array(y_true)
print('Classification Report')
target_names = [str(i) for i in range(dtws.shape[-1])]
print(classification_report(y_true, y_pred))
save_path = f"{data_dir}/NN/net_results/"
os.makedirs(save_path, exist_ok=True)

plot_roc_auc(labels.shape[0],
             y_true, y_pred,
             save_path)
plot_confusion_matrix(y_true, y_pred, target_names, save_path)
