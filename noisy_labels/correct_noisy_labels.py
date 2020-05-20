import os
import click
import copy
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import cleanlab
from cleanlab.util import onehot2int


@click.command()
@click.option("--prediction", default='predictions.csv',
              help="Predictions from model which only contains binary values")
@click.option("--labels", default='labels.csv', help="Ground truth labels")
@click.option("--out", default='.', help="Output path")
@click.option("--multi-label", default=False)
def find_noisy_labels(prediction, labels, out, multi_label):
    # create folder if not exists
    os.makedirs(out, exist_ok=True)
    # sort files by Images names
    pred_df = pd.read_csv(prediction)
    pred_df = pred_df.sort_values(by=['Images'])

    label_df = pd.read_csv(labels)
    label_df = label_df.sort_values(by=['Images'])
    label_df[list(label_df)[1:]] = 1 * (label_df[list(label_df)[1:]] >= 0.5)

    # print(label_df)
    assert len(pred_df) == len(label_df), "Mismatch between predictions and labels"

    psx = np.array(pred_df[list(pred_df)[1:]])
    _labels = np.array(label_df[list(label_df)[1:]])

    if multi_label:
        correctly_formatted_labels = onehot2int(_labels)
        label_errors_bool = cleanlab.pruning.get_noise_indices(s=correctly_formatted_labels,
                                                               psx=psx,
                                                               prune_method='prune_by_noise_rate',
                                                               sorted_index_method=None,
                                                               multi_label=True)
        label_errors_idx = cleanlab.pruning.order_label_errors(label_errors_bool=label_errors_bool,
                                                               psx=psx,
                                                               labels=correctly_formatted_labels,
                                                               sorted_index_method='normalized_margin')
        nb_dict = {'Images': list(), 'Labels': list(), 'Prediction': list(), 'Indices': list()}
        for i in label_errors_idx:
            nb_dict['Images'].append(label_df.iloc[i]['Images'])
            nb_dict['Labels'].append(','.join(list(map(str, _labels[i, :]))))
            nb_dict['Prediction'].append(','.join(list(map(str, psx[i, :]))))
            nb_dict['Indices'].append(i)

        nb_df = pd.DataFrame(nb_dict)
        nb_df.to_csv('noise_labels_multi.csv', index=False)
    else:
        header = list(pred_df.keys())[1:]

        for t in range(len(header)):
            pred_t = np.array(pred_df[header[t]])
            disease = header[t]
            # if disease not in list(label_df.keys()):
            #     raise ValueError("{} does not exists in ground truth labels!")
            label_t = np.array(label_df[disease])

            # generate noise labels
            binary = list()
            for i in range(len(pred_t)):
                binary.append([1 - pred_t[i], pred_t[i]])
            binary = np.array(binary)

            label_errors_bool = cleanlab.pruning.get_noise_indices(s=copy.deepcopy(label_t),
                                                                   psx=binary,
                                                                   prune_method='prune_by_noise_rate',
                                                                   sorted_index_method=None)

            label_errors_idx = cleanlab.pruning.order_label_errors(
                label_errors_bool=label_errors_bool, psx=binary, labels=copy.deepcopy(label_t),
                sorted_index_method='normalized_margin')

            nb_dict = {'Images': list(), header[t]: list(), 'Prob': list(), 'Indices': list()}
            for i in label_errors_idx:
                nb_dict['Images'].append(label_df.iloc[i]['Images'])
                nb_dict[header[t]].append(label_t[i])
                nb_dict['Prob'].append(float(pred_t[i]))
                nb_dict['Indices'].append(i)

            nb_df = pd.DataFrame(nb_dict)
            nb_df.to_csv(os.path.join(out, '{}.csv'.format(header[t])), index=False)


if __name__ == '__main__':
    find_noisy_labels()
