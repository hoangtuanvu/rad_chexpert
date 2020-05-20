import os
import json
import click
from metrics import get_metrics
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_curve, auc


@click.command()
@click.option("--prediction", default='predictions.csv',
              help="Predictions from model which only contains binary values")
@click.option("--labels", default='labels.csv', help="Ground truth labels")
@click.option("--out", default='.', help="Output path")
@click.option("--beta", default=1, help="parameter for f_beta_score")
def evaluate(prediction, labels, out, beta):
    # create folder if not exists
    os.makedirs(out, exist_ok=True)
    # sort files by Images names
    pred_df = pd.read_csv(prediction)
    pred_df = pred_df.sort_values(by=['Images'])

    label_df = pd.read_csv(labels)
    label_df = label_df.sort_values(by=['Images'])
    assert len(pred_df) == len(label_df), "Mismatch between predictions and labels"
    header = list(pred_df.keys())[1:]

    ## calculate scores
    resp = dict()
    # disease-wise
    ovr_fscore = 0
    for t in range(len(header)):
        pred_t = np.array(pred_df[header[t]])
        disease = header[t]
        label_t = np.array(label_df[disease])
        precision, recall, f_score = get_metrics(pred_t, label_t, beta=beta, average_type='binary')

        fpr, tpr, thresholds = roc_curve(label_t, pred_t, pos_label=1)
        _auc = auc(fpr, tpr)

        resp['precision_{}'.format(disease)] = precision
        resp['recall_{}'.format(disease)] = recall
        resp['f_score_{}'.format(disease)] = f_score
        resp['auc_{}'.format(disease)] = _auc
        ovr_fscore += f_score

    print(ovr_fscore/len(header))
    # overall
    preds_array = pred_df[header].to_numpy()
    threshold = 0.5
    preds_array = np.vectorize(lambda x: 1 if x >= threshold else 0)(preds_array)
    targets_array = label_df[[c for c in header]].to_numpy()
    resp['overall_f_score'] = fbeta_score(y_pred=preds_array, y_true=targets_array, beta=beta,
                                          average='weighted')
    resp['overall_precision'] = precision_score(y_pred=preds_array, y_true=targets_array,
                                                average='weighted')
    resp['overall_recall'] = recall_score(y_pred=preds_array, y_true=targets_array,
                                          average='weighted')

    print(resp)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(resp, f)


if __name__ == '__main__':
    evaluate()
