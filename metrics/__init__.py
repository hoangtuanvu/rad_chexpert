"""Summary
"""
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def get_metrics(y_pred, y_true, beta, threshold=0.5, average_type='binary'):
    """Summary
    
    Args:
        y_pred (TYPE): Description
        y_true (TYPE): Description
        beta (TYPE): use for f_score, beta = 0.5 or 1 or 2
        threshold (float, optional): Description
        average_type (str, optional): Description
    
    Returns:
        TYPE: Description
    """

    y_pred[y_pred >= threshold] = 1.0
    y_pred[y_pred < threshold] = 0.0
    precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label=1, average=average_type)
    recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label=1, average=average_type)
    f_score = fbeta_score(y_true=y_true, y_pred=y_pred, beta=beta, pos_label=1,
                          average=average_type)

    return precision, recall, f_score