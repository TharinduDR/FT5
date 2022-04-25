from sklearn.metrics import recall_score, precision_score, f1_score


def sentence_label_evaluation(predictions, real_values):
    labels = set(real_values)

    for label in labels:
        print()
        print("Stat of the {} Class".format(label))
        print("Recall {}".format(recall_score(real_values, predictions, labels=labels, pos_label=label)))
        print("Precision {}".format(precision_score(real_values, predictions, labels=labels, pos_label=label)))
        print("F1 Score {}".format(f1_score(real_values, predictions, labels=labels, pos_label=label)))

    print()
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighter F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))

    return f1_score(real_values, predictions, average='macro'), f1_score(real_values, predictions, average='weighted')
