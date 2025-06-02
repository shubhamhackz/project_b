from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

def compute_advanced_metrics(eval_preds, label_list):
    predictions, labels = eval_preds
    import numpy as np
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_lab = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_pred.append(label_list[p])
                true_lab.append(label_list[l])
        true_predictions.append(true_pred)
        true_labels.append(true_lab)
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions)
    }
    report = classification_report(true_labels, true_predictions, output_dict=True)
    entity_metrics = {}
    for entity in ['PER', 'ORG', 'EMAIL', 'PHONE', 'LOC', 'MISC', 'ADDR']:
        if entity in report:
            entity_metrics[f"{entity.lower()}_precision"] = report[entity]["precision"]
            entity_metrics[f"{entity.lower()}_recall"] = report[entity]["recall"]
            entity_metrics[f"{entity.lower()}_f1"] = report[entity]["f1-score"]
            entity_metrics[f"{entity.lower()}_support"] = report[entity]["support"]
    results.update(entity_metrics)
    if "weighted avg" in report:
        results["weighted_f1"] = report["weighted avg"]["f1-score"]
        results["weighted_precision"] = report["weighted avg"]["precision"]
        results["weighted_recall"] = report["weighted avg"]["recall"]
    return results