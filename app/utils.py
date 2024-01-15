import pickle
from pathlib import Path

def load_models(models_path, model_labels, models):
    models_path = Path(models_path)
    for model_file in models_path.iterdir():
        with open(model_file, 'rb') as f:
            model_labels.append(model_file.name.split('.')[0].replace('_', ' '))
            models.append(pickle.load(f))


def predict(model_labels, models, text):
    out_data_probs = []
    out_data_labels = set([])

    for model_ind, model in enumerate(models):
        pred = model.predict([text])
        pred_proba = model.predict_proba([text])
        model_label = model_labels[model_ind]
        if 'Other' > model_label:
            out_data_probs.append({model_label : str(pred_proba[0][0])})
        else:
            out_data_probs.append({model_label : str(pred_proba[0][1])})
        if pred != 'Other':
            out_data_labels.add(pred[0])
    return out_data_probs, list(out_data_labels)