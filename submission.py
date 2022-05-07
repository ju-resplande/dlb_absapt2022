import json
import os

import pandas as pd
from scipy.stats import mode

prediction_dir = "predictions"
test_data = "data/test_task2.csv"

ids = pd.read_csv(test_data, sep=";")["id"]

predictions = list()
for experiment_dir in os.listdir(prediction_dir):
    prediction_file = os.path.join(prediction_dir, experiment_dir, "predictions.json")

    with open(prediction_file) as f:
        prediction = json.load(f)

    predictions.append(prediction)

prediction_map = {"negativo": -1, "neutro": 0, "positivo": 1}
ensemble = pd.DataFrame(predictions).applymap(lambda cell: prediction_map[cell])
ensemble = pd.Series(mode(ensemble.values).mode[0])
submission = pd.concat([ids, ensemble], axis=1)
submission.to_csv("DeepLearningBrasil_task2.csv", sep=";", index=None, header=None)


print(ensemble.shape, submission.shape)
