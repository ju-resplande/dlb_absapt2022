import json
import os

import pandas as pd

prediction_dir = "predictions"
test_data = "data/test_task2.csv"

ids = pd.read_csv(test_data, sep=";")["id"]

predictions = list()
for experiment_dir in os.listdir(prediction_dir):
    prediction_file = os.path.append(prediction_dir, experiment_dir, "predictions.json")

    with open(prediction_file) as f:
        prediction = json.load(f)

    predictions.extend(prediction)

ensemble = pd.DataFrame(predictions).map({"negativo": -1, "neutro": 0, "positivo": 1})
ensemble = ensemble.mode()

submission = pd.concat([ids, ensemble], axis=0)
submission.to_csv("DeepLearningBrasil_task2.csv", sep=";", index=None, header=None)


print(ensemble.shape, submission.shape)
