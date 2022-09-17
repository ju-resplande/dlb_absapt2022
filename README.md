<br />
<div align="center">
    <h1 align="center">ABSAPT 2022</h1>
    <img src="https://images.emojiterra.com/twitter/v14.0/512px/1f3c6.png" alt="https://emojiterra.com/trophy" width="200">
  
  <br />

  <br />
  
  [Deep Learning Brasil group](https://www.linkedin.com/company/inteligencia-artificial-deep-learning-brasil) submission at [ABSAPT 2022](https://sites.google.com/inf.ufpel.edu.br/absapt2022/).
</div>


Submission files are available on [DeepLearningBrasil_task1.csv](DeepLearningBrasil_task1.csv) and [DeepLearningBrasil_task2.csv](DeepLearningBrasil_task2.csv). Here is the [presentation](presentation.pdf).
## Installation

```bash
pip install -r requirements.txt
```
All experiments were made on V100 GPU (32GB).


## How-to
### Task 1 - ATE

1. Train ensemble. Run Notebooks in order:
   1. [huggingface-roberta.ipynb](ATE/huggingface-roberta.ipynb)
   2. [huggingface-multilingual.ipynb](ATE/huggingface-multilingual.ipynb)
   3. [huggingface-futher-training.ipynb](ATE/huggingface-futher-training.ipynb)

2. Generate submission file
[eval.ipynb](ATE/eval.ipynb)

### Task 2 - SOE

1. Train ensemble

```bash
bash SOE/train_ensemble.sh
```

2. Predict ensemble

```bash
bash SOE/predict_ensemble.sh
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/). See `LICENSE.txt` for more information.

## Acknowledgments

This work has been supported by the [AI Center of Excellence (Centro de Excelência em Inteligência Artificial – CEIA)](https://www.linkedin.com/company/inteligencia-artificial-deep-learning-brasil) of the Institute of Informatics at the Federal University of Goiás (INF-UFG).
