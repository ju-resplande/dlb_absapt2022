<br />
<div align="center">
    <h1 align="center">ABSAPT 2022</h1>
    <img src="https://images.emojiterra.com/twitter/v14.0/512px/1f3c6.png" alt="https://emojiterra.com/trophy" width="200">
  
  <br />

  <br />
  
  [Deep Learning Brasil group](https://www.linkedin.com/company/inteligencia-artificial-deep-learning-brasil) submission at [ABSAPT 2022](https://sites.google.com/inf.ufpel.edu.br/absapt2022/).
</div>

## Task 1 - ATE

Submission is available on [DeepLearningBrasil_task1.csv](DeepLearningBrasil_task1.csv).

### How-to

1. Install requirements

```bash
pip install -r requirements.txt
```

2. Train ensemble  
Run Notebooks in order:
1 - [huggingface-roberta.ipynb](ATE/huggingface-roberta.ipynb)
2 - [huggingface-multilingual.ipynb](ATE/huggingface-multilingual.ipynb)
3 - [huggingface-futher-training.ipynb](ATE/huggingface-futher-training.ipynb)

3. Generate submission file
[eval.ipynb](ATE/eval.ipynb)

## Task 2 - SOE

Submission is available on [DeepLearningBrasil_task2.csv](DeepLearningBrasil_task2.csv).

### How-to

1. Install requirements

```bash
cd SOE
pip install -r requirements.txt
```

2. Train ensemble

```bash
bash train_ensemble.sh
```

4. Predict ensemble

```bash
bash predict_ensemble.sh
```

## Experimental setup

All experiments were made on V100 GPU (32GB).

## Acknowledgments

This work has been supported by the [AI Center of Excellence (Centro de Excelência em Inteligência Artificial – CEIA)](https://www.linkedin.com/company/inteligencia-artificial-deep-learning-brasil) of the Institute of Informatics at the Federal University of Goiás (INF-UFG).
