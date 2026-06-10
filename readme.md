# Star Classification using SDSS Data

This was a project I worked on to practice building ML pipelines from scratch. The idea was simple — given photometric measurements of a celestial object from the Sloan Digital Sky Survey, can a model correctly identify whether it's a Galaxy, a Star, or a Quasar?

The dataset has 100,000 observations and is available on Kaggle : (https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).

---

## What I built

I trained two models — an SVM and a Random Forest — and compared them. Before getting to the models I spent time on the preprocessing, which turned out to be the more interesting part.

The `redshift` feature has a heavy right skew so I applied a log transform to it. The remaining numeric features go through median imputation and standard scaling. There are also some categorical columns that needed their own pipeline. I wired all of this together using sklearn's 'ColumnTransformer' so the whole thing runs cleanly as a single pipeline.

For the SVM I also did a binary classification experiment first (Galaxy vs not-Galaxy) to explore the precision-recall tradeoff. I plotted the PR curve, tested different thresholds, and wrapped the final model in a custom 'ThresholdedSVC' class to lock in the best threshold. I also compared it against a 'DummyClassifier' baseline just to make sure the model was actually learning something (~59% baseline vs ~93% SVM).

---

## Results

| Model         |  Accuracy | Galaxy F1 | QSO F1 | Star F1 |
| ---           |    ---    |    ---    |  ---   |  ---    |
| SVM           |    96%    |    0.97   |  0.94  |  0.96   |
| Random Forest |    98%    |    0.98   |  0.95  |  0.99   |

Random Forest wins across the board. QSO is the hardest class for both models, which makes sense — quasars can look spectrally similar to both stars and galaxies depending on the redshift.

---

## Features

| Feature                   | What it is |
|---                        |---|
| 'alpha', 'delta'          | Sky coordinates (right ascension and declination) |
| 'ultraviolet_filter'      | u-band magnitude |
| 'green_filter'            | g-band magnitude |
| 'red_filter'              | r-band magnitude |
| 'near_infrared_filter'    | i-band magnitude |
| 'infrared_filter'         | z-band magnitude |
| 'redshift'                | How much the light is shifted red — the most important feature |
| 'cam_col', 'plate', 'MJD' | Observation metadata |

I dropped the ID columns ('obj_ID', 'run_ID', 'rerun_ID', 'field_ID', 'spec_obj_ID', 'fiber_ID') since they carry no signal.

---

## How to run it

Download the dataset from Kaggle and update the file path in the first data loading cell. Then just run the notebook top to bottom.

'''
pip install numpy pandas matplotlib scikit-learn
'''



## About me

I'm Shoaib, a final year AI student at QUEST University in Nawabshah. I'm working on building out my ML portfolio — this is one of the projects I've done around classification and data pipelines.

LinkedIn : (www.linkedin.com/in/muhammad-shoaib-jamali-5ba972377)
