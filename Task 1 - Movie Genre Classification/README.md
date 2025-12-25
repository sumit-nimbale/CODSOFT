# Movie Genre Classification (Task 1)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-orange)
![TF--IDF](https://img.shields.io/badge/Feature%20Extraction-TF--IDF-yellow)
![Naive Bayes](https://img.shields.io/badge/Algorithm-Multinomial%20Naive%20Bayes-red)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-lightgrey)

## Project Overview

![Application Interface](./assets/app_interface.png)

This project builds a **machine learning model** to predict the **genre of a movie** based solely on its **text description/plot**. The objective is to demonstrate how **Natural Language Processing (NLP)** techniques can convert raw text into numerical features and use them for supervised classification.

---

## Problem Statement

Given a movie plot as input text, automatically classify it into the most appropriate **movie genre**.

This is a **text classification** problem where the input is unstructured text and the output is a categorical label.

---

## Dataset

* The dataset consists of **movie plot summaries** and their corresponding **genres**.
* Each sample contains:

  * `text` → Movie description / plot
  * `label` → Movie genre

**Dataset Source:** ([https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb])

---

## Approach & Methodology

The project follows a standard NLP-based machine learning pipeline:

1. **Text Preprocessing**

   * Lowercasing text
   * Removing punctuation and special characters
   * Cleaning unnecessary symbols

2. **Feature Extraction**

   * **TF-IDF Vectorization** is used to convert text into numerical feature vectors.
   * This captures the importance of words relative to the dataset.

3. **Model Selection**

   * **Multinomial Naive Bayes** is used for classification.
   * Chosen due to its effectiveness and efficiency for text-based problems.

4. **Model Training & Evaluation**

   * Data is split into training and testing sets.
   * Model performance is evaluated using:

     * Accuracy
     * F1-score

---

## Algorithms Used

* **TF-IDF (Term Frequency–Inverse Document Frequency)** – Feature extraction
* **Multinomial Naive Bayes** – Classification algorithm

---

## Project Structure

```
Task 1 - Movie Genre Classification/
│
├── data/
│   ├── raw/          # Original dataset
│   └── processed/    # Cleaned and transformed data
│
├── model/
│   └── genre_model.pkl   # Trained model
│
├── notebooks/        # Exploration and training notebooks
├── app.py            # Model inference interface
├── requirements.txt  # Dependencies
└── README.md         # Project documentation
```

---

## Output

* The trained model predicts the **most probable movie genre** for a given input description.
* The output is a single genre label based on learned patterns from the dataset.

---

## Key Learnings

* Practical implementation of **text preprocessing**
* Understanding and applying **TF-IDF** for NLP tasks
* Using **Naive Bayes** for multi-class text classification
* Structuring an end-to-end machine learning project

---

## Use Case

This model can be used for:

* Automatic tagging of movies by genre
* Content recommendation systems
* NLP-based classification demonstrations

---

## Status

✅ Task 1 completed successfully
