# Financial Data Sentiment Analysis

## Project Overview
This project focuses on analyzing the sentiment of financial data using traditional machine learning algorithms and a fine-tuned BERT model. The aim is to preprocess the data, apply various machine learning models, and evaluate their performance in sentiment classification.

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Data Preprocessing](#2-data-preprocessing)
- [3. Machine Learning Algorithms](#3-machine-learning-algorithms)
- [4. Results of Machine Learning Models](#4-results-of-machine-learning-models)
- [5. Fine-tuning BERT](#5-fine-tuning-bert)
- [6. Results of BERT Model](#6-results-of-bert-model)
- [7. Conclusion](#7-conclusion)
- [8. Video Demonstration](#7-Video-Demonstration)
- [9. References](#8-references)

## 1. Introduction
The financial industry increasingly relies on sentiment analysis to inform investment decisions. This project aims to classify sentiments from financial text data using machine learning and natural language processing techniques.

## 2. Data Preprocessing
The dataset underwent a comprehensive preprocessing phase, which included:
- **Lowercasing**: All text was converted to lowercase for consistency.
- **Removing Special Characters**: Special characters were eliminated to focus on the text's core content.
- **Tokenization**: The text was split into individual words (tokens).
- **Stopwords Removal**: Common words that do not contribute to sentiment analysis were removed.
- **Lemmatization**: Words were reduced to their base or root form to enhance analysis.

The preprocessing resulted in a new column in the dataset containing cleaned text.

## 3. Machine Learning Algorithms
Four different machine learning algorithms were applied to the preprocessed data:
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Naive Bayes**
4. **Random Forest**

The data was split into training (80%) and testing (20%) sets. Each model was trained on the training set and evaluated on the test set.

## 4. Results of Machine Learning Models
The performance of the machine learning models was evaluated based on accuracy and other metrics:
- **Logistic Regression**: Accuracy of **76.52%**
- **SVM**: Accuracy of **76.57%**
- **Naive Bayes**: Accuracy of **72.43%**
- **Random Forest**: Accuracy of **75.64%**

Each model was analyzed for precision, recall, and F1-score, providing insights into their classification capabilities.

## 5. Fine-tuning BERT
Following the initial analysis with traditional machine learning models, a BERT model was fine-tuned for sentiment classification. This involved:
- **Tokenization**: The text data was tokenized using the BERT tokenizer.
- **Dataset Creation**: A custom dataset class was created to manage the training and testing data efficiently.
- **Model Training**: BERT was fine-tuned using the prepared datasets.

## 6. Results of BERT Model
The BERT model showed a significant improvement in accuracy compared to traditional models. The training and validation results over epochs were as follows:

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall  | F1      |
|-------|---------------|------------------|----------|-----------|---------|---------|
| 1     | 0.086000      | 0.683540         | 0.874942 | 0.874714  | 0.874942| 0.874413|
| 2     | 0.046400      | 0.781602         | 0.871223 | 0.871988  | 0.871223| 0.871286|
| 3     | 0.091200      | 0.759234         | 0.876337 | 0.876084  | 0.876337| 0.876183|
| 4     | 0.000300      | 0.855376         | 0.876337 | 0.876104  | 0.876337| 0.876123|
| 5     | 0.000500      | 0.892414         | 0.870293 | 0.870000  | 0.870293| 0.870115|

The evaluation metrics (accuracy, precision, recall, F1-score) demonstrate the effectiveness of the BERT model in sentiment classification.

## 7. Conclusion
This project successfully demonstrated the process of sentiment analysis in financial data through traditional machine learning and advanced deep learning techniques using BERT. The results indicated that BERT outperformed the traditional algorithms, showcasing its potential for complex text analysis tasks in the financial domain.

## 8. Video Demonstration


https://github.com/user-attachments/assets/f24ab8a5-0fc9-47f4-85b4-9a6cf6d56555



## 9. References
- Hugging Face Transformers Documentation
- Scikit-learn Documentation
- NLTK Documentation
