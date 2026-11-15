# Multiclass Sentiment Analysis Using Deep Learning

## 1. Project Overview

This project focuses on performing multiclass sentiment analysis using four deep learning architectures: Artificial Neural Network (ANN), Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).

The objective is to classify textual reviews into three sentiment categories:

- **Negative (0)**
- **Neutral (1)**
- **Positive (2)**

The models are trained and evaluated using a sentiment analysis dataset. Hyperparameter tuning is performed using KerasTuner Random Search to identify the best-performing configuration for each architecture.

## 2. Assignment Statement

**Task:**

Perform sentiment analysis on a given dataset by building and comparing Neural Network, RNN, LSTM, and GRU models.

Automatically tune their hyperparameters by testing different values within predefined ranges. Finally, identify the best-performing model and its corresponding hyperparameter configuration.

**Project Objectives:**

- Understand and preprocess textual data for sentiment classification.
- Implement ANN, RNN, LSTM, and GRU architectures.
- Train and evaluate all four models.
- Perform automatic hyperparameter tuning using KerasTuner.
- Compare model performance using evaluation metrics.
- Identify and save the best-performing model.

## 3. Dataset Description

**Dataset:** Multiclass Sentiment Analysis Dataset

**Source:** [Hugging Face Dataset – Sp1786](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset)

The dataset contains textual reviews with three sentiment labels.

| Sentiment | Label |
|---|---:|
| Negative | 0 |
| Neutral | 1 |
| Positive | 2 |

### Dataset Distribution

| Dataset Split | Number of Records |
|---|---:|
| Training | 31,231 |
| Validation | 5,204 |
| Testing | 5,205 |

One missing text record was removed from the original test split. Empty text records were also excluded where applicable.

## 4. Technologies and Libraries Used

- Python
- TensorFlow / Keras
- KerasTuner
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 5. Methodology

### Step 1: Data Preprocessing

Text preprocessing was performed using TensorFlow's TextVectorization layer.

The preprocessing steps included:

1. Text standardization.
2. Tokenization.
3. Vocabulary creation.
4. Integer encoding.
5. Padding and truncation.

The following parameters were selected:

- Maximum vocabulary size: 20,000
- Maximum sequence length: 96
- Embedding dimension: Tuned automatically

The vocabulary was adapted using training data only to prevent data leakage.

### Step 2: Deep Learning Model Implementation

Four architectures were implemented:

**1. Artificial Neural Network (ANN)**

Uses an embedding layer, Global Average Pooling, and fully connected dense layers to classify sentiment.

**2. Recurrent Neural Network (RNN)**

Uses SimpleRNN to process sequential text information and learn relationships between words.

**3. Long Short-Term Memory (LSTM)**

Uses memory cells and gating mechanisms to capture longer-term dependencies in text.

**4. Gated Recurrent Unit (GRU)**

Uses update and reset gates to learn sequential information with a comparatively simpler recurrent structure.

### Step 3: Automatic Hyperparameter Tuning

KerasTuner Random Search was used to identify suitable hyperparameters for all four architectures.

**Search space:**

| Hyperparameter | Values Tested |
|---|---|
| Embedding Dimension | 32, 64, 100 |
| Recurrent Units | 32, 64, 128 |
| Dense Layer 1 | 32, 64, 128 |
| Dense Layer 2 | 16, 32, 64 |
| Dropout Rate | 0.2, 0.3, 0.4, 0.5 |
| Learning Rate | 0.001, 0.0005, 0.0001 |
| Batch Size | 32, 64, 128 |

- Tuning method: Random Search
- Trials per architecture: 10
- Maximum epochs per trial: 15
- Early stopping patience: 3
- Optimization objective: Validation accuracy

The best hyperparameter configuration was selected separately for each architecture based on validation accuracy.

## 6. Best Hyperparameters

| Hyperparameter | ANN | RNN | LSTM | GRU |
|---|---:|---:|---:|---:|
| Embedding Dimension | 32 | 32 | 64 | 64 |
| Recurrent Units | N/A | 32 | 128 | 128 |
| Dense Layer 1 | 32 | 128 | 32 | 32 |
| Dense Layer 2 | 16 | 64 | 32 | 32 |
| Dropout Rate | 0.5 | 0.5 | 0.4 | 0.4 |
| Learning Rate | 0.0001 | 0.0001 | 0.001 | 0.001 |
| Batch Size | 64 | 32 | 32 | 32 |

## 7. Model Performance Comparison

The four tuned models were evaluated using the test dataset.

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| ANN | 67.32% | 0.7630 |
| RNN | 64.98% | 0.8536 |
| LSTM | 66.63% | 0.9277 |
| **GRU** | **68.57%** | **0.7434** |

### Validation Accuracy Comparison

| Model | Validation Accuracy |
|---|---:|
| ANN | 66.99% |
| RNN | 65.78% |
| LSTM | 67.58% |
| **GRU** | **69.02%** |

### Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Confusion matrices and classification reports were generated to understand class-wise performance and misclassification patterns.

## 8. Final Best-Performing Model – GRU

Based on the experimental results, the **Gated Recurrent Unit (GRU)** was identified as the best-performing architecture.

### Best GRU Configuration

| Parameter | Selected Value |
|---|---:|
| Embedding Dimension | 64 |
| GRU Units | 128 |
| Dense Layer 1 | 32 |
| Dense Layer 2 | 32 |
| Dropout Rate | 0.4 |
| Learning Rate | 0.001 |
| Batch Size | 32 |

### Final GRU Performance

- Validation Accuracy: **69.02%**
- Test Accuracy: **68.57%**
- Test Loss: **0.7434**
- Macro F1-score: **0.69**

### GRU Classification Report

| Sentiment | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.71 | 0.66 | 0.69 |
| Neutral | 0.63 | 0.60 | 0.61 |
| Positive | 0.72 | 0.81 | 0.76 |

The GRU model achieved the highest test accuracy and lowest test loss among the four tuned architectures in this experiment.

## 9. Model Saving and Prediction

The best-performing GRU model was saved in Keras format.

**Saved files:**

- `best_gru_sentiment_model.keras` – trained GRU model.
- `best_gru_hyperparameters.json` – selected hyperparameter configuration.

The saved model was successfully loaded and tested using new sample reviews.

**Sample Predictions:**

| Input Review | Predicted Sentiment | Confidence |
|---|---|---:|
| This product is absolutely amazing and I love it | Positive | 99.02% |
| The experience was terrible and very disappointing | Negative | 99.73% |
| The product is okay, nothing special | Neutral | 48.39% |

These predictions demonstrate that the trained model can classify new textual inputs into the three sentiment categories.

*Note: Confidence represents the model's predicted probability and is not a guarantee of prediction correctness.*

## 10. Conclusion

In this project, four deep learning architectures were implemented and compared for multiclass sentiment analysis.

Automatic hyperparameter tuning was performed using KerasTuner Random Search to identify suitable configurations for each model.

Among the four architectures, GRU achieved the best experimental performance, with a test accuracy of 68.57% and a test loss of 0.7434.

Therefore, GRU was selected as the final best-performing model for this sentiment classification task.

The project provided practical experience in text preprocessing, deep learning model development, hyperparameter optimization, model evaluation, and prediction using a saved model.

## 11. Future Improvements

- Increase the number of hyperparameter search trials.
- Experiment with pretrained word embeddings.
- Explore Bidirectional LSTM and Bidirectional GRU.
- Improve neutral sentiment classification.
- Experiment with Transformer-based models such as BERT.
- Evaluate robustness using additional datasets and multiple random seeds.
