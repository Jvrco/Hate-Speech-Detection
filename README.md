# Hate Speech Detection Using LSTM and Dense Neural Networks

This project involves building and training both an LSTM model and a Dense Neural Network (DNN) to detect hate speech, offensive language, and neutral tweets. The dataset contains tweets labeled for different types of speech, and the models are trained using various metrics including accuracy, F1-score, precision, and recall. The project includes the implementation of a custom callback to track these metrics during training and visualize them after the process.

## Project Overview

This project is based on the [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) from Kaggle. The dataset contains tweets labeled as hate speech, offensive language, or neither. The goal of the project is to build and compare two models, an **LSTM-based model** and a **Dense Neural Network (DNN)**, to classify these tweets into one of the three categories.

- **Task**: Detect hate speech and offensive language from tweets.
- **Models**: Long Short-Term Memory (LSTM) network and Dense Neural Network (DNN).
- **Metrics**: Accuracy, F1-score, precision, and recall.
- **Dataset**: A dataset of tweets with annotations for hate speech, offensive language, and neutral content.

### Data Imbalance

The dataset has a significant imbalance, with the majority of the tweets classified as offensive language, fewer as neutral, and the smallest portion as hate speech. This imbalance affects the models' performance, as they may struggle to correctly classify the minority class (hate speech).

- **LSTM**: The imbalance was addressed using **class weighting**. This method assigns higher weights to the underrepresented class (hate speech) during training, helping the model focus on harder-to-classify examples.
- **Dense Neural Network (DNN)**: In contrast, the DNN used **Synthetic Minority Over-sampling Technique (SMOTE)** to balance the dataset. SMOTE generates synthetic samples for the minority class to improve the model's ability to learn from a balanced dataset.

---

## Features

- **Data Preprocessing**: The text data is preprocessed using tokenization and padding to prepare it for input into the models.
- **Custom Metrics Tracking**: A custom Keras callback is implemented to compute and log F1-score, precision, recall, and accuracy for both the training and validation sets.
- **Visualization**: The models' performance is visualized across epochs, allowing you to track the progression of accuracy and other metrics.
- **Class Imbalance Handling**: The imbalance was treated differently for the two models: **class weighting** for the LSTM and **SMOTE** for the DNN.

---

## Dataset

The dataset consists of tweets categorized as:

- **Hate Speech**: Tweets that contain hate speech.
- **Offensive Language**: Tweets that use offensive language but do not qualify as hate speech.
- **Neither**: Neutral tweets that contain neither hate speech nor offensive language.

Make sure to provide the dataset in the expected format before running the notebook.

---

## Model Descriptions

### 1. LSTM Model

The LSTM model was chosen for its ability to capture long-term dependencies in text sequences. However, the performance of the LSTM was not as strong as the DNN, likely due to challenges in handling the class imbalance.

- **Architecture**: 
  - **Embedding Layer**: Converts input words into dense vectors.
  - **Convolutional Layer**: Extracts local patterns from the input sequences.
  - **Batch Normalization**: Normalizes activations for faster convergence.
  - **Max Pooling Layer**: Reduces dimensionality and keeps important features.
  - **LSTM Layer**: Captures long-term dependencies in the tweet sequences.
  - **Dense Layer**: Adds fully connected neurons for additional learning capacity.
  - **Dropout Layers**: Prevent overfitting by randomly setting some neurons to zero.
  - **Output Layer**: Uses softmax activation to classify the input into three categories.

- **Loss Function**: 
  The LSTM uses **Focal Loss** with `gamma=2.0` and `alpha=0.25` to handle class imbalance in combination with **class weighting**.

- **Training**: 
  The LSTM model was trained over 15 epochs with a batch size of 32. A custom `MetricsCallback` was used to log and calculate the accuracy, F1-score, precision, and recall after each epoch for both the training and validation sets.

- **Evaluation**: 
  While the LSTM model performed well in capturing long-term dependencies, it struggled to handle the imbalanced dataset as effectively as the DNN.

### 2. Dense Neural Network (DNN)

The Dense Neural Network (DNN) significantly outperformed the LSTM model, likely due to better handling of the balanced dataset with the use of **SMOTE**. Despite being a simpler model, the DNN was able to classify hate speech and offensive language with greater accuracy.

- **Architecture**: 
  - **Embedding Layer**: Converts input words into dense vectors.
  - **Dense Layers**: A series of fully connected layers with ReLU activations to capture the relationships between the input features.
  - **Dropout Layers**: Prevent overfitting by randomly deactivating a portion of neurons.
  - **Output Layer**: A softmax layer to classify the input text into one of the three categories (hate speech, offensive language, neutral).

- **Loss Function**: 
  The DNN uses **sparse_categorical_crossentropy** as the loss function.

- **Training**: 
  The DNN model was trained using the SMOTE-balanced dataset over 15 epochs with a batch size of 32. The same `MetricsCallback` was used to log and calculate accuracy, F1-score, precision, and recall for both the training and validation sets.

- **Evaluation**: 
  The DNN consistently outperformed the LSTM across all metrics, particularly excelling in detecting hate speech and offensive language. This suggests that the data balancing techniques used (SMOTE) were more effective for the DNN architecture than the class weighting approach in the LSTM.

---

## Training and Evaluation Metrics

The following metrics are computed after each epoch for both models:
- **Accuracy**: Measures the overall correctness of the model's predictions.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.
- **Precision**: Measures how many of the positive predictions were actually correct.
- **Recall**: Measures how many actual positives were correctly identified by the model.

These metrics are tracked and visualized to give insights into the models' performance and their learning progression.

---

## Visualization

The performance of both models is visualized using `matplotlib`, where the accuracy, F1-score, precision, and recall are plotted for both training and validation sets across all epochs. These visualizations help assess whether the models are overfitting and how their performance improves over time.


