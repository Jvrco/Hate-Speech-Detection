# Hate Speech Detection Using LSTM

This project involves building and training an LSTM model to detect hate speech, offensive language, and neutral tweets. The dataset contains tweets labeled for different types of speech, and the model is trained using various metrics including accuracy, F1-score, precision, and recall. The project includes the implementation of a custom callback to track these metrics during training and visualize them after the process.

## Project Overview

This project is based on the [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) from Kaggle. The dataset contains tweets labeled as hate speech, offensive language, or neither. The goal of the project is to build an LSTM-based model to classify these tweets into one of the three categories.
- **Task**: Detect hate speech and offensive language from tweets.
- **Model**: Long Short-Term Memory (LSTM) network.
- **Metrics**: Accuracy, F1-score, precision, and recall.
- **Dataset**: A dataset of tweets with annotations for hate speech, offensive language, and neutral content.

### Data Imbalance

The dataset has a significant imbalance, with the majority of the tweets classified as offensive language, fewer as neutral, and the smallest portion as hate speech. This imbalance affects the model's performance, as it may struggle to correctly classify the minority class (hate speech). To mitigate this, the model uses **Focal Loss** as the loss function, which helps focus the training process on the harder-to-classify examples.

## Features

- **Data Preprocessing**: The text data is preprocessed using tokenization and padding to prepare it for input into the LSTM model.
- **Custom Metrics Tracking**: A custom Keras callback is implemented to compute and log F1-score, precision, recall, and accuracy for both the training and validation sets.
- **Visualization**: The model's performance is visualized across epochs, allowing you to track the progression of accuracy and other metrics.
- **Class Weighting**: The class imbalance is handled by using `class_weight` to ensure the model does not favor the majority class.

## Dataset

The dataset consists of tweets categorized as:

- **Hate Speech**: Tweets that contain hate speech.
- **Offensive Language**: Tweets that use offensive language but do not qualify as hate speech.
- **Neither**: Neutral tweets that contain neither hate speech nor offensive language.

Make sure to provide the dataset in the expected format before running the notebook.

## Model Architecture

The model is built using a combination of layers to process and classify the tweet sequences into three categories: hate speech, offensive language, and neutral. The architecture includes:

- **Embedding Layer**: Converts input words into dense vectors.
- **Convolutional Layer**: Extracts local patterns from the input sequences.
- **Batch Normalization**: Normalizes activations for faster convergence.
- **Max Pooling Layer**: Reduces dimensionality and keeps important features.
- **LSTM Layer**: Captures long-term dependencies in the tweet sequences.
- **Dense Layer**: Adds fully connected neurons for additional learning capacity.
- **Dropout Layers**: Prevent overfitting by randomly setting some neurons to zero.
- **Output Layer**: Uses softmax activation to classify the input into three categories.

### Loss Function

The model uses **Focal Loss** to handle class imbalance, making it focus more on hard-to-classify examples, with `gamma=2.0` and `alpha=0.25`.

## Training

The training is performed over 15 epochs using a batch size of 32. The `MetricsCallback` is used to log and calculate the following metrics after each epoch:

- **Accuracy**
- **F1-Score**
- **Precision**
- **Recall**

These metrics are calculated for both the training and validation sets, and are stored for later visualization.

## Evaluation

After training, the model is evaluated on the test set to measure the final performance. The following metrics are used:

- **Accuracy**: Measures the overall correctness of the model's predictions.
- **F1-Score**: The harmonic mean of precision and recall.
- **Precision**: Measures how many of the positive predictions were actually correct.
- **Recall**: Measures how many actual positives were correctly identified by the model.

## Visualization

The performance of the model is visualized using `matplotlib`, where the accuracy, F1-score, precision, and recall are plotted for both training and validation sets across all epochs. These plots give insights into how well the model is learning and whether it is overfitting.

