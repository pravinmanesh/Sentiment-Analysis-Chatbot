# Sentiment Analysis Chatbot

This repository contains a simple sentiment analysis chatbot using TensorFlow and the IMDB dataset. The chatbot predicts the sentiment (positive or negative) of user-inputted text based on a pre-trained LSTM model.

## Table of Contents

- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Predicting Sentiment](#predicting-sentiment)
- [Contributing](#contributing)

## Usage

To start the chatbot and predict the sentiment of input text, run:

```bash
python sentiment_analysis_chatbot.py
You will be prompted to enter text for sentiment analysis. To exit the program, type stop.

Model Architecture
The sentiment analysis model is built using TensorFlow and Keras. It consists of the following layers:
1. An Embedding layer to convert word indices to dense vectors of fixed size.
2. An LSTM (Long Short-Term Memory) layer to capture temporal dependencies in the data.
3. A Dense layer with a sigmoid activation function to output the sentiment prediction.

Dataset
The model is trained on the IMDB dataset, which consists of movie reviews labeled as either positive or negative. The dataset is preprocessed to keep only the top 10,000 most frequent words, and all reviews are padded to a maximum length of 100 words.

Training the Model
The model is trained using binary cross-entropy loss and the Adam optimizer. The training process includes:
1. Loading and preprocessing the IMDB dataset.
2. Building the model with the specified architecture.
3. Training the model for 5 epochs with a batch size of 32.
4. Evaluating the model on the test set.

Predicting Sentiment
To predict the sentiment of a user-inputted text, the following steps are performed:
1. Preprocess the input text by encoding the words using the IMDB word index.
2. Pad the encoded sequence to match the input length expected by the model.
3. Use the trained model to predict the sentiment score.
4. Classify the sentiment as positive if the score is greater than 0.5, otherwise classify it as negative.

Contributing
Contributions are welcome! Please fork this repository and submit pull requests with your improvements.
