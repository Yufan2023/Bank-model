#  Stock Prediction Project with LSTM and Sentiment Analysis 


## Project Overview
This project aims to predict the stock price of TD Bank using a Long Short-Term Memory (LSTM) model. The model incorporates historical stock data, economic indicators, sentiment analysis from political news, and technical indicators. The project also provides the ability to evaluate predictions using future data as it becomes available.

## Project Structure

stock-prediction-project/ <br>
├── src/ <br>
│   ├── main.py                   # Main script to train and evaluate the model<br>
│   ├── data_loader.py            # Code for downloading historical and future data<br>
│   ├── preprocessing.py          # Data preprocessing and feature engineering logic<br>
│   ├── model.py                  # Model creation, training, saving, and loading logic<br>
│   ├── sentiment_analysis.py     # Code for performing sentiment analysis<br>
│   ├── collect_future_data.py    # Script for collecting future data periodically<br>
│   └── utils.py                  # Utility functions for plotting and evaluation<br>
├── models/                       # Directory for saved trained models<br>
│   └── lstm_model.h5             # Trained LSTM model file<br>
├── results/                      # Output results, plots, and logs<br>
├── tests/                        # Unit tests for code validation<br>
│   ├── test_data_loader.py<br>
│   ├── test_preprocessing.py<br>
│   ├── test_model.py<br>
│   ├── test_sentiment_analysis.py<br>
│   └── test_collect_future_data.py<br>
├── requirements.txt              # List of Python dependencies<br>
├── .gitignore                    # Git ignore file for excluding certain files/folders<br>
└── README.md                     # Project overview and setup instructions<br>

## Features
Historical Stock Data: Downloaded using yfinance for multiple banks and indices. <br>
Economic Indicators: Includes data such as interest rates and volatility indices. <br>
Sentiment Analysis: Utilizes the vaderSentiment library for extracting sentiment scores from news headlines. <br>
Technical Indicators: Computes 20-day and 200-day moving averages. <br>
Future Data Collection: Collects future data to compare predictions against real values. <br>

## Prerequisites

Ensure you have Python 3.7+ installed and the following libraries:

* `numpy`
* `pandas`
* `yfinance`
* `scikit-learn`
* `tensorflow`
* `matplotlib`
* `vaderSentiment`

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Setup

1. **Clone this repository**:
   ```bash
   git clone https://github.com/Yufan2023/Bank-model.git
   cd stock-prediction-project
   ```
2. Navigate to the `src` folder: Make sure to run scripts from the `src` directory to ensure proper pathing for data loading and processing.
3. Run the main script: To train the model and evaluate its performance
   ```bash
   python src/main.py
   ```
4. Run future data collection: To collect and test with future data:
   ```bash
   python tests/collect_future_data.py
   ```


   








