#  Stock Prediction Project with LSTM and Sentiment Analysis 


## Project Overview
This project aims to predict the stock price of TD Bank using a Long Short-Term Memory (LSTM) model. The model incorporates historical stock data, economic indicators, sentiment analysis from political news, and technical indicators. The project also provides the ability to evaluate predictions using future data as it becomes available.

## Project Structure

stock-prediction-project/
├── src/
│   ├── main.py                   # Main script to train and evaluate the model
│   ├── data_loader.py            # Code for downloading historical and future data
│   ├── preprocessing.py          # Data preprocessing and feature engineering logic
│   ├── model.py                  # Model creation, training, saving, and loading logic
│   ├── sentiment_analysis.py     # Code for performing sentiment analysis
│   ├── collect_future_data.py    # Script for collecting future data periodically
│   └── utils.py                  # Utility functions for plotting and evaluation
├── data/                         # Placeholder for raw and processed data
│   ├── raw/                      # Subfolder for raw historical data
│   ├── processed/                # Subfolder for processed data ready for training/testing
│   └── future/                   # Subfolder for storing collected future data
├── models/                       # Directory for saved trained models
│   └── lstm_model.h5             # Trained LSTM model file
├── results/                      # Output results, plots, and logs
├── notebooks/                    # Jupyter notebooks for exploration and demonstration
│   └── exploratory_analysis.ipynb  # Example notebook for analyzing predictions
├── tests/                        # Unit tests for code validation
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_sentiment_analysis.py
│   └── test_collect_future_data.py
├── requirements.txt              # List of Python dependencies
├── .gitignore                    # Git ignore file for excluding certain files/folders
└── README.md                     # Project overview and setup instructions

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
   git clone https://github.com/Yufan2023/stock-prediction-project.git
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


   








