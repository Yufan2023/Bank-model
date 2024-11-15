#  Stock Prediction Project with LSTM and Sentiment Analysis 


## Project Overview
This project predicts the stock prices of TD Bank using a Long Short-Term Memory (LSTM) model. It incorporates:

Historical stock data: Closing prices of major banks and financial sectors.
Economic indicators: Interest rates, volatility index, and S&P 500.
Political dynamics: Sentiment analysis of political news and event-based indicators.
The model forecasts both historical trends and future stock prices while analyzing the impact of political and economic factors.

## Project Structure

src/

This folder contains the core scripts essential for the project:

main.py: The primary script responsible for training and evaluating the stock prediction model.

data_loader.py: Handles the downloading of both historical and future stock data.

preprocessing.py: Contains the logic for data preprocessing and feature engineering.

model.py: Includes code for creating, training, saving, and loading the model.

sentiment_analysis.py: Script dedicated to performing sentiment analysis.

collect_future_data.py: Runs periodically to gather future data.

utils.py: Consists of utility functions used for plotting and evaluating model performance.

models/

This directory stores the trained models. For example, the LSTM model is saved as lstm_model.h5.

results/

This folder is used to store output results, visual plots, and log files generated during the project.

tests/

Contains unit test scripts to validate the functionality of the code:

test_data_loader.py: Tests for the data loading module.

test_preprocessing.py: Tests related to data preprocessing.

test_model.py: Verifies the model creation and training logic.

test_sentiment_analysis.py: Ensures the sentiment analysis script is functioning correctly.

test_collect_future_data.py: Checks the periodic data collection script.

requirements.txt

Lists all Python libraries and dependencies needed for the project.

.gitignore

Specifies files and folders to exclude from version control.

README.md

Provides an overview of the project, along with setup instructions and explanations for getting started.


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


   








