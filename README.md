#  Stock Prediction Project with LSTM and Sentiment Analysis 


## Project Overview
This project predicts the stock prices of TD Bank using a Long Short-Term Memory (LSTM) model. It incorporates:

## Features
Multivariate Time Series Analysis: Uses historical stock and economic data. <br>
Sentiment Integration: Performs sentiment analysis using VADER on political news headlines.<br>
Event Indicators: Marks significant political events (e.g., elections, economic agreements).<br>
Deep Learning Model: Predicts historical trends and future stock prices using LSTM.<br>
Visualization: Displays actual vs. predicted prices and future forecasts.<br>

Historical stock data: Closing prices of major banks and financial sectors.<br>
Economic indicators: Interest rates, volatility index, and S&P 500.<br>
Political dynamics: Sentiment analysis of political news and event-based indicators.<br>
The model forecasts both historical trends and future stock prices while analyzing the impact of political and economic factors.<br>

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


## Installation and Usage
1. Clone the Repository
   ```bash
   git clone https://github.com/Yufan2023/Bank-model-with-LSTM.git
   cd Bank-model-with-LSTM
   ```

2. Set Up a Virtual Environment
```bash
python -m venv env
source env/bin/activate   # On Windows: .\env\Scripts\activate
pip install -r requirements.txt
```

3. Run the Project
   ```bash
   python main.py
   ```

## Prerequisites

Ensure you have Python 3.7+ installed and the following libraries:

* `numpy`
* `pandas`
* `yfinance`
* `scikit-learn`
* `tensorflow`
* `matplotlib`
* `vaderSentiment`

## Results
Actual vs. Predicted Prices <br>
This plot compares the model’s predicted prices with the actual stock prices during the test period.



   








