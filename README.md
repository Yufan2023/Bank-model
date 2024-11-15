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

The project is organized into several well-defined directories and files to ensure clarity and modularity. The ```data/``` folder contains raw or processed data files used for analysis and modeling, along with a ```README.md``` that explains how to obtain or use the dataset. The ```notebooks/``` directory includes Jupyter Notebooks for each key step in the workflow: ```data_preprocessing.ipynb``` for data cleaning and feature engineering, ```model_training.ipynb``` for training the LSTM model, and ```predictions_and_visualization.ipynb``` for generating predictions and visualizations.

The ```src/``` directory houses Python scripts for modular functionality. It includes ```preprocess_data.py``` for preprocessing tasks, ```train_model.py``` for model training, and ```predict_future.py``` for making predictions and plotting results. The ```images/``` folder stores visual assets, such as plots of actual vs. predicted prices and future forecasts, for documentation and presentation purposes.

At the root level, the ```README.md``` provides a comprehensive overview of the project, while ```requirements.txt``` lists the necessary Python dependencies. The ```.gitignore``` file specifies files and directories to exclude from version control. The ```main.py``` script serves as the entry point, combining all modules to execute the complete workflow. A LICENSE file is included to define the project's usage rights. This structure ensures the project is easy to navigate, reproducible, and extensible for future improvements.


## Installation and Usage
1. Clone the Repository
   ```bash
   git clone https://github.com/Yufan2023/Bank-model.git
   cd Bank-model
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



   








