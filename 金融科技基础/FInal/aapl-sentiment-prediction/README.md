# AAPL Sentiment Prediction Project

This project aims to predict Apple Inc. (AAPL) stock prices by analyzing social media sentiment surrounding the Worldwide Developers Conference (WWDC) events over the past five years. The analysis leverages sentiment data from various social media platforms and correlates it with stock price movements to build predictive models.

## Project Structure

- **notebooks/**: Contains Jupyter notebooks for data analysis and modeling.
  - **AAPL_stock_prediction.ipynb**: The main notebook for predicting AAPL stock prices based on sentiment analysis.

- **data/**: Directory for storing datasets.
  - **raw/**: Contains raw data files, including social media posts and stock data fetched from Tushare.
  - **processed/**: Contains processed data files, such as cleaned and transformed datasets ready for analysis and modeling.

- **requirements.txt**: Lists the required Python packages and their versions needed to run the project.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd aapl-sentiment-prediction
   ```

2. **Install required packages**:
   It is recommended to create a virtual environment before installing the dependencies. You can use `venv` or `conda` for this purpose.

   ```
   pip install -r requirements.txt
   ```

3. **Data Collection**:
   - Raw data should be collected and stored in the `data/raw` directory. This includes social media posts and stock data from Tushare.
   - Ensure that the data is organized properly for further processing.

4. **Running the Notebook**:
   - Open the Jupyter notebook `AAPL_stock_prediction.ipynb` in the `notebooks` directory.
   - Follow the instructions within the notebook to perform sentiment analysis and stock price prediction.

## Usage Guidelines

- The project is designed to analyze sentiment data around WWDC events. Ensure that the sentiment analysis is conducted within the defined time windows around each WWDC event.
- The results of the analysis can be visualized and interpreted to understand the impact of social media sentiment on AAPL stock prices.
- Further enhancements can be made by incorporating additional data sources or refining the sentiment analysis techniques.

## Contributing

Contributions to improve the project are welcome. Please create a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.