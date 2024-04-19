# Intern Projects

## 1. Stock Price Prediction Project (Minor Project)

In this project, we predict the future prices of a stock based on historical data. The goal is to analyze stock market trends and develop a model that can provide accurate predictions of future prices.

### Dataset

The dataset contains historical stock price data, including:

- `Date`: The date of the stock data.
- `Open`: The opening price of the stock.
- `High`: The highest price of the stock during the day.
- `Low`: The lowest price of the stock during the day.
- `Close`: The closing price of the stock.
- `Volume`: The volume of stocks traded during the day.

### Data Exploration

The data exploration phase includes:

- **Univariate Analysis**: Descriptive statistics such as mean, median, and standard deviation for each column.
- **Time Series Analysis**: Identifying trends and patterns in stock prices over time.
- **Visualizations**: Plots such as line charts and candlestick charts to visualize price movements.

### Predictive Modeling

We applied several machine learning algorithms for stock price prediction:

- **Linear Regression**: A simple model to predict future prices based on historical data.
- **ARIMA**: A statistical model designed for time series data.
- **LSTM**: A recurrent neural network model for more complex time series forecasting.

### Results

The project results include:

- Evaluation of different models' performance using metrics such as RMSE.
- Identifying the best model for predicting future stock prices.

## 2. Wine Quality Prediction Project (Major Project)

In this project, we analyze a dataset containing physicochemical properties of wines along with their quality ratings. The goal is to perform exploratory data analysis (EDA), identify patterns, and potentially build predictive models to predict wine quality based on its attributes.

### Dataset

The dataset contains 1,599 observations and 12 columns, including:

- `fixed_acidity`: Represents the amount of non-volatile acids in the wine.
- `volatile_acidity`: Measures the amount of acetic acid in the wine.
- `citric_acid`: Indicates the presence of citric acid in the wine.
- `residual_sugar`: Describes the amount of sugar remaining in the wine after fermentation.
- `chlorides`: Represents the amount of salt in the wine.
- `free_sulfur_dioxide`: Measures the free form of SO2 present in the wine.
- `total_sulfur_dioxide`: Represents the total amount of SO2 present in the wine.
- `density`: Describes the density of the wine.
- `pH`: Indicates the acidity or basicity of the wine on a scale from 0 to 14.
- `sulphates`: Represents the amount of sulphates in the wine.
- `alcohol`: Indicates the alcohol content of the wine.
- `quality`: Represents the quality of the wine on a scale from 3 to 8.

### Data Exploration

The data exploration phase includes:

- **Univariate Analysis**: Statistical summaries such as mean, median, standard deviation, and skewness for each column.
- **Outlier Analysis**: Calculating and summarizing outliers in the dataset.
- **Visualizations**: Various plots including histograms, box plots, and pair plots to understand the distribution and relationships between features.

### Predictive Modeling

We applied several machine learning algorithms to predict wine quality:

- **Logistic Regression**: Used for binary classification of quality into two classes: high quality and low quality.
- **Random Forest**: An ensemble learning method used for classification.
- **SVM**: Support Vector Machine models with different kernels.
- **Decision Tree**: A tree-based classifier for modeling wine quality.
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.

### Pipeline and Hyperparameter Tuning

We used pipelines for each model with different scaling techniques:

- **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance.
- **MinMaxScaler**: Scales features to a given range (default: 0 to 1).
- **RobustScaler**: Scales features using the median and interquartile range.

The models were trained using grid search cross-validation to find the best hyperparameters for each algorithm.

### Results

The project results include:

- The best model found through grid search, with its corresponding training and testing scores.
- Analysis of feature importance using the Random Forest model.

### Conclusion

The Random Forest model provided the best performance for wine quality prediction, with high accuracy in training and testing sets. Further work can include additional feature engineering and model tuning.

## License

Both projects are licensed under the [MIT License](LICENSE).

## Contributing

Contributions to these projects are welcome. Please follow the standard GitHub process for contributing.
