Link = https://stock-price-fahaegqovvbfj5z4cfzrjz.streamlit.app/

TCS Stock Price Prediction using LSTM

Project Overview
The TCS Stock Price Prediction project leverages the power of Long Short-Term Memory (LSTM) neural networks to predict the future closing prices of ANY stock. By utilizing historical stock price data, this project aims to provide insights and predictions that can be beneficial for traders and investors.

Technologies Used
1. Python: The core programming language used for data manipulation, analysis, and model development.
2. Pandas: For data preprocessing and manipulation.
3. NumPy: For numerical computations.
4. Matplotlib: For data visualization.
5. yFinance: To fetch historical stock price data from Yahoo Finance.
6. Scikit-Learn: For scaling the data using MinMaxScaler.
7. Keras and TensorFlow: To build and train the LSTM neural network.

Data Collection
The historical stock price data for TCS was collected using the yFinance library. The data includes various features such as Open, High, Low, Close, Volume, and Adj Close prices over a specified date range from January 1, 2005, to December 31, 2024.

Data Preprocessing
1. Missing Values: The dataset was cleaned by dropping any rows with missing values.
2. Feature Selection: Only the 'Close' price was selected for the prediction model.
3. Scaling: The 'Close' prices were scaled to a range of 0 to 1 using the MinMaxScaler to ensure the model performs optimally.
4. Creating Training and Test Sets: The data was split into training (80%) and testing (20%) sets.
5. Generating Sequences: Sequences of 100 days' worth of data were used as input features to predict the next day's closing price.

Model Development
An LSTM neural network was developed using the Keras library. The architecture of the model includes:
1. Four LSTM layers with different units (50, 60, 80, 120), each followed by Dropout layers to prevent overfitting.
2. A Dense layer to produce the final output.
3. The model was compiled using the 'adam' optimizer and 'mean squared error' as the loss function.

Model Training
The model was trained on the training set for 50 epochs with a batch size of 32. The training process aimed to minimize the loss function and optimize the model's weights for better prediction accuracy.

Results and Predictions
The trained model was used to predict the stock prices on the test set. The predicted values were then compared with the actual stock prices to evaluate the model's performance. Visualizations were created to illustrate the predictions versus the actual prices.

Deployment
The model was deployed as a web application, allowing users to input a date range and receive predicted closing prices for TCS stock. This deployment enables real-time predictions and provides a user-friendly interface for investors and traders.

Future Work
1. Incorporate More Features: Including additional features such as trading volume, technical indicators, and news sentiment analysis to improve prediction accuracy.
2. Model Optimization: Experimenting with different model architectures and hyperparameters to enhance performance.
3. Expand to Other Stocks: Generalizing the model to predict stock prices for other companies and integrating multiple stock predictions in the web application.
4. Real-time Data: Incorporating real-time stock data for up-to-date predictions.

Conclusion
This project demonstrates the potential of LSTM neural networks in time series forecasting, specifically for predicting stock prices. By providing accurate predictions and a user-friendly web interface, this project can aid investors in making informed decisions.
