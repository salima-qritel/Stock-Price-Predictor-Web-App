# 📈 Stock Price Predictor Web App

## 📌 Project Description
This project is a **web application for predicting stock prices** using **LSTM deep learning models**.  
It allows users to select any stock ticker, visualize historical data, calculate moving averages, and predict future prices using a pre-trained LSTM model.

The app is built with **Streamlit** for an interactive interface and **Keras** for deep learning.

---

## 💻 Features
- Fetch historical stock data via **Yahoo Finance** (`yfinance`)  
- Visualize **closing prices** with **interactive plots**  
- Display **moving averages** (100, 200, 250 days)  
- Predict future stock prices using a **pre-trained LSTM model**  
- Compare **predictions vs actual values**  
- Modern **dark-themed UI** for better user experience  

---

## ⚙️ Technology Stack
- **Python 3.10+**  
- **Libraries**:
  - Data handling: `pandas`, `numpy`  
  - Visualization: `matplotlib`  
  - Stock data: `yfinance`  
  - Deep Learning: `tensorflow / keras` (LSTM)  
  - Web App: `streamlit`  
  - Preprocessing: `sklearn.preprocessing.MinMaxScaler`  

---

## 🔍 How It Works
1. **User Input**: Enter the stock ticker (e.g., `GOOG`) and select the date range.  
2. **Data Download**: Historical stock data is fetched from Yahoo Finance.  
3. **Visualization**: 
   - Plot closing prices  
   - Compute and plot moving averages  
4. **Data Preprocessing**: 
   - Scale the closing prices with `MinMaxScaler`  
   - Create sequences of 100 days for LSTM input  
5. **Prediction**: 
   - Load the pre-trained LSTM model (`Latest_stock_price_model.keras`)  
   - Predict future prices  
   - Inverse transform to actual price scale  
6. **Results**: Display plots comparing **predicted vs actual prices**  

---

## 📊 Model Details
- **Architecture**: Sequential LSTM
  - LSTM layer 1: 128 units, `return_sequences=True`  
  - LSTM layer 2: 64 units  
  - Dense layer: 25 units  
  - Output Dense layer: 1 unit (predicted price)  
- **Training**:
  - Optimizer: `Adam`  
  - Loss: `Mean Squared Error (MSE)`  
  - Epochs: 1 (for demo; can be increased for better accuracy)  
- **Performance Metrics** (on test set):
  - MSE: 11.4542  
  - MAE: 2.4774  
  - RMSE: 3.3844  
  - R²: 0.9922  
  - Adjusted R²: 0.9922  

---

## 🚀 How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/username/stock-price-predictor.git
   cd stock-price-predictor
   ```
2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app
```bash
streamlit run web_stock_price_predictor.py
```

4. Enter a stock ticker and date range, then explore visualizations and predictions.

##📌 Notes

- Make sure the pre-trained model Latest_stock_price_model.keras is in the same folder as the app.

- The app works best with stocks that have sufficient historical data.

- You can retrain the model using the provided Jupyter notebook (stock_price.ipynb) for custom datasets.

## ✨ Author

👤 Developed by Salima-Qritel
]
