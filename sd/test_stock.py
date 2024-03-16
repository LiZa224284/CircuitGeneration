"""
从alpha_vantage获取股票数据，然后使用pandas进行数据处理，最后使用matplotlib进行数据可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

# Install required packages
# !pip install alpha_vantage matplotlib

your_api_key="0P6VGT023KQ96VM"

# 获取股票数据
def get_stock_data(stock_code):
    ts = TimeSeries(key='your_api_key', output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock_code, outputsize='full')
    return data

# 数据处理
def process_data(data):
    data = data['4. close']
    data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    return data

# 数据可视化
def visualize_data(data):
    plt.figure(figsize=(10, 6))
    data['4. close'].plot()
    plt.title('Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

if __name__ == '__main__':
    stock_code = 'AAPL'
    data = get_stock_data(stock_code)
    data = process_data(data)
    visualize_data(data)

# 从alpha_vantage获取股票数据，然后使用pandas进行数据处理，最后使用matplotlib进行数据可视化