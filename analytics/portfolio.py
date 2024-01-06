import warnings
warnings.filterwarnings('ignore')
from datetime import datetime as dt
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.stats.stattools import jarque_bera
import seaborn as sns


plt.style.use('seaborn-v0_8-darkgrid')
bg_color_light = '#3A3B3C'
bg_color = '#242526'
fg_color = '#B0B3B8'

plt.rcParams['font.size'] = 11 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['axes.labelcolor'] = fg_color
plt.rcParams['axes.titlecolor'] = fg_color
plt.rcParams['axes.facecolor'] = bg_color
plt.rcParams['xtick.color'] = fg_color
plt.rcParams['xtick.labelcolor'] = fg_color
plt.rcParams['ytick.color'] = fg_color
plt.rcParams['ytick.labelcolor'] = fg_color
plt.rcParams['grid.color'] = bg_color_light
plt.rcParams['figure.facecolor'] = bg_color
plt.rcParams['legend.labelcolor'] = fg_color


class Analytics:
    
    def __init__(self, dataset:pd.DataFrame, initial_deposit:float=None, start_date:dt = None, end_date: dt = None, magic: int = None):
        self.initial_deposit = initial_deposit
        self.start_date = start_date 
        self.end_date = end_date
        self.magic = magic
        

        self.trading_data = self.process(dataset)

        
        
    def process(self, data:pd.DataFrame):
        
        

        # Get initial deposit
        self.initial_deposit = data.loc[(data.isnull().any(axis = 1)) & (~data['profit'].isnull())]['profit'].item() if self.initial_deposit is None else self.initial_deposit

        # Drop null
        data = data.dropna()
        
        # Cast types
        float_cols = ['lots','order_open_price','sl','tp','order_close_price','commission','profit']
        int_cols = ['ticket']
        string_cols = ['symbol']

        data['open_time'] = pd.to_datetime(data['order_open_time'])
        data['close_time'] = pd.to_datetime(data['order_close_time'])
        data[float_cols] = data[float_cols].astype(float) 
        data[int_cols] = data[int_cols].astype(int)
        data[string_cols] = data[string_cols].astype(str)

        
        # Derived columns

        data['net_profit'] = data['profit'] + data['commission']
        data['running_profit'] = data['net_profit'].cumsum()

        data['equity'] = data['running_profit'] + self.initial_deposit
        data['starting_equity'] = data['equity'].shift(1)
        data = data.fillna(self.initial_deposit)
        data['peak_equity'] = data['equity'].cummax()
        data['drawdown'] = (1 - (data['equity'] / data['peak_equity'])) * 100

        data['gain'] = (data['net_profit'] / data['starting_equity']) * 100
        data['cumm_gain'] = data['gain'].cumsum()
        data['returns_percent'] = (data['running_profit'] / data['starting_equity']) * 100

        data = data.set_index('open_time', drop=True)

        if self.start_date is not None:
            data = data.loc[data.index.date >= self.start_date.date()]
            
        if self.end_date is not None:
            data = data.loc[data.index.date <= self.end_date.date()]

        if self.magic is not None:
            data = data.loc[data['magic'] == self.magic]

        return data

    def plot_equity_curve(self):

        data = self.trading_data.copy()

        # required columns: balance, drawdown
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15,8), sharex=True, gridspec_kw={'height_ratios':[2,1]})

        plt.subplots_adjust(left=0.1, right=0.9, top = 0.95, bottom = 0.1, hspace=0.1)
        fig.suptitle('Equity Curve and Drawdown', color='white')

        bal = data['starting_equity'] - 100000
        bal.plot(ax=ax1, kind = 'area', color ='springgreen', alpha=0.2, stacked =False)
        ax1.set_ylabel('Profit ($)')

        dd = data['drawdown'] * -1
        dd.plot(ax=ax2, kind='area', color = 'red', alpha =0.3)
        ax2.set_ylabel('Drawdown (%)')

        plt.xlabel('Date')
        plt.show()
    
    def plot_returns_distribution(self):
        
        data = self.trading_data[['net_profit']]
        data_to_plot = data.loc[data['net_profit'] != 0]

        frmt = '{:.3g}'
        jb, jb_p, skew, kurt=tuple([j.item() for j in jarque_bera(data_to_plot)])
        jb_string = f'JB: {jb:.2f}\np-value: {frmt.format(jb_p)}\nSkew: {skew:.2f}\nKurt: {kurt:.2f}'

        sns.displot(data_to_plot, kde=True, legend=False, height=5, aspect=1.5, alpha=0.4)

        plt.xlabel('Profit ($)')
        plt.title(f'Profit Distribution\n{jb_string}', fontsize = 12)
        plt.show()

        # PL BY TIME PERIOD 
    def plot_profit_by_time_period(self, interval:str = "month"):
        
        intervals = ['month','day_of_week','year']
        if interval not in intervals:
            raise ValueError("Invalid Interval. Use month, day_of_week")

        data = self.trading_data.copy()

        start_date = data[:1].index.item()
        end_date = data[-1:].index.item()

        def grouping(data, interval):
            if interval == "month":
                return data.index.month 
            elif interval == "day_of_week":
                return data.index.dayofweek
            elif interval == "year":
                return data.index.year
            
        def index_label(interval):
            months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            days = ['Mon','Tue','Wed','Thu','Fri']
            if interval == "month":
                return 1, months 
            elif interval == "day_of_week":
                return 0, days
            elif interval == "year":
                return 0, data.index.year.unique().tolist()


        profit = data[['net_profit']]

        profit_trades = profit.loc[profit['net_profit'] > 0]
        loss_trades = profit.loc[profit['net_profit'] <= 0]


        profit_sum = profit_trades.groupby(grouping(profit_trades,interval)).sum()
        loss_sum = loss_trades.groupby(grouping(loss_trades,interval)).sum()

        profit_sum['loss'] = loss_sum * -1 

        gross = profit_sum.copy()
        gross.columns = ['gross_profit', 'gross_loss']

        if interval != 'year':
            start_index, interval_label = index_label(interval)
            gross = gross.reindex(np.arange(start_index,len(interval_label)+start_index,1))
            gross.index = gross.index.map({k+start_index:v for k,v in enumerate(interval_label)})
            gross = gross.fillna(0)

        gross.plot(kind = 'bar', color = ['dodgerblue','crimson'], alpha = 0.6, edgecolor='black', figsize=(12, 5))
        interval_title = interval.replace('_',' ').title()
        plt.title(f'P/L by {interval_title} from {start_date.date()} to {end_date.date()}')
        plt.legend(labels=['Gross Profit','Gross Loss'])
        plt.xlabel(interval_title)
        plt.ylabel('Profit ($)')
        plt.grid(axis = 'x')
        plt.show()

    def plot_profit_by_instrument(self):
        pending = ['buy_limit','sell_limit','buy_stop','sell_stop']
        traded = self.trading_data.loc[~self.trading_data['order_type'].isin(pending)]
        grouped_symbol = traded.groupby('symbol')['net_profit'].sum().sort_values(ascending = False)
        grouped_symbol.plot(kind = 'bar', figsize=(12,5), color = 'dodgerblue', alpha=0.5, edgecolor = 'black')
        plt.title('Returns by Instrument')
        plt.xlabel('Instrument')
        plt.ylabel('Net Profit ($)')
        plt.grid(axis = 'x')
        plt.show()

    def plot_profit_by_position_type(self):
        pending = ['buy_limit','sell_limit','buy_stop','sell_stop']
        traded = self.trading_data.loc[~self.trading_data['order_type'].isin(pending)]
        grouped_order = traded.groupby('order_type')['net_profit'].sum().sort_values(ascending = False)
        grouped_order.index = [g.capitalize() for g in grouped_order.index]
        grouped_order.plot(kind = 'bar', figsize=(12,5), color = 'dodgerblue', alpha=0.5, edgecolor = 'black')
        plt.title('Returns by Order Type')

        plt.xlabel('Order Type')
        plt.ylabel('Net Profit ($)')
        plt.grid(axis = 'x')
        plt.show()
