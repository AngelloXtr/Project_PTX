__author__ = 'cgomezfandino@gmail.com'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import seaborn

class Momentum_Backtester(object):

    ''' Momentum backtesting strategy:
    Attributes
    ==========
    symbol: str
        Google Finance symbol with which to work with
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g. 0.3% = 0.003) per trade

    Methods
    =========
    get_data:
        retrieves and prepares the base data set
    run_strategy:
        runs the backtest for the momentum-based strategy
    plot_strategy:
        plots the performance of the strategy compared to the symbol
    '''
    def __init__(self, symbol, start, end, amount=10000, tc=0.001, lvrage= 1, data_src='google'):

        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.lvrage = lvrage
        # price = OHLC
        self.data_source = data_src
        self.get_data()
        self.results = None
        self.toplot_c = ['creturns_c']
        self.toplot_p = ['creturns_p']
        self.toplot_hist = ['returns']

    def get_data(self):

        data = web.DataReader(name=self.symbol, data_source=self.data_source, start=self.start, end=self.end)
        ## Asset Returns
        data['returns'] = np.log(data['Close']/data['Close'].shift(1))
        self.asset = data

    def run_strategy(self, momentum = 1):

        '''
        This function run a momentum backtest.

        :param momentum:
        ================
        Number of lags you want to to test for momuntum strategy

        :return:
        ================
        The backtest returns the following values:
        aperf_c: Absolute Strategy performance in Cash
        aperf_p: Absolute Strategy performance in Percentage
        operf_c: Out-/underperformance Of strategy in Cash
        operf_p: Out-/underperformance Of strategy in Percentage
        mdd_c: Maximum Drawdown in Cash
        mdd_p:Maximum Drawdown in Percentage
       '''

        asset = self.asset.copy()
        self.momentum = momentum
        # self.str_rtrn = ['returns']
        # self.drawdown = []
        # self.cumrent = []

        ## Position

        asset['creturns_c'] = self.amount * asset['returns'].cumsum().apply(lambda x: x * self.lvrage).apply(np.exp)
        asset['creturns_p'] = asset['returns'].cumsum().apply(lambda x: x * self.lvrage).apply(np.exp)
        asset['cmreturns_c'] = asset['creturns_c'].cummax()
        asset['cmreturns_p'] = asset['creturns_p'].cummax()
        asset['ddreturns_c'] = asset['cmreturns_c'] - asset['creturns_c']
        asset['ddreturns_p'] = asset['cmreturns_p'] - asset['creturns_p']

        dicti = {'Momentum Strategies': {}}
        x = []
        y = []
        z = []

        for i in momentum:
            asset['position_%i' % i] = np.sign(asset['returns'].rolling(i).mean())
            asset['strategy_%i' % i] = asset['position_%i' % i].shift(1) * asset['returns']

            asset['lstrategy_%i' % i] = asset['strategy_%i' % i] * self.lvrage
            self.toplot_hist.append('lstrategy_%i' % i)

            ## determinate when a trade takes places (long or short)
            trades = asset['position_%i' % i].diff().fillna(0) != 0

            ## subtracting transaction cost from return when trade takes place
            asset['lstrategy_%i' % i][trades] -= self.tc

            ## Cumulative returns in Cash
            # asset['cstrategy_c_%i' % i] = self.amount * asset['strategy_%i' % i].cumsum().apply(lambda x: x * self.lvrage).apply(np.exp)
            asset['cstrategy_c_%i' % i] = self.amount * asset['lstrategy_%i' % i].cumsum().apply(np.exp)

            ## Cumulative returns in percentage
            # asset['cstrategy_p_%i' % i] = asset['strategy_%i' % i].cumsum().apply(lambda x: x * self.lvrage).apply(np.exp)
            asset['cstrategy_p_%i' % i] = asset['lstrategy_%i' % i].cumsum().apply(np.exp)

            ## Max Cummulative returns in cash
            asset['cmstrategy_c_%i' % i] = asset['cstrategy_c_%i' % i].cummax()

            ## Max Cummulative returns in percentage
            asset['cmstrategy_p_%i' % i] = asset['cstrategy_p_%i' % i].cummax()

            ## Max Drawdown un Cash
            asset['ddstrategy_c_%i' % i] = asset['cmstrategy_c_%i' % i] - asset['cstrategy_c_%i' % i]

            ## Max Drawdown in Percentage
            asset['ddstrategy_p_%i' % i] = asset['cmstrategy_p_%i' % i] - asset['cstrategy_p_%i' % i]

            self.toplot_c.append('cstrategy_c_%i' % i)
            self.toplot_p.append('cstrategy_p_%i' % i)

            ## save asset df into self.results
            self.results = asset

            ## Final calculations for return

            ## absolute Strategy performance in Cash:
            aperf_c = self.results['cstrategy_c_%i' % i].ix[-1]
            ## absolute Strategy performance in Percentage:
            aperf_p = self.results['cstrategy_p_%i' % i].ix[-1]
            ## Out-/underperformance Of strategy in Cash
            operf_c = aperf_c - self.results['creturns_c'].ix[-1]
            ## Out-/underperformance Of strategy in Percentage
            operf_p = aperf_p - self.results['creturns_p'].ix[-1]

            ## Maximum Drawdown in Cash
            mdd_c = self.results['ddstrategy_c_%i' % i].max()
            ## Maximum Drawdown in Percentage
            mdd_p = self.results['ddstrategy_p_%i' % i].max()

            keys = ['aperf_c_%i' % i, 'aperf_p_%i' % i, 'operf_c_%i' % i, 'operf_p_%i' % i, 'mdd_c_%i' % i, 'mdd_p_%i' % i]
            values = ['%.2f' % np.round(aperf_c, 2), '%.2f' % np.round(aperf_p, 2), '%.2f' % np.round(operf_c, 2),
                      '%.2f' % np.round(operf_p, 2), '%.2f' % np.round(mdd_c, 2), '%.2f' % np.round(mdd_p, 2)]

            res = dict(zip(keys, values))

            dicti['Momentum Strategies']['strategy_%i' % i] = res

            x.append(i)
            y.append(aperf_p)
            z.append(mdd_p)


        self.x = x #momentums
        self.y = y #final returns
        self.z = z #mdd

        return dicti


    def plot_strategy(self):
        # self.results = self.run_strategy()

        if self.results is None:
            print('No results to plot yet. Run a strategy.')

        title = 'Momentum Backtesting - %s ' % (self.symbol)
        self.results[self.toplot_p].plot(title=title, figsize=(10, 6)) #Percentage
        self.results[self.toplot_c].plot(title=title, figsize=(10, 6))  # Cash
        plt.show()


    def hist_returns(self):
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = 'Histogram Returns - Momentum Backtesting - %s ' % (self.symbol)
        self.results[self.toplot_hist].plot.hist(title=title, figsize=(10, 6), alpha=0.5, bins=30)
        # plt.hist(self.results['creturns_p'])
        plt.show()


    def plot_bstmom(self):

        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = 'All momentum Strategies Final Returns - %s ' % (self.symbol)

        # fig, ax1 = plt.subplots()
        # ax1.plot(self.x,self.y, 'b-', alpha = 0.5)
        # ax1.set_ylabel('Final Returns', color='b')
        # ax1.tick_params('y', colors='b')
        # ax2 = ax1.twinx()
        # ax2.plot(self.x,self.z, 'r--')
        # ax2.set_ylabel('Max.Drawdown', color='r')
        # ax2.tick_params('y', colors='r')
        # fig.tight_layout()
        # plt.legend()

        plt.plot(self.x, self.y, 'b-o', alpha = 0.5)
        plt.plot(self.x, self.z, 'r--', alpha = 0.5)
        plt.title(title)
        plt.legend(['Final Returns', 'Maximum Drawdown'])
        plt.xlabel('Momentums')
        plt.ylabel('Returns/MDD')
        plt.show()


if __name__ == '__main__':
    mombt = Momentum_Backtester('AAPL', '2015-12-8', '2016-12-10', lvrage=10)
    print(mombt.run_strategy(momentum=[x for x in range(20,200,20)]))
    #print(mombt.plot_strategy())
    #print(mombt.hist_returns())
    #print(mombt.plot_bstmom())

