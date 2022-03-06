import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import seaborn as sns

class create_figures:

    def __init__(self, model_name, threshold, hold_till):
        
        self.model = model_name
        self.threshold = threshold
        self.hold_till = hold_till
        
        results_dir = ## '\\results' folder directory
        self.folder_name = f'{str(self.model)}_{self.threshold}_{self.hold_till}'
        self.folder_dir = os.path.join(results_dir, self.folder_name)

        history_df_path = os.path.join(self.folder_dir, 'history_df.csv')
        self.history_df = pd.read_csv(history_df_path)
        self.history_df['buy_date'] = pd.to_datetime(self.history_df['buy_date'])
        self.history_df['sell_date'] = pd.to_datetime(self.history_df['sell_date'])
        
        params_path = os.path.join(self.folder_dir, 'params')
        with open(params_path, 'rb') as fp:
            self.params = pickle.load(fp)
        
        results_summary_path = os.path.join(self.folder_dir, 'results_summary')
        with open(results_summary_path, 'rb') as fp:
            self.results_summary = pickle.load(fp)
        
        #get params from stored files
        self.initial_capital = self.results_summary[0]
        self.total_gain = self.results_summary[1]
        self.start_date = self.params[4]
        self.end_date = self.params[5]
        
        self.create_history_df()
        self.create_history_figure()
        self.gain_loss_plot()
        self.hold_hist()

    def create_history_df(self):
        """
        this function creates a dataframe from start date to end date
        """
        cols = ['Date', 'Stock', 'Gain', 'Total']
        self.running_gains_df = pd.DataFrame(columns = cols)
        #create a row with the start date - 1 with the starting balance
        self.running_gains_df = self.running_gains_df.append({'Date': self.start_date - timedelta(days = 1), 'Stock': np.nan, 'Gain': \
            np.nan, 'Total': self.initial_capital}, ignore_index = True)
        values_total = self.initial_capital #start balance
        
        #create date range
        def date_range(start, end):
            delta = end - start  # as timedelta
            days = [start + timedelta(days=i) for i in range(delta.days + 1)]
            return days
        
        days = date_range(self.start_date, self.end_date)
        
        for day in days:
            #find if data is available. 1 if available 0 if not
            available = self.history_df[self.history_df['buy_date'] == day].values.shape[0]
            if available > 0:
                values = self.history_df.loc[self.history_df['buy_date'] == day].reset_index(drop = True)
                values_date = day
                values_stock = values['stock'][0]
                values_gain = values['net_gain'][0]
                values_total = values_total + values_gain
                #dict to append
                dict_ = {'Date': values_date, 'Stock': values_stock, 'Gain': values_gain, 'Total': values_total}
                self.running_gains_df = self.running_gains_df.append(dict_, ignore_index = True)
            else:
                dict_ = {'Date': day, 'Stock': np.nan, 'Gain': np.nan, 'Total': values_total}
                self.running_gains_df = self.running_gains_df.append(dict_, ignore_index = True)
        
        #save the running gains df
        self.running_gains_df.to_csv(f'{self.folder_dir}/running_gains_df.csv', index = False)

    def create_history_figure(self):
        """
        plot the running gains
        """
        plt.figure(figsize = (10, 6))
        plt.plot(self.running_gains_df['Date'], self.running_gains_df['Total'])
        plt.xlabel('Time (days)', fontsize = '6')
        plt.ylabel('Total Balance ($)')
        plt.xticks(rotation = 45)
        # plt.show()
        fig_path = os.path.join(self.folder_dir, 'total_balance_history.jpg')
        plt.savefig(fig_path)

    def gain_loss_plot(self):
        """
        a bar plot with gains and losses on the day
        """
        positives = self.running_gains_df[self.running_gains_df['Gain'] > 0]
        negatives = self.running_gains_df[self.running_gains_df['Gain'] < 0]
        plt.figure(figsize = (10, 6))
        plt.bar(positives['Date'], positives['Gain'], color = 'g', width = 1.5, label = 'wins')
        plt.bar(negatives['Date'], negatives['Gain'], color = 'r', width = 1.5, label = 'losses')
        plt.axhline(linewidth = 1, color = 'k')
        plt.legend()
        plt.xlabel('Time (days)', fontsize = '6')
        plt.ylabel('Gain ($)')
        plt.xticks(rotation = 45)
        # plt.show()
        fig_path = os.path.join(self.folder_dir, 'gain_loss.jpg')
        plt.savefig(fig_path)

    def hold_hist(self):
        """
        This function creates a histogram with buy times and color code it with winner or loser
        """
        #create columns
        df = self.history_df
        df['winner_bool'] = ['win' if x > 0 else 'loss' for x in df.net_gain]
        df['held_time'] = [(t2 - t1).days for t2, t1 in zip(df['sell_date'], df['buy_date'])]

        plt.figure(figsize = (10, 6))
        sns.histplot(x = 'held_time', data = df, hue = 'winner_bool', bins = 100)
        plt.xticks(rotation = 45)
        plt.xlabel('Held time (days)')
        # plt.show()
        fig_path = os.path.join(self.folder_dir, 'held_time_histogram.jpg')
        plt.savefig(fig_path)

if __name__ == '__main__':
    create_figures('LR_v1_predict', 1, 1)