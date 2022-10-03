import os

import pandas as pd
import matplotlib.pyplot as plt
from pm4py.objects.conversion.log import converter as log_converter
from discover.util import freedman_diaconis_rule
from fitter import Fitter, get_common_distributions

class Timing:

    dcr_to_rule_mapping = {'CONDITION':'DELAY','RESPONSE':'DEADLINE'}
    rule_to_dcr_mapping = {'DELAY':'CONDITION','DEADLINE':'RESPONSE'}

    def simple_distribution_fit_all_timings(self,timings,folder, xmax=1000):
        res = {}
        for (rule, e1, e2), data in timings.items():
            if len(data) > 5:  # too low statistics, needs at least 5 data points (or another limit).
                file_name = os.path.join(folder, f'{rule}-{e1}-{e2}_simple_fit.jpg')
                plot_title = f'{rule}, {e1} --> {e2}'
                res[(rule, e1, e2)] = self.simple_distribution_fitter(data, file_name, plot_title, None, xmax, save=True)
            else:
                print(f'[!] NOT Enough Data Points for fitting {rule}: {e1}-->{e2}')
                res[(rule, e1, e2)] = None
        return res

    def simple_distribution_fitter(self,data,filename,title,Nbins=None,xmax=None,save=True):
        '''
        :param data:
        :param filename:
        :param Nbins:
        :param xmax:
        :return: name of the best fitting distribution
        '''
        if not Nbins:
            Nbins, bin_width = freedman_diaconis_rule(data)
        f = Fitter(data, distributions=get_common_distributions(), xmax=xmax, timeout=2 * 60, bins=Nbins)
        f.fit()
        if save:
            fig, ax = plt.subplots(figsize=(16, 5))
            f.summary(plot=True)
            fig.tight_layout()
            ax.set_xlabel('Duration (Days)')
            ax.set_ylabel('Binned count')
            ax.set_title(title)
            plt.savefig(filename)
            plt.close()
        return f.get_best()


    def get_log_with_pair(self,event_log,e1,e2):
        first_e1 = event_log[event_log['concept:name']==e1].groupby('case:concept:name')[['case:concept:name','time:timestamp']].first().reset_index(drop=True)
        subset_is_in = first_e1.merge(event_log,on='case:concept:name',how='inner',suffixes=('_e1', ''))
        cids = subset_is_in[((subset_is_in['time:timestamp_e1']<subset_is_in['time:timestamp']) & (subset_is_in['concept:name']==e2))]['case:concept:name'].unique()
        return event_log[event_log['case:concept:name'].isin(cids)].copy(deep=True)

    def get_max_for_response(self,temp_df):
        '''
        This method is a way to find the max response (deadline).
        Within a trace it keeps track of the max delta between either (e1,e1) or (e1,e2) pairs.
        When it reaches a (e1,e2) pair it updates the delta on that row with the max delta found in preceeding pairs.
        This means that it will either take the current row delta because this is the only occurence of (e1,e2)
        or it will take a delta from a previous pair of (e1,e1) and assign it to that pair (e1,e2)
        :param temp_df: this is a dataframe with only the event pairs (e1 and e2) where at least 1 e1 preceeds an e2 for all traces
        :return:
        '''
        cids = temp_df['case:concept:name'].unique()
        for cid in cids:
            max_days = 0
            for index, row in temp_df[temp_df['case:concept:name'] == cid].iterrows():
                max_days = max(max_days, row['delta'])
                if row['concept:name'] != row['concept:name:to']:
                    temp_df.loc[index, 'delta'] = max_days
                    max_days = 0
        return temp_df

    def get_delta_between_events(self,filtered_df, event_pair, rule):

        filtered_df['time:timestamp'] = pd.to_datetime(filtered_df['time:timestamp'], utc=True)
        filtered_df = filtered_df[(filtered_df['concept:name']==event_pair[1]) |
                                   (filtered_df['concept:name']==event_pair[0])].sort_values(['case:concept:name','time:timestamp'])
        temp_df = pd.concat([filtered_df, filtered_df.groupby('case:concept:name').shift(-1)
                             .rename({'concept:name':'concept:name:to','time:timestamp':'time:timestamp:to'},axis=1)],axis=1)

        temp_df['delta'] = (temp_df['time:timestamp:to'] - temp_df['time:timestamp']).dt.days

        if rule=='RESPONSE':
            temp_df = self.get_max_for_response(temp_df)
        temp_df = temp_df[(temp_df['concept:name']==event_pair[0]) & (temp_df['concept:name:to']==event_pair[1])]
        data = temp_df['delta'].values
        return data

    def create_timing_input_dict(self,model):
        with open(model) as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]

        events = []
        conditions = []
        responses = []
        includes = []
        excludes = []
        for line in lines:
            temp = line.split(',')
            if temp[0] == 'EVENT':
                events.append(temp[1])
            elif temp[0] == 'CONDITION':
                conditions.append(temp[1:])
            elif temp[0] == 'RESPONSE':
                responses.append(temp[1:])
            elif temp[0] == 'INCLUDE':
                includes.append(temp[1:])
            elif temp[0] == 'EXCLUDE':
                excludes.append(temp[1:])

        timing_input_dict = {'CONDITION' : conditions,
                             'RESPONSE': responses}
        return timing_input_dict

    def get_timings(self,log,timing_input_dict):
        if isinstance(log,pd.DataFrame):
            event_log = log
        else:
            event_log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        res = {}

        total = len(timing_input_dict['CONDITION']) + len(timing_input_dict['RESPONSE'])
        i = 0
        for rule, event_pairs in timing_input_dict.items():
            for event_pair in event_pairs:
                filtered_df = self.get_log_with_pair(event_log,event_pair[0],event_pair[1]) #= pm4py.filter_between(log,event_pair[0],event_pair[1])
                data = self.get_delta_between_events(filtered_df,event_pair,rule)
                print(f'Done for {rule} {event_pair} {i}/{total}')
                res[(rule,event_pair[0],event_pair[1])] = data
                i = i + 1
        return res