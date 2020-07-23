# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:31:56 2020

@author: DELL
"""

import pandas as pd
import numpy as np
import Preprocessing_data as preda
import matplotlib.pyplot as plt
# index for deflated data
index = preda.get_index()

elec = preda.get_raw_data('Electronics_raw.xls')

elec = preda.clean_dateelec(elec)
old_elec, young_elec, newborn_elec = preda.get_f_wrt_age(elec)
newborn_elec = preda.get_data_newborn(newborn_elec)
'''dataframe of electronic for the whole period'''
total_elec = pd.concat([old_elec,young_elec,newborn_elec], axis =0, ignore_index = True)
'''drop some firms that have unreasonable entry, firms with multi-industry, and 
    data for the whole cooperation around the world'''
total_elec = total_elec.drop(total_elec.index[total_elec['ID'].isin([6,11,23,423,98,85])]).reset_index()
total_elec = total_elec.drop('index', axis = 1)
total_elec.columns = ['ID', 'name',
                      'fa_18', 'fa_17', 'fa_16', 'fa_15', 'fa_14', 'fa_13', 'fa_12', 'fa_11', 'fa_10',
                      'em_18', 'em_17', 'em_16', 'em_15', 'em_14','em_13', 'em_12', 'em_11', 'em_10',
                      'ec_18', 'ec_17', 'ec_16', 'ec_15', 'ec_14', 'ec_13', 'ec_12', 'ec_11', 'ec_10',
                      'mc_18', 'mc_17', 'mc_16','mc_15', 'mc_14', 'mc_13','mc_12', 'mc_11', 'mc_10',
                      's_18', 's_17', 's_16', 's_15', 's_14','s_13','s_12', 's_11', 's_10',
                      'year_cor']
total_elec, total_elec_edit = preda.deflate_data(total_elec,index)
total_elec_edit.to_csv('elec.csv', sep ='|')
elec_malm = preda.Malmquist_data(total_elec_edit,233)
elec_malm.loc[(elec_malm['year']==2010) & (elec_malm['firm'].isin([324])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
elec_malm.loc[(elec_malm['year']==2011) & (elec_malm['firm'].isin([1010])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
elec_malm.loc[(elec_malm['year']==2012) & (elec_malm['firm'].isin([1010])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
elec_malm.loc[(elec_malm['year']==2013) & (elec_malm['firm'].isin([38,65,1010])), ['fixed_assets','employees',
                                                                                 'emp_cost','mat_cost','sales']] = np.nan
elec_malm.loc[(elec_malm['year']==2014) & (elec_malm['firm'].isin([38,65])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan

"""for robustness check"""
elec_malm.loc[(elec_malm['year']==2010) & (elec_malm['firm'].isin([324])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
elec_malm.loc[(elec_malm['year']==2013) & (elec_malm['firm'].isin([38,65])), ['fixed_assets','employees',
                                                                                 'emp_cost','mat_cost','sales']] = np.nan
elec_malm.loc[(elec_malm['year']==2014) & (elec_malm['firm'].isin([38,65])), ['fixed_assets','employees',
                                                                                 'emp_cost','mat_cost','sales']] = np.nan

elec_malm.to_csv('elec_malm.csv', sep='|')

###extract only sale data for weighted average of the result

sales_elec = total_elec_edit.loc[:, ['ID', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15',
                                     's_16', 's_17', 's_18']]
sales_elec.loc[sales_elec['ID'] == 324, ['s_10']] = np.nan
sales_elec.loc[sales_elec['ID'] == 1010, ['s_11', 's_12', 's_13']] = np.nan
sales_elec.loc[sales_elec['ID'].isin([38, 65]), ['s_13', 's_14']] = np.nan
for i in range(10,19):
    sales_elec.loc[:,f's_{i}'] = sales_elec.loc[:,f's_{i}']/np.nansum(sales_elec.loc[:,f's_{i}'])
    
sales_elec.to_csv('sales_elec.csv', sep = '|')

#read results
#eff_elec = pd.read_csv('efficiency_electronic.csv').drop('Unnamed: 0', axis =1)
eff_elec = pd.read_csv('eff_score_elec.csv').drop('Unnamed: 0', axis =1)

eff_elec.replace(0, np.nan, inplace=True)
#distribution of efficiency score
preda.hist_eff_score(eff_elec)
eff_elec_dmu = preda.eff_dmu(eff_elec)
preda.eff_static(eff_elec_dmu,20,8,4, 'Electronic')
# plot the efficiency score change of efficient newborn firms
plt.subplots(figsize =(12,12))
plt.ylim(0, 1.2)
plt.plot(eff_elec_dmu.loc[eff_elec_dmu['year'] >=2012, 'year'], 
         eff_elec_dmu.loc[eff_elec_dmu['firm_54'].notnull(),'firm_54'],
         color='skyblue', linewidth=3, marker='o', markerfacecolor='blue', markersize=10)
plt.plot(eff_elec_dmu.loc[eff_elec_dmu['year'] >=2012, 'year'], 
         eff_elec_dmu.loc[eff_elec_dmu['firm_111'].notnull(),'firm_111'],
         color='olive', linewidth=1, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot(eff_elec_dmu.loc[eff_elec_dmu['year'] >=2011, 'year'], 
         eff_elec_dmu.loc[eff_elec_dmu['firm_533'].notnull(),'firm_533'],
         color='purple', linewidth=1, marker='o', markerfacecolor='violet', markersize=6)
plt.plot(eff_elec_dmu.loc[eff_elec_dmu['year'] >=2015, 'year'], 
         eff_elec_dmu.loc[eff_elec_dmu['firm_643'].notnull(),'firm_643'],
         color='darkred', linewidth=1, marker='o', markerfacecolor='red', markersize=6)
plt.title('Efficient newborn firms in electronic sector', color='blue', fontsize=20)
plt.ylabel('efficiency score', color='blue', fontsize=14)
#aggregate the result
summary = preda.efficiency_sum_stats(eff_elec)
summary_by_age = preda.eff_stats_by_age_merged(eff_elec, 175, 41, 17)
from time import strftime 
writer = pd.ExcelWriter(strftime('Report_electronics_geo %Y-%m-%d.xlsx'))
eff_elec_dmu.to_excel(writer, 'eff_firms')
summary.to_excel(writer, 'eff_summary_stats')
summary_by_age.to_excel(writer, 'eff_score_by_age')

# Malmquist index
df = preda.read_malmquist('malmquist_electronic.csv')

df_elec = preda.read_malmquist('malmquist_tovrs_elec.csv')

df_compare = preda.encode_change(df, 177, 42, 18)
growth_dmu = preda.growth_dmu(df_compare, 237)
efficiency_growth = preda.ec_dmu(df_compare)
# visualization
avg_change_total = preda.average_change(df,'total')
avg_change_total.to_excel(writer, 'overall_avg_change')
writer.save()
preda.visualize_change_by_group(avg_change_total, 'total')
df_old = df.iloc[0:177,]
df_young = df.iloc[177:219,]
df_newborn = df.iloc[219:237,]
avg_change_by_age = preda.avg_change_bygroup(df_old, df_young, df_newborn)
preda.visualize_change_by_group(avg_change_by_age, 'newborn')
preda.visualize_change_by_group(avg_change_by_age, 'old')
preda.visualize_change_by_group(avg_change_by_age, 'young')

preda.visualize_change_by_component(avg_change_by_age,'MI')
preda.visualize_change_by_component(avg_change_by_age,'EC')
preda.visualize_change_by_component(avg_change_by_age,'TC')

