# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:39:17 2020

@author: DELL
"""
import pandas as pd
import numpy as np
import Preprocessing_data as preda
import matplotlib.pyplot as plt
import seaborn as sns
# index for deflated data
index = preda.get_index()
# manipulate raw data, prepare data for main analysis
chemicals_raw = preda.get_raw_data('Chemicals_raw.xls')
chem =preda.clean_datechem(chemicals_raw)
old_chem, young_chem, newborn_chem = preda.get_f_wrt_age(chem)
newborn_chem = preda.get_data_newborn(newborn_chem)
'''dataframe of chemicals for the whole period'''
total_chem = pd.concat([old_chem,young_chem,newborn_chem], axis = 0, ignore_index = True)
'''drop some firms that have unreasonable entry, firms with multi-industry, and 
    data for the whole cooperation around the world'''
total_chem = total_chem.drop(total_chem.index[total_chem['ID'].isin([8,21,51,54,156])]).reset_index()
total_chem = total_chem.drop('index', axis = 1)
total_chem.columns = ['ID', 'name',
                      'fa_18', 'fa_17', 'fa_16', 'fa_15', 'fa_14', 'fa_13', 'fa_12', 'fa_11', 'fa_10',
                      'em_18', 'em_17', 'em_16', 'em_15', 'em_14','em_13', 'em_12', 'em_11', 'em_10',
                      'ec_18', 'ec_17', 'ec_16', 'ec_15', 'ec_14', 'ec_13', 'ec_12', 'ec_11', 'ec_10',
                      'mc_18', 'mc_17', 'mc_16','mc_15', 'mc_14', 'mc_13','mc_12', 'mc_11', 'mc_10',
                      's_18', 's_17', 's_16', 's_15', 's_14','s_13','s_12', 's_11', 's_10',
                      'year_cor']
total_chem, total_chem_edit = preda.deflate_data(total_chem, index)
total_chem_edit.to_csv('chem.csv', sep='|')
chem_malm = preda.Malmquist_data(total_chem_edit, 197)
chem_malm.loc[(chem_malm['year']==2012) & (chem_malm['firm'].isin([335])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
chem_malm.loc[(chem_malm['year']==2013) & (chem_malm['firm'].isin([151,299,335])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
chem_malm.loc[(chem_malm['year']==2014) & (chem_malm['firm'].isin([151,299])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
chem_malm.to_csv('chem_malm.csv', sep = '|')

###extract only sale data for weighted average of the result

sales_chem = total_chem_edit.loc[:, ['ID', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15',
                                     's_16', 's_17', 's_18']]
sales_chem.loc[sales_chem['ID'] == 335, ['s_12', 's_13']] =np.nan
sales_chem.loc[sales_chem['ID'].isin([151,299]), ['s_13', 's_14']] = np.nan
for i in range(10,19):
    sales_chem.loc[:,f's_{i}'] = sales_chem.loc[:,f's_{i}']/np.nansum(sales_chem.loc[:,f's_{i}'])
sales_chem.to_csv('sales_chem.csv', sep = '|')

# read result

eff_chem = pd.read_csv('efficiency_chemical.csv').drop('Unnamed: 0', axis =1)

eff_chem = pd.read_csv('eff_score_chem.csv').drop('Unnamed: 0', axis =1)

eff_chem.replace(0,np.nan, inplace=True)
summary = preda.efficiency_sum_stats(eff_chem)
summary_by_age = preda.eff_stats_by_age_merged(eff_chem, 140, 36, 21)

preda.eff_distribution_OT(eff_chem, summary)

preda.eff_distribution_OT_by_age(eff_chem, summary)

'''plt.subplots(figsize=(10,8))
sns.boxplot(x='Year', y='Eff_score', data= eff_melt.dropna(), linewidth=.9, color='steelblue')
sns.pointplot(x=eff_melt.iloc[0:9,1], y=summary['Mean'], scale=1, color='k', errwidth=1.5, capsize=0.2, markers='x', linestyles=' ')
sns.pointplot(x=eff_melt.iloc[0:9,1], y=summary['Mean'], scale=0.4, color='k', errwidth=0, capsize=0, linestyles='--')
plt.title('Efficiency level of chemical sector over time', fontsize=20)

plt.subplots(figsize=(10,8))
sns.boxplot(x='Year', y='Eff_score', data= eff_melt.dropna(), hue='Age', palette='GnBu_d')
plt.title('Efficiency level of chemical firms by age over time', fontsize=20)'''

#distribution of efficiency score
preda.hist_eff_score(eff_chem)
eff_chem_dmu = preda.eff_dmu(eff_chem)
preda.eff_static(eff_chem_dmu)
#change in efficiency level of efficiecnt newborns
plt.subplots(figsize =(12,12))
plt.ylim(0, 1.2)
plt.plot(eff_chem_dmu.loc[eff_chem_dmu['year'] >=2013, 'year'], 
         eff_chem_dmu.loc[eff_chem_dmu['firm_9'].notnull(),'firm_9'],
         color='skyblue', linewidth=3, marker='o', markerfacecolor='blue', markersize=10)
plt.plot(eff_chem_dmu.loc[eff_chem_dmu['year'] >=2015, 'year'], 
         eff_chem_dmu.loc[eff_chem_dmu['firm_221'].notnull(),'firm_221'],
         color='olive', linewidth=1, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.title('Efficient newborn firms in chemical sector', color='blue', fontsize=20)
plt.ylabel('efficiency score', color='blue', fontsize=14)


from time import strftime 
writer = pd.ExcelWriter(strftime('Report_chemicals_geo %Y-%m-%d.xlsx'))
eff_chem_dmu.to_excel(writer, 'eff_firms')
summary.to_excel(writer, 'eff_summary_stats')
summary_by_age.to_excel(writer, 'eff_score_by_age')
writer.save()
# Malmquist index
df = preda.read_malmquist('malmquist_chemical.csv')

df_chem = preda.read_malmquist('malmquist_tovrs_chem.csv')

df_compare = preda.encode_change(df, 140, 36, 22)
growth_dmu = preda.growth_dmu(df_compare, 198)
efficiency_growth = preda.ec_dmu(df_compare)
source_pc_machinery = preda.source_pc_sector(df, 140, 36, 22)
comparison = preda.comparison(df, 140, 36, 22)
source_pd_sector = preda.source_pd_sector(source_pc_machinery, comparison, 140, 36, 22)
avg_change_total = preda.average_change(df,'total')
avg_change_total.to_excel(writer, 'overall_avg_change')
writer.save()
preda.visualize_change_by_group(avg_change_total, 'total')
df_old = df.iloc[0:140,]
df_young = df.iloc[140:176,]
df_newborn = df.iloc[176:198,]
avg_change_by_age = preda.avg_change_bygroup(df_old, df_young, df_newborn)
avg_change_by_age.replace(1,np.nan,inplace=True)
preda.visualize_change_by_group(avg_change_by_age, 'newborn')
preda.visualize_change_by_group(avg_change_by_age, 'old')
preda.visualize_change_by_group(avg_change_by_age, 'young')

preda.visualize_change_by_component(avg_change_by_age,'MI')
preda.visualize_change_by_component(avg_change_by_age,'EC')
preda.visualize_change_by_component(avg_change_by_age,'TC')

data_viz = pd.read_excel('summary_eff_scores.xlsx')
data_viz.replace(0,np.nan, inplace=True)
preda.visualize_clustered_toward_eff(data_viz, 'among sectors')
preda.visualize_clustered_toward_eff(data_viz, 'chemical')
preda.visualize_clustered_toward_eff(data_viz, 'electronic')
preda.visualize_clustered_toward_eff(data_viz, 'machinery')
