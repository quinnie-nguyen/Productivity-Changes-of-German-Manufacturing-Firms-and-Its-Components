# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:51:36 2020

@author: DELL
"""


'''
################################## CHEMICALS #################################
chemicals_raw = preda.get_data_chem()
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
chem_malm = preda.Malmquist_data(total_chem_edit, 198)
chem_malm.loc[(chem_malm['year']==2012) & (chem_malm['firm'].isin([335])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
chem_malm.loc[(chem_malm['year']==2013) & (chem_malm['firm'].isin([151,299,335])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
chem_malm.loc[(chem_malm['year']==2014) & (chem_malm['firm'].isin([151,299])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
chem_malm.to_csv('chem_malm.csv', sep = '|')
# read results
eff_chem = pd.read_csv('efficiency_chemical.csv').drop('Unnamed: 0', axis =1)
#distribution of efficiency score
preda.hist_eff_score(eff_chem)
eff_chem_dmu = preda.eff_dmu(eff_chem)
#change in efficiency level of newborns
plt.subplots(figsize =(12,12))
plt.ylim(0, 1.2)
plt.plot(eff_chem_dmu.loc[eff_chem_dmu['year'] >=2013, 'year'], 
         eff_chem_dmu.loc[eff_chem_dmu['firm_9'] != 0,'firm_9'],
         color='skyblue', linewidth=3, marker='o', markerfacecolor='blue', markersize=10)
plt.plot(eff_chem_dmu.loc[eff_chem_dmu['year'] >=2015, 'year'], 
         eff_chem_dmu.loc[eff_chem_dmu['firm_221'] != 0,'firm_221'],
         color='olive', linewidth=1, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.title('Efficient newborn firms in chemical sector', color='blue', fontsize=20)
plt.ylabel('efficiency score', color='blue', fontsize=14)

################################## ELECTRONICS #################################

elec = preda.get_data_elec()
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
elec_malm = preda.Malmquist_data(total_elec_edit,237)
elec_malm.loc[(elec_malm['year']==2011) & (elec_malm['firm'].isin([1010])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
elec_malm.loc[(elec_malm['year']==2012) & (elec_malm['firm'].isin([1010])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
elec_malm.loc[(elec_malm['year']==2013) & (elec_malm['firm'].isin([38,65,1010])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
elec_malm.loc[(elec_malm['year']==2014) & (elec_malm['firm'].isin([38,65])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
elec_malm.to_csv('elec_malm.csv', sep='|')

eff_elec = pd.read_csv('efficiency_electronic.csv').drop('Unnamed: 0', axis =1)
preda.hist_eff_score(eff_elec)
eff_elec_dmu = preda.eff_dmu(eff_elec)
#change in efficiency level of newborns
plt.subplots(figsize =(12,12))
plt.ylim(0, 1.2)
plt.plot(eff_elec_dmu.loc[eff_elec_dmu['year'] >=2012, 'year'], 
         eff_elec_dmu.loc[eff_elec_dmu['firm_54'] != 0,'firm_54'],
         color='skyblue', linewidth=3, marker='o', markerfacecolor='blue', markersize=10)
plt.plot(eff_elec_dmu.loc[eff_elec_dmu['year'] >=2012, 'year'], 
         eff_elec_dmu.loc[eff_elec_dmu['firm_111'] != 0,'firm_111'],
         color='olive', linewidth=1, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot(eff_elec_dmu.loc[eff_elec_dmu['year'] >=2011, 'year'], 
         eff_elec_dmu.loc[eff_elec_dmu['firm_533'] != 0,'firm_533'],
         color='purple', linewidth=1, marker='o', markerfacecolor='violet', markersize=6)
plt.plot(eff_elec_dmu.loc[eff_elec_dmu['year'] >=2015, 'year'], 
         eff_elec_dmu.loc[eff_elec_dmu['firm_643'] != 0,'firm_643'],
         color='darkred', linewidth=1, marker='o', markerfacecolor='red', markersize=6)
plt.title('Efficient newborn firms in electronic sector', color='blue', fontsize=20)
plt.ylabel('efficiency score', color='blue', fontsize=14)
'''

################################## MACHINERY ##################################
import pandas as pd
import numpy as np
import Preprocessing_data as preda
import matplotlib.pyplot as plt
#import DEA
index = preda.get_index()
mac = preda.get_raw_data('Machinery_raw.xls')
mac = preda.clean_datemac(mac)
old_mac, young_mac, newborn_mac = preda.get_f_wrt_age(mac)
newborn_mac = preda.get_data_newborn(newborn_mac)
'''dataframe of machinery for the whole period'''
total_mac = pd.concat([old_mac,young_mac,newborn_mac], axis =0, ignore_index = True)
'''drop some firms that have unreasonable entry, firms with multi-industry, and 
    data for the whole cooperation around the world'''
total_mac = total_mac.drop(total_mac.index[total_mac['ID'].isin([271])]).reset_index()
total_mac = total_mac.drop('index', axis = 1)
total_mac.columns = ['ID', 'name',
                      'fa_18', 'fa_17', 'fa_16', 'fa_15', 'fa_14', 'fa_13', 'fa_12', 'fa_11', 'fa_10',
                      'em_18', 'em_17', 'em_16', 'em_15', 'em_14','em_13', 'em_12', 'em_11', 'em_10',
                      'ec_18', 'ec_17', 'ec_16', 'ec_15', 'ec_14', 'ec_13', 'ec_12', 'ec_11', 'ec_10',
                      'mc_18', 'mc_17', 'mc_16','mc_15', 'mc_14', 'mc_13','mc_12', 'mc_11', 'mc_10',
                      's_18', 's_17', 's_16', 's_15', 's_14','s_13','s_12', 's_11', 's_10',
                      'year_cor']
total_mac, total_mac_edit = preda.deflate_data(total_mac,index)
total_mac_edit.to_csv('mac.csv', sep='|')
mac_malm = preda.Malmquist_data(total_mac_edit, 305)
# if any input is null, then set others also null values
mac_malm.loc[(mac_malm['year']==2010) & (mac_malm['firm'].isin([547,716])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
mac_malm.loc[(mac_malm['year']==2011) & (mac_malm['firm'].isin([377,716])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
mac_malm.loc[(mac_malm['year']==2012) & (mac_malm['firm'].isin([377,274])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
mac_malm.loc[(mac_malm['year']==2013) & (mac_malm['firm'].isin([92,274])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
mac_malm.loc[(mac_malm['year']==2014) & (mac_malm['firm']==274), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan

"""for robustness check"""

mac_malm.loc[(mac_malm['year']==2010) & (mac_malm['firm'].isin([274,716])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
mac_malm.loc[(mac_malm['year']==2011) & (mac_malm['firm'].isin([274])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
mac_malm.loc[(mac_malm['year']==2012) & (mac_malm['firm'].isin([274])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
mac_malm.loc[(mac_malm['year']==2013) & (mac_malm['firm'].isin([274])), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan
mac_malm.loc[(mac_malm['year']==2014) & (mac_malm['firm']==274), ['fixed_assets','employees',
                                                          'emp_cost','mat_cost','sales']] = np.nan


mac_malm.to_csv('mac_malm.csv', sep='|')

###extract only sale data for weighted average of the result
sales_mac = total_mac_edit.loc[:, ['ID', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15',
                                     's_16', 's_17', 's_18']]
sales_mac.loc[sales_mac['ID'] == 547, ['s_10']] = np.nan
sales_mac.loc[sales_mac['ID'] == 716, ['s_10', 's_11']] = np.nan
sales_mac.loc[sales_mac['ID'] == 377, ['s_11', 's_12']] = np.nan
sales_mac.loc[sales_mac['ID'] == 92, ['s_13']] = np.nan
sales_mac.loc[sales_mac['ID'] == 274, ['s_12', 's_13', 's_14']] = np.nan
for i in range(10,19):
    sales_mac.loc[:,f's_{i}'] = sales_mac.loc[:,f's_{i}']/np.nansum(sales_mac.loc[:,f's_{i}'])

sales_mac.to_csv('sales_mac.csv', sep = '|')

# read result
#eff_mac = pd.read_csv('efficiency_machinery.csv').drop('Unnamed: 0', axis =1)
eff_mac = pd.read_csv('eff_score_mac.csv').drop('Unnamed: 0', axis =1)
eff_mac.replace(0,np.nan, inplace=True)
preda.hist_eff_score(eff_mac)
eff_mac_dmu = preda.eff_dmu(eff_mac)
preda.eff_static(eff_mac_dmu,50,15,6, 'Machinery')
#change in efficiency level of newborns
plt.subplots(figsize =(12,12))
plt.ylim(0, 1.2)
plt.plot(eff_mac_dmu.loc[eff_mac_dmu['year'] >=2012, 'year'], 
         eff_mac_dmu.loc[eff_mac_dmu['firm_67'] != 0,'firm_67'],
         color='skyblue', linewidth=1, marker='o', markerfacecolor='blue', markersize=6)
plt.plot(eff_mac_dmu.loc[eff_mac_dmu['year'] >=2013, 'year'], 
         eff_mac_dmu.loc[eff_mac_dmu['firm_95'] != 0,'firm_95'],
         color='olive', linewidth=1, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot(eff_mac_dmu.loc[eff_mac_dmu['year'] >=2014, 'year'], 
         eff_mac_dmu.loc[eff_mac_dmu['firm_292'] != 0,'firm_292'],
         color='purple', linewidth=1, marker='o', markerfacecolor='violet', markersize=6)
plt.plot(eff_mac_dmu.loc[eff_mac_dmu['year'] >=2013, 'year'], 
         eff_mac_dmu.loc[eff_mac_dmu['firm_381'] != 0,'firm_381'],
         color='darkred', linewidth=1, marker='o', markerfacecolor='red', markersize=6)
plt.plot(eff_mac_dmu.loc[eff_mac_dmu['year'] >=2014, 'year'], 
         eff_mac_dmu.loc[eff_mac_dmu['firm_554'] != 0,'firm_554'],
         color='black', linewidth=1, marker='o', markerfacecolor='silver', markersize=6)
plt.plot(eff_mac_dmu.loc[eff_mac_dmu['year'] >=2014, 'year'], 
         eff_mac_dmu.loc[eff_mac_dmu['firm_1859'] != 0,'firm_1859'],
         color='forestgreen', linewidth=1, marker='o', markerfacecolor='limegreen', markersize=6)
plt.title('Efficient newborn firms in machinery sector', color='blue', fontsize=20)
plt.ylabel('efficiency score', color='blue', fontsize=14)
summary=preda.efficiency_sum_stats(eff_mac)
summary_by_age = preda.eff_stats_by_age_merged(eff_mac, 241, 41, 23)
from time import strftime 
writer = pd.ExcelWriter(strftime('Report_machinery_geo %Y-%m-%d.xlsx'))
eff_mac_dmu.to_excel(writer, 'eff_firms')
summary.to_excel(writer, 'eff_summary_stats')
summary_by_age.to_excel(writer, 'eff_score_by_age')

#malmquist index
# how many firms grow over time? pc>1
df = preda.read_malmquist('malmquist_machinery.csv')

df_mac = preda.read_malmquist('malmquist_tovrs_mac.csv')

df_compare = preda.encode_change(df, 241, 41, 23)
growth_dmu = preda.growth_dmu(df_compare, 307)
efficiency_growth = preda.ec_dmu(df_compare)
source_pc_machinery = preda.source_pc_sector(df, 242, 42, 23)
comparison = preda.comparison(df, 242, 42, 23)
source_pd_sector = preda.source_pd_sector(source_pc_machinery, comparison, 242, 42, 23)
avg_change_total = preda.average_change(df,'total')
avg_change_total.to_excel(writer, 'overall_avg_change')
writer.save()
preda.visualize_change_by_group(avg_change_total, 'total')
df_old = df.iloc[0:242,]
df_young = df.iloc[242:284,]
df_newborn = df.iloc[284:307,]
avg_change_by_age = preda.avg_change_bygroup(df_old, df_young, df_newborn)

preda.visualize_change_by_group(avg_change_by_age, 'newborn')
preda.visualize_change_by_group(avg_change_by_age, 'old')
preda.visualize_change_by_group(avg_change_by_age, 'young')

preda.visualize_change_by_component(avg_change_by_age,'MI')
preda.visualize_change_by_component(avg_change_by_age,'EC')
preda.visualize_change_by_component(avg_change_by_age,'TC')
