# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:37:21 2020

@author: Quinn
@ver1. 20.04
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Preprocessing_data as preda
chemicals_raw=preda.get_data_chem()
chemicals_raw.head()


chemicals_raw['year_cor'].isnull().sum()
chemicals_raw['year_cor'].unique()
chemicals_raw['year_cor']=chemicals_raw['year_cor'].replace('NaT', np.nan)
chemicals_raw[chemicals_raw['year_cor'].isnull()]['company']
'''ULTRABIO WERKE GMBH 2010
   MORSA WACHSWARENFABRIK SALLINGER GMBH 
   RARO PLASTICS GMBH null
   ADVANCE PHARMA GMBH 
   1320 duplicate vs 1319
   1527 ATEKU GMBH & CO. KG 2009
   1948 null value
   2242 null
   2413 bỏ
   2724 bỏ
   bỏ
   '''
chemicals_raw.info()
chemicals_raw[chemicals_raw['date_cor'].isnull()]['year_cor']
date_na_index = [695, 828, 973, 1297, 1320, 1527, 1948, 2242, 2413, 
                 2724, 2745, 2797, 3316, 3702, 3998, 4374, 4809, 4920, 4935, 5239]
for i in date_na_index:
    print(chemicals_raw['company'][i])
    
    
chem =preda.clean_date(chemicals_raw)
chem.info()
chem[chem['year_cor']=='NaT']['company']
chem['year_cor'].replace('NaT', '2010', inplace = True)
chem['year_cor'].unique()
old_chem, new_chem, current_chem = preda.get_chem_age(chem)
old_chem.shape
new_chem.shape
current_chem.shape
new_chem['year_cor'].unique()
old_chem.dropna().shape
new_chem.dropna().shape
new_chem.isnull().sum()
current_chem.dropna().shape
current_chem.info()
current_chem_2018 = current_chem[['ID','company','fixed_asset_2018',
                                      'employee_2018','material_cost_2018','sale_2018','year_cor']]
current_chem_2018 = current_chem_2018.dropna()
current_chem_2018.head()
current_chem_2018['year_cor'].unique()
r_2018, r_2017, r_2016, r_2015, r_2014, r_2013, r_2012, r_2011, r_2010 = preda.current_entry(current_chem)
def get_index(r):
    r = r.dropna()
    arr = r['ID']
    return r, arr
r_18, id_18 = get_index(r_2018)
r_17, id_17 = get_index(r_2017)
r_16, id_16 = get_index(r_2016)
r_15, id_15 = get_index(r_2015)
r_14, id_14 = get_index(r_2014)
r_13, id_13 = get_index(r_2013)
r_12, id_12 = get_index(r_2012)
r_11, id_11 = get_index(r_2011)
r_10, id_10 = get_index(r_2010)
a = set(id_18) & set(id_15) & set(id_17) & set(id_16)
# 23 firms have info for 2015-2018
# 5 firms have info from 2011-2018
a
r_18.head()
r_17.head()
b = set(id_14) & set(id_13) & set(id_12) & set(id_11)
b
'''205 1903
476 1922
'''
elec = preda.get_data_elec()
elec[elec['year_cor']=='NaT'].index
elec = pd.read_excel('Electronics_raw.xls')
elec.info()
elec = preda.clean_dateelec(elec)
old_elec, new_elec, current_elec = preda.get_year_elec(elec)
old_elec = old_elec.dropna()
old_elec.info()
current_elec = preda.get_current(current_elec)
current_elec.dropna().index

e_2018, e_2017, e_2016, e_2015, e_2014, e_2013, e_2012, e_2011, e_2010 = preda.current_entry(current_elec)
e_2018.dropna().shape
e_2017.dropna().shape
e_2016.dropna().shape
e_2015.dropna().shape
e_2014.dropna().shape

e_2018, id_2018=preda.get_index(e_2018)
e_2017, id_2017=preda.get_index(e_2017)
e_2016, id_2016=preda.get_index(e_2016)
e_2015, id_2015=preda.get_index(e_2015)
e_2014, id_2014=preda.get_index(e_2014)
e_2013, id_2013=preda.get_index(e_2013)
e_2012, id_2012=preda.get_index(e_2012)
e_2011, id_2011=preda.get_index(e_2011)
e_2010, id_2010=preda.get_index(e_2010)
set(id_2018) & set(id_2017) & set(id_2016) & set(id_2015)
current_elec.dropna()
new_elec = new_elec.dropna()
new_elec.shape
mac = preda.get_data_mac()
mac[mac['year_cor']=='NaT'].index
mac = preda.clean_datemac(mac)
old_mac, new_mac, current_mac = preda.get_year_elec(mac)
old_mac = old_mac.dropna()
old_mac.shape
new_mac = new_mac.dropna()
new_mac.shape
current_mac = preda.get_current(current_mac)
current_mac = current_mac.dropna()
current_mac.shape
'''28.04.2020
deflate the output and correct the input'''
np.zeros((3, 1), dtype=np.float)
ca_use = pd.read_csv('Capacity_Utilization.csv')
ca_use = ca_use.iloc[29:37,0:2]
ca_use.columns = ['Year', 'CU']
ca_use = ca_use.reset_index(drop=True)
ppi = pd.read_csv('Producer_Price_index.csv')
ppi['DATE'] = [2010,2011,2012,2013,2014,2015,2016,2017,2018]
ppi.info()
index =preda.get_index()
old_chem.info()
new_chem.info()
old_new = pd.concat([old_chem, new_chem], axis =0,ignore_index=True)
current_chem
total = pd.concat([old_new, current_chem], axis =0,ignore_index=True)
inout_2018 = total.iloc[:,[0,2,11,29,38]].dropna()
import pyDEA
import pulp
avg = inout_2018.mean()
avg
inout_2018.iloc[:,1:5] = inout_2018.iloc[:,1:5]/avg
inout_2018.mean()
inout_2018.iloc[:,1] = inout_2018.iloc[:,1] * index.iloc[8,2]/100
inout_2018.iloc[:,2] = inout_2018.iloc[:,2] * index.iloc[8,3]
inout_2018.iloc[:,3:5] = inout_2018.iloc[:,3:5] / index.iloc[8,1] * 100
inout_2018.to_csv('data_18chem.csv', sep='|')

def deflate_data(df, index):
    '''
    fixed_asset*capacity use/100
    employee * effective hour
    material cost and sale /ppi * 100   
    '''
    for i in range(2,11):
        df.iloc[:,i] = df.iloc[:,i] * index.iloc[10-i,2]/100
    for i in range(11, 20):
        df.iloc[:,i] = df.iloc[:,i] * index.iloc[19-i,3]
    for i in range(20, 29):
        df.iloc[:,i] = df.iloc[:,i] / index.iloc[28-i,1]*100
    for i in range(29,38):
        df.iloc[:,i] = df.iloc[:,i] / index.iloc[37-i,1]*100
    for i in range(38,47):
        df.iloc[:,i] = df.iloc[:,i] / index.iloc[46-i,1]*100
    return df

eff_elec = pd.read_csv('efficiency_electronic.csv').drop('Unnamed: 0', axis =1)
#eff_elec = pd.DataFrame(data = eff_elec.T)
#eff_elec.columns = str(eff_elec.iloc[0,:])
sns.distplot(eff_elec.loc[eff_elec['eff_scores_10'] != 0, ['eff_scores_10']], bins=50)

#plot the hist of eff_scores over time
    
'''126,133,149,213,219,259,382,590,33,263,486,54,111,533,643'''
eff_elec_dmu_id = eff_elec.loc[(eff_elec['eff_scores_10'] == 1) | (eff_elec['eff_scores_11'] == 1) |
             (eff_elec['eff_scores_12'] == 1) | (eff_elec['eff_scores_13'] == 1) |
             (eff_elec['eff_scores_14'] == 1) | (eff_elec['eff_scores_15'] == 1) |
             (eff_elec['eff_scores_16'] == 1) | (eff_elec['eff_scores_17'] == 1) |
             (eff_elec['eff_scores_18'] == 1), 'firm_id']
eff_elec_dmu = eff_elec.loc[eff_elec['firm_id'].isin(eff_elec_dmu_id),].reset_index()
eff_elec_dmu = pd.DataFrame(eff_elec_dmu.T)
eff_elec_dmu.columns = ['firm_' + str(x) for x in eff_elec_dmu_id]
years = list(range(2008,2019,1))
eff_elec_dmu['year'] = years
eff_elec_dmu.index = years
eff_elec_dmu=eff_elec_dmu.iloc[2:11,]

'''plt.subplots(figsize =(12,12))
plt.plot('year', 'firm_126', data=eff_elec_dmu, color='skyblue', linewidth=2, marker='o', markerfacecolor='blue', markersize=6)
plt.plot('year', 'firm_133', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_149', data=eff_elec_dmu, color='brown', linewidth=2, marker='o', markerfacecolor='lightcoral', markersize=6)
plt.plot('year', 'firm_213', data=eff_elec_dmu, color='darkred', linewidth=2, marker='o', markerfacecolor='red', markersize=6)
plt.plot('year', 'firm_219', data=eff_elec_dmu, color='salmon', linewidth=2, marker='o', markerfacecolor='tomato', markersize=6)
plt.plot('year', 'firm_259', data=eff_elec_dmu, color='forestgreen', linewidth=2, marker='o', markerfacecolor='limegreen', markersize=6)
plt.plot('year', 'firm_382', data=eff_elec_dmu, color='slategrey', linewidth=2, marker='o', markerfacecolor='cadetblue', markersize=6)
plt.plot('year', 'firm_590', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_31', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_33', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_129', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_263', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_486', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_54', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_111', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_533', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.plot('year', 'firm_643', data=eff_elec_dmu, color='olive', linewidth=2, marker='o', markerfacecolor='yellowgreen', markersize=6)
plt.legend()'''
ncols = 3
nrows = int(eff_elec.shape[1]/ncols)
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(10,10))
counter = 1
for i in range(nrows):
    for j in range(ncols):
        ax = axes[i][j]
        if counter < len(eff_elec.columns)+1:
            ax.hist(eff_elec.loc[eff_elec.iloc[:,counter] != 0, eff_elec.columns[counter]], bins=20, facecolor='slategray',
                        alpha=1,rwidth=0.85, label='{}'.format(eff_elec.columns[counter]))
                #ax.set_xlabel('efficiency score')
                #ax.set_ylabel('number of firms')
            ax.set_ylim([0,60])
            leg=ax.legend(loc='upper right')
            leg.draw_frame(False)
        else:
            ax.set_axis_off()
        counter+=1
plt.plot()
eff_mac = pd.read_csv('efficiency_machinery.csv').drop('Unnamed: 0', axis =1)
preda.hist_eff_score(eff_mac)
eff_mac_dmu = preda.eff_dmu(eff_mac)

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


eff_elec_dmu = preda.eff_dmu(eff_elec)


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

eff_chem = pd.read_csv('efficiency_chemical.csv').drop('Unnamed: 0', axis =1)
preda.hist_eff_score(eff_chem)
eff_chem_dmu = preda.eff_dmu(eff_chem)
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

# robustness check

eff_chem_robust = pd.read_csv('efficiency_chemical_robustcheck.csv').drop('Unnamed: 0', axis = 1)
preda.hist_eff_score(eff_chem_robust)
eff_chem_dmu_robust = preda.eff_dmu(eff_chem_robust)

eff_elec_robust = pd.read_csv('efficiency_electronic_robustcheck.csv').drop('Unnamed: 0', axis = 1)
preda.hist_eff_score(eff_elec_robust)
eff_chem_dmu_robust = preda.eff_dmu(eff_elec_robust)

eff_chem_robust = pd.read_csv('efficiency_machinery_robustcheck.csv').drop('Unnamed: 0', axis = 1)
preda.hist_eff_score(eff_chem_robust)
eff_chem_dmu_robust = preda.eff_dmu(eff_chem_robust)
# malmquist index
df = pd.read_csv('malmquist_chemical.csv').drop('Unnamed: 0', axis=1)
df.replace(np.inf, np.nan, inplace=True)
df_compare = pd.DataFrame().reindex_like(df)
for i in df.columns:
    if i == 'dmu':
        df_compare[i] = df[i]
    else:
        df_compare[i] = np.where(df[i] > 1, 1, 0)
age = pd.concat([pd.Series(np.repeat('old', 140)), pd.Series(np.repeat('young', 36)), pd.Series(np.repeat('newborn', 22))])
df_compare['age'] = list(age)

def growth_dmu(df_compare):
    growth = 0
    dmu = []
    for i in range(0, 198):
        if all(df_compare.iloc[i,[1,4,7,10,13,16,19,22]] ==1):
            growth += 1
            dmu.append(i)
    return growth, dmu

growth_dmu(df_compare)

def ec_dmu(df_compare):
    efficiency_growth = 0
    dmu = []
    for i in range(0, 198):
        if all(df_compare.iloc[i,[2,5,8,11,14,17,20]] ==1):
            efficiency_growth += 1
            dmu.append(i)
    return efficiency_growth, dmu

ec_dmu(df_compare)
            
df_compare.sum()
df_compare = preda.encode_change(df)
growth_dmu(df_compare)
ec_dmu(df_compare)
ec_over_tc_old = 0
for i in range(0,140):
    if df.loc[i,'pc_11'] > 1 and df.loc[i,'ec_11'] > df.loc[i,'tc_11']:
        ec_over_tc_old += 1
ec_over_tc_young = 0
for i in range(140, 176):
    if df.loc[i,'pc_11'] > 1 and df.loc[i,'ec_11'] > df.loc[i,'tc_11']:
        ec_over_tc_young += 1        

ec_over_tc_young

        




df.loc[1,'pc_11'] > 1



r_11 = preda.source_pc(df, 140, 36, 22, df_compare,11)
r_12 = preda.source_pc(df, 140, 36, 22, df_compare,12)
r_13 = preda.source_pc(df, 140, 36, 22, df_compare,13)
r_14 = preda.source_pc(df, 140, 36, 22, df_compare,14)
r_15 = preda.source_pc(df, 140, 36, 22, df_compare,15)
r_16 = preda.source_pc(df, 140, 36, 22, df_compare,16)
r_17 = preda.source_pc(df, 140, 36, 22, df_compare,17)
r_18 = preda.source_pc(df, 140, 36, 22, df_compare,18)
r_chem = pd.DataFrame(data = [r_11, r_12, r_13, r_14, r_15, r_16, r_17, r_18])

r_chem.index = list(range(2011,2019))


r= preda.source_pc_sector(df, 140, 36, 22, df_compare)



plt.subplots(figsize =(8,8))
plt.ylim(0, 1.2)
plt.plot(r.index, r.loc[:,'%ec_driving_old'], color='skyblue', linewidth=1, 
         marker='o', markerfacecolor='blue', markersize=6, label='old_dmu')
plt.plot(r.index, r.loc[:,'%ec_driving_young'], color='olive', linewidth=1, 
         marker='o', markerfacecolor='yellowgreen', markersize=6, label='young_dmu')
plt.plot(r.index, r.loc[:,'%ec_driving_newborn'], color='purple', linewidth=1, 
         marker='o', markerfacecolor='violet', markersize=6, label='newborn_dmu')
plt.legend(loc='upper right')
plt.title('The contribution of efficicency change on productivity growth of chemical sector',
          color='blue', fontsize=15)

preda.effciency_change(df, 140, 36, 22,13)

preda.technical_change(df, 140, 36, 22, df_compare, 11)
a=df.iloc[140:176, 22:25]
a=a.loc[df['pc_18']>1]


preda.source_pc(df, 140, 36, 22, 15)
r=preda.source_pc_sector(df, 140, 36, 22)
preda.technical_change(df, 140, 36, 22, 17)

preda.effciency_change(df, 140, 36, 22,18)


x = preda.comparison(df, 140, 36, 22)


data_viz = pd.read_excel('summarize_report.xlsx')
data_viz.replace(0,np.nan, inplace=True)
preda.visualize_clustered_toward_eff(data_viz, 'among sectors')
preda.visualize_clustered_toward_eff(data_viz, 'chemical')
preda.visualize_clustered_toward_eff(data_viz, 'electronic')
preda.visualize_clustered_toward_eff(data_viz, 'machinery')

firm_id = eff_chem1.iloc[0,:]

arr =[]
for id in firm_id:
    arr.append(np.repeat(id, 9))

yrs =[[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],]
arr_yrs = yrs*19


def rearrange_eff_data(eff_chem):
    eff_melt = pd.melt(eff_chem.T.iloc[1:,])
    id_list = eff_chem.iloc[:,0]
    arr_id = []
    for i in id_list:
        arr_id.append(np.repeat(i, 9))
    yrs =[[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],]
    arr_yrs = yrs * 198
    arr_id = pd.melt(pd.DataFrame(data= arr_id).T)
    arr_yrs = pd.melt(pd.DataFrame(data = arr_yrs).T)
    eff_melt = pd.concat([arr_id['value'], arr_yrs['value'], eff_melt['value']], axis = 1)
    eff_melt.columns = ['ID', 'Year', 'Eff_score']
    arr_age = [['matured',]*1260, ['intermediate',]*324, ['young',]*198]
    arr_age = pd.melt(pd.DataFrame(data=arr_age).T)
    arr_age.replace('None', np.nan, inplace=True)
    arr_age.dropna(inplace=True)
    arr_age= arr_age.reset_index()
    eff_melt['Age'] = arr_age['value']
    return eff_melt

eff_melt = rearrange_eff_data(eff_chem)

eff_melt = pd.melt(eff_chem.T.iloc[1:,])

id_list = eff_chem.iloc[:,0]
arr_id = []
for i in id_list:
   arr_id.append(np.repeat(i, 9))
yrs =[[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],]
arr_yrs = yrs * 198
arr_id = pd.melt(pd.DataFrame(data= arr_id).T)
arr_id = pd.melt(arr_id.T)
arr_yrs = pd.DataFrame(data = arr_yrs.T)

eff_melt['ID'] = arr_id['value']

sns.boxplot(x='Year', y='Eff_score', data=eff_melt)
age = ['matured', 'intermediate', 'young']
arr_age = [['matured',]*1260, ['intermediate',]*324, ['young',]*198]
arr_age = pd.melt(pd.DataFrame(data=arr_age).T)
arr_age.replace('None', np.nan, inplace=True)
arr_age.dropna(inplace=True)
arr_age= arr_age.reset_index()




plt.subplots(figsize=(10,8))
sns.boxplot(x='Year', y='Eff_score', data= eff_melt.dropna(), linewidth=.9, color='c')
sns.pointplot(x=eff_melt.iloc[0:9,1], y=summary['Mean'], scale=1, color='k', errwidth=1.5, capsize=0.2, markers='x', linestyles=' ')
sns.pointplot(x=eff_melt.iloc[0:9,1], y=summary['Mean'], scale=0.4, color='k', errwidth=0, capsize=0, linestyles='--')
plt.title('Efficiency level of chemical sector over time', fontsize=20)



x = np.arange(1, eff_chem_dmu.shape[1], 1)
geo_mean=[]
for i in range(eff_chem_dmu.shape[1]-1):
    geo_mean.append(np.nanprod(eff_chem_dmu.iloc[:,i])**(1/eff_chem_dmu.iloc[:,i].notnull().sum()))
geo_mean = pd.Series(data=geo_mean)
table= pd.DataFrame(pd.concat([pd.Series(x), pd.Series(geo_mean)], axis = 1))
table.columns=['x', 'geo_mean']
table['inefficient'] = (1- table['geo_mean'])*5
table['y'] = table['x'] - table['inefficient']
eff = []
for i in range(table.shape[0]):
    if table.loc[i,'geo_mean']==1:
        eff.append(1)
    else:
        eff.append(0)
table['Efficient or not'] = eff
firm_id = eff_chem_dmu.columns
table['firm_id'] = firm_id[:-1]
fig, ax = plt.subplots(figsize=(10,10))
plt.axvspan(-1, 34.5, facecolor='g', alpha=0.5)
plt.axvspan(34.5, 41.5,facecolor='g', alpha=0.2 )
plt.axvspan(41.5, 48, facecolor='g', alpha=0.1)
ax=sns.pointplot(x='x', y='y', data= table, hue='Efficient or not', scale=1, errwidth=1.5, capsize=0.2, markers=['x','o'], linestyles=' ', palette='dark')
ax=sns.pointplot(x='x', y='x', data =table, scale=0.2, color='k', errwidth=0, capsize=0, linestyles='-')
plt.axis('off')



'''T TEST ON TFPC, EFFCH, TECHCH'''



from scipy import stats
np.random.seed(7654567)  # fix seed to get the same result
rvs = stats.norm.rvs(loc=5, scale=10, size=(50,1))
stats.ttest_1samp(rvs,5.0)
stats.ttest_1samp(rvs,1.0)

pc11 = df.iloc[:140,10]-1

stats.ttest_1samp(pc11,0)

df1 = df-1


t_test = []
p_value = []

stats.ttest_1samp(df1.loc[:140, 'tc_18'],0)
stats.ttest_1samp(df1.loc[140: 175, 'pc_12'],0)
stats.ttest_1samp(df1.loc[140: 175, 'pc_13'],0)
stats.ttest_1samp(df1.loc[140: 176, 'pc_14'],0)
stats.ttest_1samp(df1.loc[140: 176, 'pc_15'],0)
stats.ttest_1samp(df1.loc[140: 176, 'pc_16'],0)
stats.ttest_1samp(df1.loc[140: 176, 'pc_17'],0)
stats.ttest_1samp(df1.loc[140: 176, 'pc_18'],0)

stats.ttest_1samp(df1.loc[140: 176, 'pc_19'],0)

stats.ttest_1samp(df1.loc[176:198, 'ec_17'],0)





df_matured, df_intermediate, df_young = preda.malmquist_by_age(df, 140, 36, 22)

t_test = []
p_value = []
for i in range(1, df_matured.shape[1]):
    t_test.append(stats.ttest_1samp(df_matured.iloc[:,i], 1, nan_policy='omit')[0])
    p_value.append(stats.ttest_1samp(df_matured.iloc[:,i], 1, nan_policy='omit')[1])
    


'''THE EFFECT OF DISTANCE TO FRONTIER ON TFPCH, EFFCH, TECHCH'''

eff_chem['rank_10'] = eff_chem['eff_scores_17'].rank(ascending= False)

a=eff_chem['rank_10'].median()

near_frontier = eff_chem.loc[eff_chem['rank_10'] <= a,['firm_id','eff_scores_17', 'rank_10']]
further_frontier = eff_chem.loc[eff_chem['rank_10'] > a,['firm_id','eff_scores_17', 'rank_10']]

firm_near_frontier = near_frontier['firm_id']
firm_further_frontier = further_frontier['firm_id']

malm_11_near = df.loc[df['dmu'].isin(firm_near_frontier), ['pc_18', 'tc_18', 'ec_18']]
malm_11_further = df.loc[df['dmu'].isin(firm_further_frontier), ['pc_18', 'tc_18', 'ec_18']]

malm_11_near.mean(axis=0)
malm_11_further.mean(axis=0)

from scipy.stats.mstats import gmean

def malm_vs_efflevel(eff_chem, df):
    result_near = []
    result_further = []
    for i in range(10, 18):
        j=i+1
        eff_chem[f'rank_{i}'] = eff_chem[f'eff_scores_{i}'].rank(ascending=False)
        firm_near = eff_chem.loc[eff_chem[f'rank_{i}'] <= eff_chem[f'rank_{i}'].median(), 'firm_id']
        firm_further = eff_chem.loc[eff_chem[f'rank_{i}'] > eff_chem[f'rank_{i}'].median(), 'firm_id']
        malm_near = df.loc[df['dmu'].isin(firm_near), [f'pc_{j}', f'tc_{j}', f'ec_{j}']]
        result_near.append(gmean(malm_near, axis = 0))
        malm_further = df.loc[df['dmu'].isin(firm_further), [f'pc_{j}', f'tc_{j}', f'ec_{j}']]
        result_further.append(gmean(malm_further, axis = 0))
    result_near = pd.DataFrame(data=result_near)
    result_near.columns = ['tfpch_near', 'techch_near', 'effch_near']
    result_further = pd.DataFrame(data=result_further)
    result_further.columns = ['tfpch_further', 'techch_further', 'effch_further']
    result = pd.concat([result_near, result_further], axis = 1)
    result['year'] = [2011,2012,2013,2014,2015,2016,2017,2018]
    result = result.loc[:,['year','tfpch_near', 'tfpch_further', 'techch_near','techch_further', 'effch_near', 'effch_further']]
    return result


result = malm_vs_efflevel(eff_chem, df)

result['tfpch_near'] < result['tfpch_further']
result['techch_near'] > result['techch_further']
result['effch_near'] < result['effch_further']

pd.melt(result, ['year'])
plt.subplots(figsize=(8,6))
sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(result.iloc[:,[0,5,6]], ['year']), marker='o')
plt.title('Average efficiency scores of chemical firms by ages', color='black')

'''independent samples t-test to confirm the statistical significance of the effect of the distance from
frontier on TFPCH, EFFCH, and TECHCH'''

from scipy import stats

malm_near, malm_further, result = preda.malm_by_efflevel(eff_chem, df)

malm_near, malm_further = preda.malm_by_efflevel(eff_chem, df)

p7_11_n = malm_near.loc[:,['pc_15', 'tc_15', 'ec_15']].dropna()
p7_11_f = malm_further.loc[:,['pc_15', 'tc_15', 'ec_15']].dropna()

stats.ttest_ind(p7_11_n.iloc[:,1], p7_11_f.iloc[:,1], equal_var=False)

from scipy import stats
def independent_ttest_malm(eff_chem, df):
    result = []
    malm_near, malm_further, r = preda.malm_by_efflevel(eff_chem, df)
    for i in range(0,24):
        result.append(stats.ttest_ind(malm_further.iloc[:,i].dropna(),malm_near.iloc[:,i].dropna(),equal_var=False))
    result = pd.DataFrame(data=result)
    result['pvalue_one_tail'] = result['pvalue']/2
    result=result.T
    result.columns = ['pc_11', 'tc_11','ec_11',
                      'pc_12', 'tc_12','ec_12',
                      'pc_13', 'tc_13','ec_13',
                      'pc_14', 'tc_14','ec_14',
                      'pc_15', 'tc_15','ec_15',
                      'pc_16', 'tc_16','ec_16',
                      'pc_17', 'tc_17','ec_17',
                      'pc_18', 'tc_18','ec_18']
    result=result.round(3)
    return result


ttest= independent_ttest_malm(eff_chem, df)
ttest.iloc[2,:] <= 0.05

preda.eff_static(eff_chem_dmu)

'''%ec_tfp = ec(ec>tc)/pc(pc>1)
%tc_tfp = tc(tc>ec)/pc(pc>1)'''

source_pc = preda.source_pc_sector(df, 140, 36, 22)

t_test_r = preda.independent_ttest_malm(eff_chem, df)

t_test = preda.independent_ttest_malm(eff_chem, df).iloc[0,:].T

ax=plt.subplots(figsize=(10,8))
ax= t_test.plot(kind='bar', color='blue')
ax.axhline(1.645, linestyle='--', color='grey', linewidth=2)
ax.axhline(0, color='black', linewidth=2)
ax.axhline(-1.645, linestyle='--', color='grey', linewidth=2)
ax.axvline(7.5, linewidth=0.5)
ax.axvline(15.5, linewidth=0.5)
plt.xlabel('Malmquist Indices')
plt.ylabel('T-Statistic')
plt.title('Malmquist indices differences for Firms that is near and further the frontier')
plt.show()


malm_n, malm_f, result = preda.malm_by_efflevel(eff_chem, df)


preda.visualization_ttest(eff_chem, df)


def visualization_ttest(eff, df):
    malm_near, malm_further, r = preda.malm_by_efflevel(eff_chem, df)
    pc_n = pd.melt(malm_n.loc[:,['pc_11', 'pc_12', 'pc_13', 'pc_14', 'pc_15',
                                 'pc_16', 'pc_17', 'pc_18']])
    pc_n=pc_n.iloc[:,1].dropna()
    
    pc_f = pd.melt(malm_f.loc[:,['pc_11', 'pc_12', 'pc_13', 'pc_14', 'pc_15',
                                 'pc_16', 'pc_17', 'pc_18']])
    pc_f=pc_f.iloc[:,1].dropna()
    
    pc_ttest = stats.ttest_ind(pc_n, pc_f)[0]




    tc_n = pd.melt(malm_n.loc[:,['tc_11', 'tc_12', 'tc_13', 'tc_14', 'tc_15',
                                 'tc_16', 'tc_17', 'tc_18']])
    tc_n=tc_n.iloc[:,1].dropna()
    
    tc_f = pd.melt(malm_f.loc[:,['tc_11', 'tc_12', 'tc_13', 'tc_14', 'tc_15',
                                 'tc_16', 'tc_17', 'tc_18']])
    tc_f=tc_f.iloc[:,1].dropna()
    
    tc_ttest = stats.ttest_ind(tc_n, tc_f)[0]



    ec_n = pd.melt(malm_n.loc[:,['ec_11', 'ec_12', 'ec_13', 'ec_14', 'ec_15',
                                 'ec_16', 'ec_17', 'ec_18']])
    ec_n=ec_n.iloc[:,1].dropna()
    
    ec_f = pd.melt(malm_f.loc[:,['ec_11', 'ec_12', 'ec_13', 'ec_14', 'ec_15',
                                 'ec_16', 'ec_17', 'ec_18']])
    ec_f=ec_f.iloc[:,1].dropna()
    
    ec_ttest = stats.ttest_ind(ec_n, ec_f)[0]
    
    ttest = [pc_ttest, tc_ttest, ec_ttest]
    
    ttest = pd.DataFrame(data=ttest)
    
    ttest.index = ['tfpch', 'techch', 'effch']
    
    ax=plt.subplots(figsize=(10,8))
    ax= ttest.plot(kind='bar', color='blue', legend = None)
    ax.axhline(1.645, linestyle='--', color='grey', linewidth=2)
    ax.axhline(0, color='black', linewidth=2)
    ax.axhline(-1.645, linestyle='--', color='grey', linewidth=2)
    plt.xlabel('Malmquist Indices')
    plt.ylabel('T-Statistic')
    plt.title('Malmquist indices differences for laggard and leader firms')
    return ax
    
visualization_ttest(eff_chem, df)


preda.visualization_ttest_years(eff_chem, df)

preda.visualization_ttest(eff_chem, df)


ttest = preda.malm_ttest_sectors(eff_chem, df_chem, eff_elec, df_elec, eff_mac, df_mac)

preda.visualization_ttest(ttest)


malm_n, malm_f, r = preda.malm_by_efflevel(eff_chem, df_chem)
pc_n = pd.melt(malm_n.loc[:,['pc_11', 'pc_12', 'pc_13', 'pc_14', 'pc_15',
                                 'pc_16', 'pc_17', 'pc_18']])
pc_n=pc_n.iloc[:,1].dropna()
pc_f = pd.melt(malm_f.loc[:,['pc_11', 'pc_12', 'pc_13', 'pc_14', 'pc_15',
                                 'pc_16', 'pc_17', 'pc_18']])
pc_f=pc_f.iloc[:,1].dropna()
    
stats.mannwhitneyu(pc_f, pc_n, alternative='greater')

preda.malm_ttest_period(eff_chem, df_chem)


tc_n = pd.melt(malm_n.loc[:,['tc_11', 'tc_12', 'tc_13', 'tc_14', 'tc_15',
                                 'tc_16', 'tc_17', 'tc_18']])
tc_n=tc_n.iloc[:,1].dropna()
    
tc_f = pd.melt(malm_f.loc[:,['tc_11', 'tc_12', 'tc_13', 'tc_14', 'tc_15',
                                 'tc_16', 'tc_17', 'tc_18']])
tc_f=tc_f.iloc[:,1].dropna()
stats.mannwhitneyu(tc_f, tc_n, alternative='less')


'''Mann Whitney U test'''
#chemical
cpc_n, cpc_f, ctc_n, ctc_f, cec_n, cec_f = preda.data_malm_ttest_period(eff_chem, df_chem)

from scipy import stats

stats.mannwhitneyu(cpc_n, cpc_f, alternative='less')

stats.mannwhitneyu(ctc_n, ctc_f, alternative='greater')

stats.mannwhitneyu(cec_n, cec_f, alternative='less')

stats.ttest_ind(cpc_n, cpc_f)
stats.ttest_ind(ctc_n, ctc_f)
stats.ttest_ind(cec_n, cec_f)
cpc_n.plot()

plt.hist(cpc_n, bins=1000)
plt.hist(cpc_f, bins=1000)
max(cpc_n)
max(cpc_f)
malm_n, malm_f, result = preda.malm_by_efflevel(eff_chem, df_chem)

preda.visualization_ttest_years(eff_chem, df_chem)

e= eff_chem_dmu.T.iloc[:-1,:]
for i in range(2010,2019):
    e[i].values[e[i] <1]=np.nan
e['firm'] = e.index    

e= pd.melt(e, 'firm')
a=[]
a.append(1)

sns.scatterplot(x='variable', y='value', hue='firm', data=e)

stats.ks_2samp(cpc_n, cpc_f)
np.linspace(-15, 15, 9)
[2,3]


s1 = pd.Series(['100', '200', 'python', '300.12', '400'])
print("Original Data Series:")
print(s1)
print("Series to an array")
a = np.array(s1.values.tolist())
print (a)


df_matured, df_intermediate, df_young = preda.df_by_age(df_chem, 140, 36, 21)

df_young = df_young.iloc[:,[1,4,7,10,13,16,19,22]]
arr=[]
for i in range(0,8):
    arr.append(stats.ttest_1samp(df_young.iloc[:,i],1,nan_policy='omit'))
stats.ttest_1samp(df_matured.iloc[:,1],1,nan_policy='omit')
rvs = stats.norm.rvs(loc=5, scale=10, size=(50,2))
stats.ttest_1samp(rvs,5.0)

"""robustness test"""

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

def eff_distribution_OT(eff_chem, summary, n_matured= 140, n_intermediate=36, n_young=22,
                        title =' Efficiency level of chemical sector over time'):
    '''boxplot and geometric mean for the whole sector
        require table of efficient score from R and summary efficient score as a whole sector'''
    eff_melt = rearrange_eff_data(eff_chem)
    eff_melt['Year'].astype('int')
    ax=plt.subplots(figsize=(10,8))
    ax=sns.boxplot(x='Year', y='Eff_score', data= eff_melt.dropna(), linewidth=.9, color='steelblue')
    ax=sns.pointplot(x=eff_melt.iloc[0:9,1], y=summary['Mean'], scale=1, color='k', errwidth=1.5, capsize=0.2, markers='x', linestyles=' ')
    ax=sns.pointplot(x=eff_melt.iloc[0:9,1], y=summary['Mean'], scale=0.4, color='k', errwidth=0, capsize=0, linestyles='--')
    plt.title(title, fontsize=20)
    return ax

eff_elec_melt = preda.rearrange_eff_data(eff_elec,175, 41, 17)

eff_chem_melt = preda.rearrange_eff_data(eff_chem,140, 36, 21)


eff_elec_melt['Year'].astype('int')
ax=plt.subplots(figsize=(10,8))
ax=sns.boxplot(x='Year', y='Eff_score', data= eff_elec_melt.dropna(), linewidth=.9, color='steelblue')
ax=sns.pointplot(x=eff_elec_melt.iloc[0:9,1], y=summary_elec['Mean'], scale=1, color='k', errwidth=1.5, capsize=0.2, markers='x', linestyles=' ')
ax=sns.pointplot(x=eff_elec_melt.iloc[0:9,1], y=summary_elec['Mean'], scale=0.4, color='k', errwidth=0, capsize=0, linestyles='--')
plt.title(title, fontsize=20)


eff_chem_melt.info()

def absd(eff):
    
    arr = []
    for yrs in range(11,19):
        for i in range(eff.shape[0]):
            if eff.loc[i, f'pc_{yrs}'] > 1:
                arr.append(np.nanprod(eff[f'ec_{yrs}'])**(1.0/eff[f'ec_{yrs}'].notnull().sum()))
                arr.append(np.nanprod(eff[f'tc_{yrs}'])**(1.0/eff[f'tc_{yrs}'].notnull().sum())) 
    return arr
    
def abds(eff):    
    eff_matured = eff.loc[eff['age']==1]
    eff_intermediate = eff[eff['age']==2]
    eff_young = eff[eff['age']==3]   
    arr = []
    arr.append(absd(eff_matured))
    arr.append(absd(eff_intermediate))
    arr.append(absd(eff_young))
    return arr

x = abds(df_chem[df_chem['age']==1])


eff_matured = eff.loc[eff['age']==1]
for i in range(11,19):
    

ax= sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(malm_change_chem.iloc[:,0:4], ['year']))
ax.axhline(y=1)


yrs = [i for i in range(2011,2019)]
tstats = pd.DataFrame(data=pd.concat([yrs, tstats_matured_chem[:,1], tstats_intermediate_chem[:,1], tstats_young_chem[:,1]], axis = 1))


near_c, further_c = preda.malm_by_efflevel(eff_chem, df_chem)
df_chem.iloc[:,1:25] = df_chem.iloc[:,1:25] - 1
for yrs in range(11,19):
        near_c[f'tc_{yrs}'] = near_c[f'tc_{yrs}']/near_c[f'pc_{yrs}']
        further_c[f'tc_{yrs}'] = further_c[f'tc_{yrs}']/further_c[f'pc_{yrs}']
        
malm_n = pd.DataFrame(data=[])
malm_f = pd.DataFrame(data=[])
for i in range(10, 18):
        j=i+1
        eff_chem[f'rank_{i}'] = eff_chem[f'eff_scores_{i}'].rank(ascending=False)
        firm_near = eff_chem.loc[eff_chem[f'rank_{i}'] <= eff_chem[f'rank_{i}'].median(), 'firm_id']
        firm_further = eff_chem.loc[eff_chem[f'rank_{i}'] > eff_chem[f'rank_{i}'].median(), 'firm_id']
        malm_near = df_chem.loc[df_chem['dmu'].isin(firm_near), [f'pc_{j}', f'tc_{j}', f'ec_{j}']]
        malm_n = pd.concat([malm_n, malm_near], axis = 1)
        #result_near.append(gmean(malm_near, axis = 0))
        malm_further = df_chem.loc[df_chem['dmu'].isin(firm_further), [f'pc_{j}', f'tc_{j}', f'ec_{j}']]
        malm_f = pd.concat([malm_f, malm_further], axis = 1)
        #result_further.append(gmean(malm_further, axis = 0))

def source_pc_ec(df, n_matured, n_intermediate, n_young, yrs):
    ec_matured_a = 0
    ec_intermediate_a = 0 # number of young firms have pc mainly from ec
    ec_young_a = 0
    for i in range(0,n_matured):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'pec_{yrs}'] > df.loc[i,f'tc_{yrs}']:
            ec_matured_a += 1
    for i in range(n_matured, n_matured+n_intermediate):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'pec_{yrs}'] > df.loc[i,f'tc_{yrs}']:
            ec_intermediate_a += 1
    for i in range(n_matured+n_intermediate, n_matured+n_intermediate+n_young):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'pec_{yrs}'] > df.loc[i,f'tc_{yrs}']:
            ec_young_a  += 1
    ec_matured_b = 0
    ec_intermediate_b = 0 # number of young firms have pc only from ec
    ec_young_b = 0
    for i in range(0,n_matured):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'tc_{yrs}'] <= 1:
            ec_matured_b += 1
    for i in range(n_matured, n_matured+n_intermediate):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'tc_{yrs}'] <= 1:
            ec_intermediate_b += 1
    for i in range(n_matured+n_intermediate, n_matured+n_intermediate+n_young):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'tc_{yrs}'] <= 1:
            ec_young_b += 1
    return ec_matured_a, ec_matured_b, ec_intermediate_a, ec_intermediate_b, ec_young_a, ec_young_b


def weighted_avg(eff, sales, n_matured, n_intermediate, n_young):
    for i in range(10,19):
        eff.loc[:,f'eff_scores_{i}'] = eff.loc[:,f'eff_scores_{i}'] * sales.loc[,f's_{i}']
    mean_pc = []
    mean_ec = []
    mean_tc = []
    for i in range(11,19):
        geo_mean_pc.append(np.nanmean(df[f'pc_{i}']))
        geo_mean_ec.append(np.nanmean(df[f'ec_{i}']))
        geo_mean_tc.append(np.nanmean(df[f'tc_{i}']))
    geo_mean_pc = pd.DataFrame(data=geo_mean_pc)
    geo_mean_ec = pd.DataFrame(data=geo_mean_ec)
    geo_mean_tc = pd.DataFrame(data=geo_mean_tc)
    avg_change = pd.concat([geo_mean_pc, geo_mean_ec, geo_mean_tc], axis = 1)
    avg_change.columns = ['avg_pc', 'avg_ec', 'avg_tc']
    avg_change.index = list(range(2011,2019))
    if keyword=='total':
        avg_change['year'] = list(range(2011,2019))
        avg_change = avg_change.loc[:,['year', 'avg_pc', 'avg_ec', 'avg_tc']]
    if keyword=='partial':
        avg_change=avg_change


a = preda.efficiency_sum_stats_weighted(eff_chem, sales_chem)
c_w = preda.eff_stats_by_age_merged_weighted(eff_chem, sales_chem, 140,36,21)


import pylab as plt
import seaborn as sns

tips = sns.load_dataset("tips")
fig, ax = plt.subplots()

sns.barplot(data=tips, ax=ax, x="time", y="tip", hue="sex")

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .05)
plt.show()


fig, ax = plt.subplots(figsize=(15,15))
sns.barplot(x='year', y='value', hue='variable', ax= ax,
                 data=pd.melt(malm_change_chem.iloc[:,[0,1,4,7,10]], ['year']))
change_width(ax, .15)
plt.show()

df = pd.concat([eff_chem, eff_elec, eff_mac], axis = 0).iloc[:, 1:]
preda.eff_distribution_OT_by_sector(eff_chem, eff_elec, eff_mac)
