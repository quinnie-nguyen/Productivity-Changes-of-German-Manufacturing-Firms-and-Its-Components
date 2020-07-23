# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:34:55 2020

@author: Quinn
@preprocessing data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns_name = ['ID','company', 'city', 'region', 'date_cor', 
                'fixed_asset_2019','fixed_asset_2018', 'fixed_asset_2017', 'fixed_asset_2016', 'fixed_asset_2015', 'fixed_asset_2014', 'fixed_asset_2013', 'fixed_asset_2012', 'fixed_asset_2011', 'fixed_asset_2010',
                'employee_2019','employee_2018', 'employee_2017', 'employee_2016', 'employee_2015', 'employee_2014', 'employee_2013', 'employee_2012', 'employee_2011', 'employee_2010',
                'employee_cost_2019','employee_cost_2018', 'employee_cost_2017', 'employee_cost_2016', 'employee_cost_2015', 'employee_cost_2014', 'employee_cost_2013', 'employee_cost_2012', 'employee_cost_2011', 'employee_cost_2010',
                'material_cost_2019','material_cost_2018', 'material_cost_2017', 'material_cost_2016','material_cost_2015', 'material_cost_2014', 'material_cost_2013', 'material_cost_2012', 'material_cost_2011', 'material_cost_2010',
                'sale_2019', 'sale_2018', 'sale_2017', 'sale_2016', 'sale_2015', 'sale_2014', 'sale_2013', 'sale_2012', 'sale_2011','sale_2010']
num_var = ['fixed_asset_2019','fixed_asset_2018', 'fixed_asset_2017', 'fixed_asset_2016', 'fixed_asset_2015', 'fixed_asset_2014', 'fixed_asset_2013', 'fixed_asset_2012', 'fixed_asset_2011', 'fixed_asset_2010',
                'employee_2019','employee_2018', 'employee_2017', 'employee_2016', 'employee_2015', 'employee_2014', 'employee_2013', 'employee_2012', 'employee_2011', 'employee_2010',
                'employee_cost_2019','employee_cost_2018', 'employee_cost_2017', 'employee_cost_2016', 'employee_cost_2015', 'employee_cost_2014', 'employee_cost_2013', 'employee_cost_2012', 'employee_cost_2011', 'employee_cost_2010',
                'material_cost_2019','material_cost_2018', 'material_cost_2017', 'material_cost_2016','material_cost_2015', 'material_cost_2014', 'material_cost_2013', 'material_cost_2012', 'material_cost_2011', 'material_cost_2010',
                'sale_2019', 'sale_2018', 'sale_2017', 'sale_2016', 'sale_2015', 'sale_2014', 'sale_2013', 'sale_2012', 'sale_2011','sale_2010']

'''import the very messy data'''
def get_raw_data(title ='Chemicals_raw.xls'):
    chem = pd.read_excel(title, header=0, index=0, na_values='n.a.', parse_dates=True)
    chem = chem.drop('Release date', axis = 1)
    chem.columns = columns_name
    chem.ID = chem.ID.astype(int)
    '''because min timestamp is 1677, some firms existed before'''
    chem['date_cor']=pd.to_datetime(chem['date_cor'], errors = 'coerce')
    chem['date_cor']=chem['date_cor'].replace('NaT', np.nan)
    year = []
    for date in chem.date_cor:
        year.append(str(date).split('-')[0])
    chem['year_cor'] = year
    #chem['year_cor'] = chem['year_cor'].astype(int)
    return chem
'''def get_data_elec():
    elec = pd.read_excel('Electronics_raw.xls', header=0, index=0, na_values='n.a.', parse_dates=True)
    elec = elec.drop('Release date', axis = 1)
    elec.columns = columns_name
    elec.ID = elec.ID.astype(int)
    #because min timestamp is 1677, some firms existed before
    elec['date_cor']=pd.to_datetime(elec['date_cor'], errors = 'coerce')
    elec['date_cor']=elec['date_cor'].replace('NaT', np.nan)
    year = []
    for date in elec.date_cor:
        year.append(str(date).split('-')[0])
    elec['year_cor'] = year
    #chem['year_cor'] = chem['year_cor'].astype(int)
    return elec
def get_data_mac():
    mac = pd.read_excel('Machinery_raw.xls', header=0, index=0, na_values='n.a.', parse_dates=True)
    mac = mac.drop('Release date', axis = 1)
    mac.columns = columns_name
    mac.ID = mac.ID.astype(int)
    #because min timestamp is 1677, some firms existed before
    mac['date_cor']=pd.to_datetime(mac['date_cor'], errors = 'coerce')
    mac['date_cor']=mac['date_cor'].replace('NaT', np.nan)
    year = []
    for date in mac.date_cor:
        year.append(str(date).split('-')[0])
    mac['year_cor'] = year
    #chem['year_cor'] = chem['year_cor'].astype(int)
    return mac'''
def get_index():
    index = pd.read_excel('index_for_correcting_data.xlsx')
    return index

'''
    delete obs that dont have year of incorporation'''
def clean_datechem(chem):
    date_na_index = [828, 973, 1120, 1297, 1320, 1527, 1948, 2242, 2413, 
                 2724, 2745, 2797, 2978, 3316, 3702, 3802, 3998, 4374, 4809, 4920, 4935, 5239]
    chem=chem.drop(chem.index[date_na_index])
    chem['year_cor'].replace('NaT', '2010', inplace = True)
    return chem
def clean_dateelec(elec):
    nulldate_elec = [46, 76, 204, 475, 485,  566, 703,  728,   986,  1829,  2138,  2494,  3561,
             3679,  3950,  4412,  4516,  4837,  5552,  6223,  6238,  6494,
             6718,  7241,  7320,  7433,  7939,  7984,  8359,  8719,  8848,
             8874,  8888,  9663, 10343, 10711, 11524, 13120, 13694, 13743]
    elec = elec.drop(elec.index[nulldate_elec])
    return elec
def clean_datemac(mac):
    nulldate_mac = [  10, 15, 260,  311,   651,  1172,  1214,  1479,  1805,  2473,  2490,
             2806,  2913,  3029,  3177,  3185,  3797,  3957,  4434,  4449,
             4470,  4630,  4817,  5294,  5430,  5501,  5523,  5747,  6304,
             6379,  6778,  7097,  7273,  8624,  8940,  9022,  9253,  9593,
             9956, 10280, 10396, 10744, 10882, 10987, 11229, 11297, 11489,
            11917, 12182, 12186, 12191, 12320, 12476, 12526, 13536, 13818,
            13841, 13843, 13844, 13845, 13965, 14138, 14632, 15076, 15255,
            15716]
    mac = mac.drop(mac.index[nulldate_mac])
    return mac

def get_f_wrt_age(chem):
    '''get usefull data for old and young firm, but need one more step to get data for
    newborn firms
        use this function for 3 sectors'''
    chem['year_cor']=chem['year_cor'].astype(int)
    chem=chem.drop(['city', 'region', 'date_cor', 
                'fixed_asset_2019', 'employee_2019','employee_cost_2019',
                'material_cost_2019','sale_2019'], axis = 1)
    old_chem = chem.loc[chem['year_cor'] < 2000].dropna()
    young_chem = chem.loc[chem['year_cor'].isin(range(2000,2010))].dropna()
    newborn_chem = chem.loc[chem['year_cor'] > 2009]
    return old_chem, young_chem, newborn_chem
def get_data_newborn(r):
    '''make sure that newborn firms have data availabel for at least 4 years, any additional
    year is a bonus'''
    def get_newborn(r):
        r_1815 = r[['ID','company','fixed_asset_2018',
        'employee_2018','material_cost_2018','sale_2018','fixed_asset_2017',
        'employee_2017','material_cost_2017','sale_2017',
        'fixed_asset_2016','employee_2016','material_cost_2016','sale_2016',
        'fixed_asset_2015','employee_2015','material_cost_2015','sale_2015','year_cor']]
        return r_1815.dropna()
    id_list = list(get_newborn(r)['ID'])
    r = r.loc[r['ID'].isin(id_list),:]
    return r
def deflate_data(df_raw, index):
    '''
    fixed_asset*capacity use/100
    employee * effective hour
    material cost and sale /ppi * 100   
    '''
    df = df_raw.copy()
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
    return df_raw, df
'''def current_entry(r):
    r_2018 = r[['ID','company','fixed_asset_2018',
        'employee_2018','material_cost_2018','sale_2018','year_cor']]
    r_2017 = r[['ID','company','fixed_asset_2017',
        'employee_2017','material_cost_2017','sale_2017','year_cor']]
    r_2016 = r[['ID','company','fixed_asset_2016',
        'employee_2016','material_cost_2016','sale_2016','year_cor']]
    r_2015 = r[['ID','company','fixed_asset_2015',
        'employee_2015','material_cost_2015','sale_2015','year_cor']]
    r_2014 = r[['ID','company','fixed_asset_2014',
        'employee_2014','material_cost_2014','sale_2014','year_cor']]
    r_2013 = r[['ID','company','fixed_asset_2013',
        'employee_2013','material_cost_2013','sale_2013','year_cor']]
    r_2012 = r[['ID','company','fixed_asset_2012',
        'employee_2012','material_cost_2012','sale_2012','year_cor']]
    r_2011 = r[['ID','company','fixed_asset_2011',
        'employee_2011','material_cost_2011','sale_2011','year_cor']]
    r_2010 = r[['ID','company','fixed_asset_2010',
        'employee_2010','material_cost_2010','sale_2010','year_cor']]
    return r_2018, r_2017, r_2016, r_2015, r_2014, r_2013, r_2012, r_2011, r_2010'''
def Malmquist_data(df, n_firms = 198):
    arr = []
    for i in range(0, df.shape[1]):
        col = df.iloc[:, i]
        arr.append(col)
    y_10 = pd.Series(np.repeat(2010, n_firms))
    y_11 = pd.Series(np.repeat(2011, n_firms))
    y_12 = pd.Series(np.repeat(2012, n_firms))
    y_13 = pd.Series(np.repeat(2013, n_firms))
    y_14 = pd.Series(np.repeat(2014, n_firms))
    y_15 = pd.Series(np.repeat(2015, n_firms))
    y_16 = pd.Series(np.repeat(2016, n_firms))
    y_17 = pd.Series(np.repeat(2017, n_firms))
    y_18 = pd.Series(np.repeat(2018, n_firms))
    year= pd.concat([y_10, y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18])
    firm = pd.concat([arr[0],arr[0],arr[0],arr[0],arr[0],arr[0],arr[0],arr[0],arr[0]])
    fa = pd.concat([arr[i] for i in range(10, 1, -1)])
    em = pd.concat([arr[i] for i in range(19, 10, -1)])
    ec = pd.concat([arr[i] for i in range(28,19, -1)])
    mc = pd.concat([arr[i] for i in range(37, 28,-1)])
    sa = pd.concat([arr[i] for i in range(46, 37, -1)])
    malmquist = pd.DataFrame({
                                'year': year,
                                'firm': firm,
                                'fixed_assets': fa,
                                'employees': em,
                                'emp_cost': ec,
                                'mat_cost': mc,
                                'sales': sa})
    return malmquist
def read_eff_score(title):
    '''read result of DEA model from R
       title is in title list '''
    df = pd.read_csv(title).drop('Unnamed: 0', axis =1)
    df.replace(0,np.nan, inplace=True)
    return df

def hist_eff_score(eff_elec):
    ncols = 3
    nrows = int(eff_elec.shape[1]/ncols)
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(10,10))
    counter = 1
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            if counter < len(eff_elec.columns)+1:
                ax.hist(eff_elec.loc[eff_elec.iloc[:,counter].notnull(), eff_elec.columns[counter]], bins=20, facecolor='slategray',
                        alpha=1,rwidth=0.85, label='{}'.format(eff_elec.columns[counter]))
                #ax.set_xlabel('efficiency score')
                #ax.set_ylabel('number of firms')
                ax.set_ylim([0,60])
                leg=ax.legend(loc='upper right')
                leg.draw_frame(False)
            else:
                ax.set_axis_off()
            counter+=1
    plt.title('robustness check', fontsize=20, color='black')
    return ax

def eff_dmu(eff_elec):
    eff_dmu_id = eff_elec.loc[(eff_elec['eff_scores_10'] == 1) | (eff_elec['eff_scores_11'] == 1) |
             (eff_elec['eff_scores_12'] == 1) | (eff_elec['eff_scores_13'] == 1) |
             (eff_elec['eff_scores_14'] == 1) | (eff_elec['eff_scores_15'] == 1) |
             (eff_elec['eff_scores_16'] == 1) | (eff_elec['eff_scores_17'] == 1) |
             (eff_elec['eff_scores_18'] == 1), 'firm_id']
    eff_dmu = eff_elec.loc[eff_elec['firm_id'].isin(eff_dmu_id),].reset_index()
    eff_dmu = pd.DataFrame(eff_dmu.T)
    eff_dmu.columns = ['firm_' + str(x) for x in eff_dmu_id]
    years = list(range(2008,2019,1))
    eff_dmu['year'] = years
    eff_dmu.index = years
    eff_dmu=eff_dmu.iloc[2:11,]
    return eff_dmu

def read_malmquist(title, n_matured, n_intermediate, n_young):
    '''enter the title of file'''
    df = pd.read_csv(title).drop('Unnamed: 0', axis=1)
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(0, np.nan, inplace=True)
    age = pd.concat([pd.Series(np.repeat(1, n_matured)), pd.Series(np.repeat(2, n_intermediate)), pd.Series(np.repeat(3, n_young))])
    df['age'] = list(age)
    return df

def malmquist_by_age(df, n_old, n_young, n_newborn):
    '''split the malmquist file by age'''
    df_matured = df.iloc[0:n_old,:]
    df_intermediate = df.iloc[n_old:(n_old+n_young),:]
    df_young = df.iloc[(n_old+n_young):(n_old+n_young+n_newborn),:] 
    return df_matured, df_intermediate, df_young

def encode_change(df, n_old=140, n_young=36, n_newborn=22):
    '''if grow then 1, else 0
    for pc, ec, tc'''
    df_compare = pd.DataFrame().reindex_like(df)
    for i in df.columns:
        if i == 'dmu':
            df_compare[i] = df[i]
        else:
            df_compare[i] = np.where(df[i] > 1, 1, 0)
    age = pd.concat([pd.Series(np.repeat('old', n_old)), pd.Series(np.repeat('young', n_young)), pd.Series(np.repeat('newborn', n_newborn))])
    df_compare['age'] = list(age)
    return df_compare.reset_index().drop('index', axis = 1)


def growth_dmu(df, n_firm=197):
    '''how many dmu grow over time in productivity'''
    growth = 0
    dmu = []
    for i in range(0, n_firm):
        if all(df.iloc[i,[1,4,7,10,13,16,19,22]] > 1):
            growth += 1
            dmu.append(df.iloc[i,0])
    return growth, dmu

def ec_dmu(df_compare):
    '''how many dmu grow over time in efficiency change'''
    efficiency_growth = 0
    dmu = []
    for i in range(0, 198):
        if all(df_compare.iloc[i,[2,5,8,11,14,17,20,23]] ==1):
            efficiency_growth += 1
            dmu.append(i)
    return efficiency_growth, dmu

def prod_growth(df, n_matured, n_intermediate, n_young, yrs):
    '''how many firms grow in each year,
        in which firm is matured, intermediate or young'''
    g = 0
    g_m = 0
    g_i = 0
    g_y = 0
    for i in range(0, n_matured+n_intermediate+ n_young):
        if df.loc[i,f'pc_{yrs}'] > 1:
            g += 1
            if df.loc[i,'age'] == 1:
                g_m += 1
            elif df.loc[i,'age'] == 2:
                g_i += 1
            else:
                g_y += 1
    g_percent  = g/ (n_matured+n_intermediate+ n_young)
    gm_percent = g_m / n_matured
    gi_percent = g_i / n_intermediate
    return g, g_m, g_i, g_y, g_percent, gm_percent, gi_percent

def prod_growth_1018(df, n_matured, n_intermediate, n_young):
    arr = []
    for yrs in range(11,19):
        arr.append(prod_growth(df, n_matured, n_intermediate, n_young, yrs))
    result = pd.DataFrame(data = arr)
    result.columns = ['prod_growth', 'matured_growth', 'intermediate_growth',
                      'young_growth', '%growth', '%mgrowth', '%igrowth']
    return result
        
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
    
def source_pc_sector_ec(df, n_matured, n_intermediate, n_young):
    arr = []
    for i in range(11, 19):
        arr.append(source_pc_ec(df, n_matured, n_intermediate, n_young,i))
    r = pd.DataFrame(data=arr)
    r.columns = ['ec_driving_matured', 'only_ec_driving_matured','ec_driving_intermediate', 'only_ec_driving_intermediate', 
                 'ec_driving_young', 'only_ec_driving_young']
    r.index = list(range(2011,2019))
    return r

def source_pc_tc(df, n_matured, n_intermediate, n_young, yrs):
    tc_matured_a = 0
    tc_intermediate_a = 0 # number of young firms have pc mainly from ec
    tc_young_a = 0
    for i in range(0,n_matured):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'tc_{yrs}'] > df.loc[i,f'ec_{yrs}']:
            tc_matured_a += 1
    for i in range(n_matured, n_matured+n_intermediate):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'tc_{yrs}'] > df.loc[i,f'ec_{yrs}']:
            tc_intermediate_a += 1
    for i in range(n_matured+n_intermediate, n_matured+n_intermediate+n_young):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'tc_{yrs}'] > df.loc[i,f'ec_{yrs}']:
            tc_young_a  += 1
    tc_matured_b = 0
    tc_intermediate_b = 0 # number of young firms have pc only from ec
    tc_young_b = 0
    for i in range(0,n_matured):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'ec_{yrs}'] <= 1:
            tc_matured_b += 1
    for i in range(n_matured, n_matured+n_intermediate):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'ec_{yrs}'] <= 1:
            tc_intermediate_b += 1
    for i in range(n_matured+n_intermediate, n_matured+n_intermediate+n_young):
        if df.loc[i,f'pc_{yrs}'] > 1 and df.loc[i,f'ec_{yrs}'] <= 1:
            tc_young_b += 1
    return tc_matured_a, tc_matured_b, tc_intermediate_a, tc_intermediate_b, tc_young_a, tc_young_b
    
def source_pc_sector_tc(df, n_matured, n_intermediate, n_young):
    arr = []
    for i in range(11, 19):
        arr.append(source_pc_tc(df, n_matured, n_intermediate, n_young,i))
    r = pd.DataFrame(data=arr)
    r.columns = ['tc_driving_matured', 'only_tc_driving_matured','tc_driving_intermediate', 'only_tc_driving_intermediate', 
                 'tc_driving_young', 'only_tc_driving_young']
    r.index = list(range(2011,2019))
    return r

def effciency_change(df, n_old, n_young, n_newborn, yrs = 11):
    '''number of firms have ec > 1 and ec > tc, regardless the direction of pc'''
    ec_old = 0
    ec_young = 0 
    ec_newborn = 0
    # arr contains index of firms 
    arr=[]
    tc_depression_old = 0
    tc_depression_young = 0
    tc_depression_newborn = 0
    for i in range(0,n_old):
        if df.loc[i,f'ec_{yrs}'] > 1 and df.loc[i,f'ec_{yrs}'] > df.loc[i,f'tc_{yrs}']:
            ec_old += 1
            arr.append(i)
            if df.loc[i, f'pc_{yrs}'] < 1:
                tc_depression_old +=1
    for i in range(n_old, n_old+n_young):
        if df.loc[i,f'ec_{yrs}'] > 1 and df.loc[i,f'ec_{yrs}'] > df.loc[i,f'tc_{yrs}']:
            ec_young += 1
            arr.append(i)
            if df.loc[i, f'pc_{yrs}'] < 1:
                tc_depression_young += 1 
    for i in range(n_old+n_young, n_old+n_young+n_newborn):
        if df.loc[i,f'ec_{yrs}'] > 1 and df.loc[i,f'ec_{yrs}'] > df.loc[i,f'tc_{yrs}']:
            ec_newborn += 1
            arr.append(i)
            if df.loc[i, f'pc_{yrs}'] < 1:
                tc_depression_newborn += 1
    arr_2 =[]
    for i in arr:
        if df.loc[df.index[i], f'pc_{yrs}'] < 1:
            arr_2.append(df.loc[df.index[i], 'dmu'])
    return ec_old, tc_depression_old, ec_young, tc_depression_young, ec_newborn, tc_depression_newborn, arr_2

def technical_change(df, n_old, n_young, n_newborn, yrs = 11):
   '''number of firms have tc > 1 and tc > ec, regardless the direction of pc'''
   tc_old = 0
   tc_young = 0 
   tc_newborn = 0
    # arr contains index of firms 
   arr=[]
   ec_depression_old = 0
   ec_depression_young = 0
   ec_depression_newborn = 0
   for i in range(0,n_old):
        if df.loc[i,f'tc_{yrs}'] > 1 and df.loc[i,f'ec_{yrs}'] < df.loc[i,f'tc_{yrs}']:
            tc_old += 1
            arr.append(i)
            if df.loc[i, f'pc_{yrs}'] < 1:
                ec_depression_old +=1
   for i in range(n_old, n_old+n_young):
        if df.loc[i,f'tc_{yrs}'] > 1 and df.loc[i,f'ec_{yrs}'] < df.loc[i,f'tc_{yrs}']:
            tc_young += 1
            arr.append(i)
            if df.loc[i, f'pc_{yrs}'] < 1:
                ec_depression_young += 1     
   for i in range(n_old+n_young, n_old+n_young+n_newborn):
        if df.loc[i,f'tc_{yrs}'] > 1 and df.loc[i,f'ec_{yrs}'] < df.loc[i,f'tc_{yrs}']:
            tc_newborn += 1
            arr.append(i)
            if df.loc[i, f'pc_{yrs}'] < 1:
                ec_depression_newborn += 1
   arr_2 =[]
   for i in arr:
        if df.loc[df.index[i], f'pc_{yrs}'] < 1:
            arr_2.append(df.loc[df.index[i], 'dmu'])
   return tc_old, ec_depression_old, tc_young, ec_depression_young, tc_newborn, ec_depression_newborn, arr_2

def comparison(df, n_old, n_young, n_newborn):
    arr_ec=[]
    arr_tc=[]
    for i in range(11, 19):
        arr_ec.append(effciency_change(df, n_old, n_young, n_newborn, i)[0:6])
        arr_tc.append(technical_change(df, n_old, n_young, n_newborn, i)[0:6])
    r_ec = pd.DataFrame(data=arr_ec)
    r_tc = pd.DataFrame(data=arr_tc)
    r_ec.columns = ['only_ec_old','tc_depression_old', 'only_ec_young','tc_depression_young', 'only_ec_newborn','tc_depression_newborn']
    r_tc.columns = ['only_tc_old','ec_depression_old', 'only_tc_young','ec_depression_young', 'only_tc_newborn', 'ec_depression_newborn']
    total = pd.concat([r_ec, r_tc], axis = 1)
    total.index = list(range(2011,2019))
    return total       

def source_pd_sector(source_pc_machinery, comparison, n_old, n_young, n_newborn):
    pd_old = n_old - source_pc_machinery['pc_old']
    pd_young = n_young - source_pc_machinery['pc_young']
    pd_newborn = n_newborn - source_pc_machinery['pc_newborn']
    ec_depression_old_pct = comparison['ec_depression_old']/pd_old
    ec_depression_young_pct = comparison['ec_depression_young']/pd_young
    ec_depression_newborn_pct = comparison['ec_depression_newborn']/pd_newborn
    r = pd.concat([pd_old,ec_depression_old_pct,pd_young,ec_depression_young_pct,pd_newborn,ec_depression_newborn_pct], axis =1)
    r.columns = ['pd_old', '%ec_pd', 'pd_young', '%ec_young', 'pd_newborn', '%ec_newborn']
    return r

"""def average_change(df, keyword):
    geo_mean_pc = []
    geo_mean_ec = []
    geo_mean_tc = []
    for i in range(11,19):
        geo_mean_pc.append(np.nanprod(df[f'pc_{i}'])**(1.0/df[f'pc_{i}'].notnull().sum()))
        geo_mean_ec.append(np.nanprod(df[f'ec_{i}'])**(1.0/df[f'ec_{i}'].notnull().sum())) 
        geo_mean_tc.append(np.nanprod(df[f'tc_{i}'])**(1.0/df[f'tc_{i}'].notnull().sum()))
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
    return avg_change"""

def average_change(df, keyword):
    '''variable return to scale,
    including pc, pech, sech, tch
    '''
    geo_mean_pc = []
    geo_mean_pec = []
    geo_mean_tc = []
    geo_mean_sec = []

    for i in range(11,19):
        geo_mean_pc.append(np.nanprod(df[f'pc_{i}'])**(1.0/df[f'pc_{i}'].notnull().sum()))
        geo_mean_pec.append(np.nanprod(df[f'pec_{i}'])**(1.0/df[f'pec_{i}'].notnull().sum())) 
        geo_mean_tc.append(np.nanprod(df[f'tc_{i}'])**(1.0/df[f'tc_{i}'].notnull().sum()))
        geo_mean_sec.append(np.nanprod(df[f'sec_{i}'])**(1.0/df[f'sec_{i}'].notnull().sum())) 

    geo_mean_pc = pd.DataFrame(data=geo_mean_pc)
    geo_mean_pec = pd.DataFrame(data=geo_mean_pec)
    geo_mean_tc = pd.DataFrame(data=geo_mean_tc)
    geo_mean_sec = pd.DataFrame(data=geo_mean_sec)

    avg_change = pd.concat([geo_mean_pc, geo_mean_pec, geo_mean_tc, geo_mean_sec], axis = 1)
    avg_change.columns = ['avg_pc', 'avg_pec', 'avg_tc', 'avg_sec']
    avg_change.index = list(range(2011,2019))  
    if keyword=='total':
        avg_change['year'] = list(range(2011,2019))
        avg_change = avg_change.loc[:,['year', 'avg_pc', 'avg_pec', 'avg_tc', 'avg_sec']]
    if keyword=='partial':
        avg_change=avg_change
    return avg_change

def weighted_average_change(df, sales, keyword):
    mean_pc = []
    mean_pec = []
    mean_tc = []
    mean_sec = []
    for i in range(11,19):
        j = i - 1
        mean_pc.append(np.nansum(df.loc[:,f'pc_{i}'] * sales.loc[:,f's_{j}'])/np.nansum(sales.loc[:,f's_{j}']))
        mean_pec.append(np.nansum(df.loc[:,f'pec_{i}'] * sales.loc[:,f's_{j}'])/np.nansum(sales.loc[:,f's_{j}']))                  
        mean_tc.append(np.nansum(df.loc[:,f'tc_{i}'] * sales.loc[:,f's_{j}'])/np.nansum(sales.loc[:,f's_{j}']))
        mean_sec.append(np.nansum(df.loc[:,f'sec_{i}'] * sales.loc[:,f's_{j}'])/np.nansum(sales.loc[:,f's_{j}']))
    mean_pc = pd.DataFrame(data=mean_pc)
    mean_pec = pd.DataFrame(data=mean_pec)
    mean_tc = pd.DataFrame(data=mean_tc)
    mean_sec = pd.DataFrame(data=mean_sec)

    avg_change = pd.concat([mean_pc, mean_pec, mean_tc, mean_sec], axis = 1)
    avg_change.columns = ['avg_pc', 'avg_pec', 'avg_tc', 'avg_sec']
    avg_change.index = list(range(2011,2019))
    if keyword=='total':
        avg_change['year'] = list(range(2011,2019))
        avg_change = avg_change.loc[:,['year', 'avg_pc', 'avg_pec', 'avg_tc', 'avg_sec']]
    if keyword=='partial':
        avg_change=avg_change
    return avg_change

def avg_change_bygroup(df, n_matured = 140, n_intermediate = 36, n_young = 21):
    
    """a table that indicates the average of pc, tc, ec overtime for different age group"""
    
    df_matured = df.iloc[:n_matured,:]
    df_intermediate = df.iloc[n_matured:n_matured+n_intermediate,:]
    df_young = df.iloc[n_matured+n_intermediate:, :]
    avg_matured = average_change(df_matured, 'partial')
    avg_intermediate = average_change(df_intermediate, 'partial')
    avg_young = average_change(df_young, 'partial')
    r = pd.DataFrame(data = pd.concat([avg_matured, avg_intermediate, avg_young], axis=1))
    r['year'] = list(range(2011, 2019))
    r.columns = ['MI_matured', 'PEC_matured', 'TC_matured', 'SEC_matured',
                 'MI_intermediate', 'PEC_intermediate', 'TC_intermediate', 'SEC_intermediate',
                 'MI_young', 'PEC_young', 'TC_young', 'SEC_young', 'year']
    r = r.loc[:,["year",'MI_matured','MI_intermediate', 'MI_young', 
                           'PEC_matured', 'PEC_intermediate', 'PEC_young',
                           'TC_matured', 'TC_intermediate', 'TC_young',
                           'SEC_matured', 'SEC_intermediate', 'SEC_young',]]
    r = r-1
    r['year'] = r['year'] + 1
    return r


def weighted_avg_change_bygroup(df, sales, n_matured, n_intermediate, n_young):
    
    """a table that indicates the average of pc, tc, pec, sec over time for different age group"""
    
    df_matured, df_intermediate, df_young = df_by_age(df, n_matured, n_intermediate, n_young)
    sales_matured, sales_intermediate, sales_young = df_by_age(sales, n_matured, n_intermediate, n_young)
    avg_matured = weighted_average_change(df_matured, sales_matured, 'partial')
    avg_intermediate = weighted_average_change(df_intermediate, sales_intermediate, 'partial')
    avg_young = weighted_average_change(df_young, sales_young, 'partial')
    r = pd.DataFrame(data = pd.concat([avg_matured, avg_intermediate, avg_young], axis=1))
    r['year'] = list(range(2011, 2019))
    r.columns = ['MI_matured', 'PEC_matured', 'TC_matured', 'SEC_matured',
                 'MI_intermediate', 'PEC_intermediate', 'TC_intermediate', 'SEC_intermediate',
                 'MI_young', 'PEC_young', 'TC_young', 'SEC_young', 'year']
    r = r.loc[:,["year",'MI_matured','MI_intermediate', 'MI_young', 
                           'PEC_matured', 'PEC_intermediate', 'PEC_young',
                           'TC_matured', 'TC_intermediate', 'TC_young',
                           'SEC_matured', 'SEC_intermediate', 'SEC_young',]]
    r = r-1
    r['year'] = r['year'] + 1
    return r

"""def visualize_change_by_component(change, keyword):
    if keyword=='MI':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,0:4], ['year']), marker='o')
        ax.plot([2011,2018], [1,1], color='black', linestyle=':')
        plt.title('MI over time', color='black')

    
    if keyword=='EC':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,4,5,6]], ['year']), marker='o')
        ax.plot([2011,2018], [1,1], color='black', linestyle=':')
        plt.title('EC over time', color='black')
    
    if keyword=='TC':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,7,8,9]], ['year']), marker='o')
        ax.plot([2011,2018], [1,1], color='black', linestyle=':')
        plt.title('TC over time', color='black')
    
    return ax"""

def visualize_change_by_component(change, keyword, title = "Total factor productivity change of chemical sector (2010-2018)"):
    if keyword=='MI':
        plt.subplots(figsize=(8,6))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,0:4], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')

    
    if keyword=='EC':
        plt.subplots(figsize=(8,6))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,4,5,6]], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')
    
    if keyword=='TC':
        plt.subplots(figsize=(8,6))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,7,8,9]], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')
    
    return ax

"""def visualize_change_by_group(change, keyword, title = 'The evolution in MI, EC, and TC during 2011-2018 of matured firms'):
    if keyword=='matured':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,1,4,7]], ['year']), marker='o')
        ax.plot([2011,2017], [0,0], color='black', linestyle=':')
        plt.title(title, color='black')

    
    if keyword=='intermediate':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,2,5,8]], ['year']), marker='o')
        ax.plot([2011,2017], [0,0], color='black', linestyle=':')
        plt.title(title, color='black')
    
    if keyword=='young':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,3,6,9]], ['year']), marker='o')
        ax.plot([2011,2017], [0,0], color='black', linestyle=':')
        plt.title(title, color='black')
    if keyword=='total':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,0:4], ['year']), marker='o')
        ax.plot([2011,2017], [0,0], color='black', linestyle=':')
        plt.title(title, color='black')
    return ax"""

'''def visualize_change_by_group(change, keyword, title = 'The evolution in MI, EC, and TC during 2011-2018 of matured firms'):
    if keyword=='matured':
        plt.subplots(figsize=(8,6))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,1,4,7]], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')

    
    if keyword=='intermediate':
        plt.subplots(figsize=(8,6))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,2,5,8]], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')
    
    if keyword=='young':
        plt.subplots(figsize=(8,6))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,3,6,9]], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')
    if keyword=='total':
        plt.subplots(figsize=(8,6))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,0:4], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')
    return ax'''


def visualize_change_by_group(change, keyword, title = 'The evolution in MI, EC, and TC during 2011-2018 of matured firms'):
    '''pech, sech seperately
    '''   
    if keyword=='matured':
        plt.subplots(figsize=(15,18))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,1,4,7,10]], ['year']))
        ax.axhline(y=0, color = "black", ls = '-.', lw = 0.5)
        plt.title(title, color='black')

    
    if keyword=='intermediate':
        plt.subplots(figsize=(15,18))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,2,5,8,11]], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')
    
    if keyword=='young':
        plt.subplots(figsize=(15,18))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,[0,3,6,9,12]], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')
    if keyword=='total':
        plt.subplots(figsize=(15,18))
        ax = sns.barplot(x='year', y='value', hue='variable',
                 data=pd.melt(change.iloc[:,0:4], ['year']))
        ax.axhline(y=0, color = "red", ls = '-.')
        plt.title(title, color='black')
    return ax


def efficiency_sum_stats(eff_df):
    n=[]
    mean=[]
    std=[]
    skw=[]
    kurt=[]
    minimum=[]
    Q25=[]
    median=[]
    Q75=[]
    for i in range(10,19):
        n.append(eff_df[f'eff_scores_{i}'].notnull().sum())
        mean.append(np.nanprod(eff_df[f'eff_scores_{i}'])**(1/eff_df[f'eff_scores_{i}'].notnull().sum()))
        std.append(np.nanstd(eff_df[f'eff_scores_{i}']))
        skw.append(eff_df[f'eff_scores_{i}'].skew(skipna=True))
        kurt.append(eff_df[f'eff_scores_{i}'].kurtosis(skipna=True))
        minimum.append(np.nanmin(eff_df[f'eff_scores_{i}']))
        Q25.append(np.nanquantile(eff_df[f'eff_scores_{i}'], 0.25))
        median.append(np.nanmedian(eff_df[f'eff_scores_{i}']))
        Q75.append(np.nanquantile(eff_df[f'eff_scores_{i}'], 0.75))
    n=pd.DataFrame(data=n)
    mean=pd.DataFrame(data=mean)
    std=pd.DataFrame(data=std)
    skw=pd.DataFrame(data=skw)
    kurt=pd.DataFrame(data=kurt)
    minimum=pd.DataFrame(data=minimum)
    Q25=pd.DataFrame(data=Q25)
    median=pd.DataFrame(data=median)
    Q75=pd.DataFrame(data=Q75)
    summary = pd.concat([n, mean, std, skw, kurt, minimum, Q25, median, Q75], axis =1)
    summary['year']=list(range(2010,2019))
    summary.columns =['n', 'Mean', 'Standard deviation', 'Skewness', 'Kurtosis', 'Minimum',
                      'Q25', 'Median', 'Q75', 'year']
    summary=summary.loc[:,['year','n', 'Mean', 'Standard deviation', 'Skewness', 'Kurtosis', 'Minimum',
                      'Q25', 'Median', 'Q75']]
    return summary

def efficiency_sum_stats_weighted(eff_df, sales):
    '''weighted average based on market share (sales)'''
    n=[]
    mean=[]
    std=[]
    skw=[]
    kurt=[]
    minimum=[]
    Q25=[]
    median=[]
    Q75=[]
    for i in range(10,19):
        n.append(eff_df[f'eff_scores_{i}'].notnull().sum())
        mean.append(np.nansum([eff_df.loc[:,f'eff_scores_{i}'] * sales.loc[:,f's_{i}']]))
        std.append(np.nanstd(eff_df[f'eff_scores_{i}']))
        skw.append(eff_df[f'eff_scores_{i}'].skew(skipna=True))
        kurt.append(eff_df[f'eff_scores_{i}'].kurtosis(skipna=True))
        minimum.append(np.nanmin(eff_df[f'eff_scores_{i}']))
        Q25.append(np.nanquantile(eff_df[f'eff_scores_{i}'], 0.25))
        median.append(np.nanmedian(eff_df[f'eff_scores_{i}']))
        Q75.append(np.nanquantile(eff_df[f'eff_scores_{i}'], 0.75))
    n=pd.DataFrame(data=n)
    mean=pd.DataFrame(data=mean)
    std=pd.DataFrame(data=std)
    skw=pd.DataFrame(data=skw)
    kurt=pd.DataFrame(data=kurt)
    minimum=pd.DataFrame(data=minimum)
    Q25=pd.DataFrame(data=Q25)
    median=pd.DataFrame(data=median)
    Q75=pd.DataFrame(data=Q75)
    summary = pd.concat([n, mean, std, skw, kurt, minimum, Q25, median, Q75], axis =1)
    summary['year']=list(range(2010,2019))
    summary.columns =['n', 'Mean', 'Standard deviation', 'Skewness', 'Kurtosis', 'Minimum',
                      'Q25', 'Median', 'Q75', 'year']
    summary=summary.loc[:,['year','n', 'Mean', 'Standard deviation', 'Skewness', 'Kurtosis', 'Minimum',
                      'Q25', 'Median', 'Q75']]
    return summary


def df_by_age(df, n_matured, n_intermediate, n_young):
    
    '''split the df by age'''
    
    df_matured = df.iloc[0:n_matured,:]
    df_intermediate = df.iloc[n_matured:(n_matured+n_intermediate),:]
    df_young = df.iloc[(n_matured+n_intermediate):,:] 
    return df_matured, df_intermediate, df_young

def eff_stats_by_age(eff_df):
    
    '''descriptive statistic for age group'''
    
    n = []
    mean = []
    for i in range(10,19):
        n.append(eff_df[f'eff_scores_{i}'].notnull().sum())
        mean.append(np.nanprod(eff_df[f'eff_scores_{i}'])**(1/eff_df[f'eff_scores_{i}'].notnull().sum()))
    n=pd.DataFrame(data=n)
    mean=pd.DataFrame(data=mean)
    summary_by_age = pd.concat([n, mean], axis=1)
    summary_by_age['year']=list(range(2010,2019))
    summary_by_age.columns =['n', 'Mean', 'year']
    summary_by_age=summary_by_age.loc[:,['year','n', 'Mean']]
    return summary_by_age

from functools import reduce
def eff_stats_by_age_merged(eff_df, n_old, n_young, n_newborn):
    '''summarize the result of statistic from three age groups'''
    eff_old, eff_young, eff_newborn = df_by_age(eff_df, n_old, n_young, n_newborn)
    sum_old = eff_stats_by_age(eff_old)
    sum_old.columns=['year', 'n_old', 'Mean_old']
    sum_young = eff_stats_by_age(eff_young)
    sum_young.columns=['year', 'n_young', 'Mean_young']
    sum_newborn = eff_stats_by_age(eff_newborn)
    sum_newborn.columns=['year', 'n_newborn', 'Mean_newborn']
    sum_final = reduce(lambda left, right: pd.merge(left, right, on='year'), [sum_old, sum_young, sum_newborn])
    return sum_final

def eff_stats_by_age_weighted(eff_df, sales):
    
    '''descriptive statistic for age group with weighted average'''
    
    n = []
    mean = []
    for i in range(10,19):
        n.append(eff_df[f'eff_scores_{i}'].notnull().sum())
        mean.append(np.nansum(eff_df.loc[:,f'eff_scores_{i}']*sales.loc[:,f's_{i}'])/np.nansum(sales.loc[:,f's_{i}']))
    n=pd.DataFrame(data=n)
    mean=pd.DataFrame(data=mean)
    summary_by_age = pd.concat([n, mean], axis=1)
    summary_by_age['year']=list(range(2010,2019))
    summary_by_age.columns =['n', 'Mean', 'year']
    summary_by_age=summary_by_age.loc[:,['year','n', 'Mean']]
    return summary_by_age

def eff_stats_by_age_merged_weighted(eff_df, sales, n_matured, n_intermediate, n_young):
    '''summarize the result of statistic from three age groups'''
    eff_matured, eff_intermediate, eff_young = df_by_age(eff_df, n_matured, n_intermediate, n_young)
    sales_matured, sales_intermediate, sales_young = df_by_age(sales, n_matured, n_intermediate, n_young)
    sum_matured = eff_stats_by_age_weighted(eff_matured, sales_matured)
    sum_matured.columns=['year', 'n_old', 'Mean_old']
    sum_intermediate = eff_stats_by_age_weighted(eff_intermediate, sales_intermediate)
    sum_intermediate.columns=['year', 'n_young', 'Mean_young']
    sum_young = eff_stats_by_age_weighted(eff_young, sales_young)
    sum_young.columns=['year', 'n_newborn', 'Mean_newborn']
    sum_final = reduce(lambda left, right: pd.merge(left, right, on='year'), [sum_matured, sum_intermediate, sum_young])
    return sum_final


def visualize_clustered_toward_eff(data_viz, keyword):
    if keyword=='among sectors':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(data_viz.iloc[:,[0,1,2,3]], ['year']), marker='o')
        plt.title('Average efficiency scores of chemical, electronic and mechinery firms', color='black')

    
    if keyword=='chemical':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(data_viz.iloc[:,[0,4,5,6]], ['year']), marker='o')
        plt.title('Average efficiency scores of chemical firms by ages', color='black')
    
    if keyword=='electronic':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(data_viz.iloc[:,[0,7,8,9]], ['year']), marker='o')
        plt.title('Average efficiency scores of electronic firms by ages', color='black')
   
    if keyword=='machinery':
        plt.subplots(figsize=(8,6))
        ax = sns.lineplot(x='year', y='value', hue='variable',
                 data=pd.melt(data_viz.iloc[:,[0,10,11,12]], ['year']), marker='o')
        plt.title('Average efficiency scores of machinery firms by ages', color='black')
    return ax

def rearrange_eff_data(eff_chem, n_matured= 140, n_intermediate=36, n_young=22):
    '''reconstruct a dataframe to draw boxplot'''
    eff_melt = pd.melt(eff_chem.T.iloc[1:,])
    id_list = eff_chem.iloc[:,0]
    arr_id = []
    for i in id_list:
        arr_id.append(np.repeat(i, 9))
    yrs =[[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],]
    arr_yrs = yrs * (n_matured+ n_intermediate+n_young)
    arr_id = pd.melt(pd.DataFrame(data= arr_id).T)
    arr_yrs = pd.melt(pd.DataFrame(data = arr_yrs).T)
    eff_melt = pd.concat([arr_id['value'], arr_yrs['value'], eff_melt['value']], axis = 1)
    eff_melt.columns = ['ID', 'Year', 'Eff_score']
    arr_age = [['matured',]*n_matured*9, ['intermediate',]*n_intermediate*9, ['young',]*n_young*9]
    arr_age = pd.melt(pd.DataFrame(data=arr_age).T)
    arr_age.replace('None', np.nan, inplace=True)
    arr_age.dropna(inplace=True)
    arr_age= arr_age.reset_index()
    eff_melt['Age'] = arr_age['value']
    return eff_melt

def eff_distribution_OT(eff_chem, summary, n_matured= 140, n_intermediate=36, n_young=22,
                        title =' Efficiency level of chemical sector over time'):
    '''boxplot and geometric mean for the whole sector
        require table of efficient score from R and summary efficient score as a whole sector'''
    eff_melt = rearrange_eff_data(eff_chem)
    #eff_melt['Year'].astype('int')
    ax=plt.subplots(figsize=(10,8))
    ax=sns.boxplot(x='Year', y='Eff_score', data= eff_melt.dropna(), linewidth=.9, color='steelblue')
    ax=sns.pointplot(x=eff_melt.iloc[0:9,1], y=summary['Mean'], scale=1, color='k', errwidth=1.5, capsize=0.2, markers='x', linestyles=' ')
    ax=sns.pointplot(x=eff_melt.iloc[0:9,1], y=summary['Mean'], scale=0.4, color='k', errwidth=0, capsize=0, linestyles='--')
    plt.title(title, fontsize=20)
    return ax


def eff_distribution_OT_by_age(eff_chem, n_matured= 140, n_intermediate=36, n_young=22,
                               title='Efficiency level of chemical firms by age over time'):
    '''boxplot of efficient for firm in different group over time'''
    eff_melt = rearrange_eff_data(eff_chem)
    ax=plt.subplots(figsize=(10,8))
    ax=sns.boxplot(x='Year', y='Eff_score', data= eff_melt.dropna(), hue='Age', palette='GnBu_d')
    plt.title(title, fontsize=20)
    return ax

'''def eff_static(eff_chem_dmu, matured_eff=35, intermediate_eff= 7, young_eff=6):
   # which firms are always on the frontier
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
    table['Always_on_Frontier'] = eff
    firm_id = eff_chem_dmu.columns
    table['firm_id'] = firm_id[:-1]
    fig, ax = plt.subplots(figsize=(10,10))
    plt.axvspan(-1, matured_eff-0.5, facecolor='g', alpha=0.5)
    plt.axvspan(matured_eff-0.5, matured_eff-0.5+intermediate_eff,facecolor='g', alpha=0.3)
    plt.axvspan(matured_eff-0.5+intermediate_eff, matured_eff+intermediate_eff+young_eff, facecolor='g', alpha=0.1)
    ax=sns.pointplot(x='x', y='y', data= table, hue='Always_on_Frontier', scale=1, errwidth=1.5, capsize=0.2, markers=['x','o'], linestyles=' ', palette='dark')
    ax=sns.pointplot(x='x', y='x', data =table, scale=0.2, color='k', errwidth=0, capsize=0, linestyles='-')
    plt.title('Chemical firms on frontier by age')
    plt.axis('off')    
    return ax'''
def rearrange_eff_data_v2(eff_chem, eff_elec, eff_mac, n_chem= 197, n_elec=233, n_mac=305):
    '''reconstruct a dataframe to draw boxplot
       this is used to convey the comparison between 3 sectors '''
    df = pd.concat([eff_chem, eff_elec, eff_mac], axis = 0).iloc[:, 1:]
    eff_melt = pd.melt(df.T)
    #id_list = eff_chem.iloc[:,0]
    #arr_id = []
    #for i in id_list:
        #arr_id.append(np.repeat(i, 9))
    yrs =[[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],]
    arr_yrs = yrs * (n_chem+ n_elec+n_mac)
    #arr_id = pd.melt(pd.DataFrame(data= arr_id).T)
    arr_yrs = pd.melt(pd.DataFrame(data = arr_yrs).T)
    eff_melt = pd.concat([arr_yrs['value'], eff_melt['value']], axis = 1)
    eff_melt.columns = ['Year', 'Eff_score']
    arr_sector = [['chemical',]*n_chem*9, ['electronic',]*n_elec*9, ['machinery',]*n_mac*9]
    arr_sector = pd.melt(pd.DataFrame(data=arr_sector).T)
    arr_sector.replace('None', np.nan, inplace=True)
    arr_sector.dropna(inplace=True)
    arr_sector= arr_sector.reset_index()
    eff_melt['Sector'] = arr_sector['value']
    return eff_melt



def eff_distribution_OT_by_sector(eff_chem, eff_elec, eff_mac,  n_chem= 197, n_elec=233, n_mac=305,
                               title='Efficiency level across sectors over time'):
    '''boxplot of efficient for firm in different sectors over time'''
    eff_melt = rearrange_eff_data_v2(eff_chem, eff_elec, eff_mac)
    ax=plt.subplots(figsize=(10,8))
    ax=sns.boxplot(x='Year', y='Eff_score', data= eff_melt.dropna(), hue='Sector', palette='GnBu_d')
    plt.title(title, fontsize=20)
    ax.legend(loc='lower left')
    return ax

def eff_static(eff_chem_dmu, matured_eff=35, intermediate_eff= 7, young_eff=6, a='Chemical'):
    '''which firms are always on the frontier'''
    x = np.arange(1, eff_chem_dmu.shape[1], 1)
   
    geo_mean=[]
    for i in range(eff_chem_dmu.shape[1]-1):
        geo_mean.append(np.nanprod(eff_chem_dmu.iloc[:,i])**(1/eff_chem_dmu.iloc[:,i].notnull().sum()))
    geo_mean = pd.Series(data=geo_mean)
    table= pd.DataFrame(pd.concat([pd.Series(x), pd.Series(geo_mean)], axis = 1))
    table.columns=['x', 'geo_mean']
    table['y'] = table['geo_mean']
   
    eff = []
    for i in range(table.shape[0]):
        if table.loc[i,'geo_mean']==1:
            eff.append(1)
        else:
            eff.append(0)
    table['Always_on_Frontier'] = eff
   
    eff_time = []
    for i in range(eff_chem_dmu.shape[1]-1):
        eff_time.append(str(eff_chem_dmu[eff_chem_dmu.iloc[:,i]==1].shape[0]))
    table['eff_times'] = eff_time
    
    firm_id = eff_chem_dmu.columns
    table['firm_id'] = firm_id[:-1]
    fig, ax = plt.subplots(figsize=(10,10))
    plt.axvspan(-1, matured_eff-0.5, facecolor='g', alpha=0.5)
    plt.axvspan(matured_eff-0.5, matured_eff-0.5+intermediate_eff,facecolor='g', alpha=0.3)
    plt.axvspan(matured_eff-0.5+intermediate_eff, matured_eff+intermediate_eff+young_eff, facecolor='g', alpha=0.1)
    ax=sns.pointplot(x='x', y='y', data= table, hue='Always_on_Frontier', scale=1, errwidth=1.5, capsize=0.2, markers=['x','o'], linestyles=' ', palette='dark')
    plt.hlines(1, xmin=0, xmax= eff_chem_dmu.shape[1]-1, linestyles='dashed', label='technical efficiency frontier', colors='k')
    plt.title(f'{a} Firms on frontier by age')
    #[ax.text(p[0], p[1]+0.005, p[2], color='r') for p in zip(ax.get_xticks(),table['y'], table['eff_times'])]
    plt.xticks([])
    plt.xlabel('Firms by age')
    plt.ylabel('Efficiency level')
    return ax

def malm_by_efflevel(eff_chem, df):
    '''The distance to the frontier affects TFP growth in two ways. The further an firm lies behind 
    the frontier, the higher will be the TFP growth rates (TFPCH) and the respective component EFFCH 
    (indicating specifically catching-up), the smaller will be productivity growth through frontier-shifts'''
    malm_n = pd.DataFrame(data=[])
    malm_f = pd.DataFrame(data=[])
    for i in range(10, 18):
        j=i+1
        eff_chem[f'rank_{i}'] = eff_chem[f'eff_scores_{i}'].rank(ascending=False)
        firm_near = eff_chem.loc[eff_chem[f'rank_{i}'] <= eff_chem[f'rank_{i}'].median(), 'firm_id']
        firm_further = eff_chem.loc[eff_chem[f'rank_{i}'] > eff_chem[f'rank_{i}'].median(), 'firm_id']
        malm_near = df.loc[df['dmu'].isin(firm_near), [f'pc_{j}', f'tc_{j}', f'ec_{j}']]
        malm_n = pd.concat([malm_n, malm_near], axis = 1)
        malm_further = df.loc[df['dmu'].isin(firm_further), [f'pc_{j}', f'tc_{j}', f'ec_{j}']]
        malm_f = pd.concat([malm_f, malm_further], axis = 1)
    return malm_n, malm_f

from scipy import stats
def independent_ttest_malm(eff_chem, df):
    '''2 tailed indenpendent ttest for malmquist indices'''
    result = []
    malm_near, malm_further = malm_by_efflevel(eff_chem, df)
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
    result = result.loc[:,['pc_11','pc_12','pc_13','pc_14','pc_15','pc_16', 'pc_17','pc_18',
                    'tc_11','tc_12','tc_13','tc_14','tc_15','tc_16', 'tc_17','tc_18',
                    'ec_11','ec_12','ec_13','ec_14','ec_15','ec_16', 'ec_17','ec_18']]
    result=result.round(3)
    return result

def visualization_ttest_years(eff_chem, df, title = 'Malmquist indices differences for laggard and leader firms in chemical sector'):
    '''a bar graph to easily inspect the t-test to see whether it is over the thresshold 
    of p-value (0.05)'''
    t_test = independent_ttest_malm(eff_chem, df).iloc[0,:].T   
    ax=plt.subplots(figsize=(10,8))
    ax= t_test.plot(kind='bar', color='blue')
    ax.axhline(2.576, linestyle='--', color='grey', linewidth=2)
    ax.axhline(2.327, linestyle='--', color='red', linewidth=2)
    ax.axhline(1.96, linestyle='--', color='green', linewidth=2)

    ax.axhline(0, color='black', linewidth=2)
    ax.axhline(-2.576, linestyle='--', color='grey', linewidth=2)
    ax.axhline(-2.327, linestyle='--', color='red', linewidth=2)
    ax.axhline(-1.96, linestyle='--', color='green', linewidth=2)
    ax.axvline(7.5, linestyle='--',  linewidth=0.5)
    ax.axvline(15.5, linestyle='--', linewidth=0.5)
    plt.xlabel('Malmquist Indices')
    plt.ylabel('T-Statistic')
    plt.title(title)
    plt.show()
    return ax

def malm_ttest_period(eff_chem, df):
    '''over the study period'''
    malm_n, malm_f = malm_by_efflevel(eff_chem, df)
    pc_n = pd.melt(malm_n.loc[:,['pc_11', 'pc_12', 'pc_13', 'pc_14', 'pc_15',
                                 'pc_16', 'pc_17', 'pc_18']])
    pc_n=pc_n.iloc[:,1].dropna()
    
    pc_f = pd.melt(malm_f.loc[:,['pc_11', 'pc_12', 'pc_13', 'pc_14', 'pc_15',
                                 'pc_16', 'pc_17', 'pc_18']])
    pc_f=pc_f.iloc[:,1].dropna()
    
    pc_ttest = stats.ttest_ind(pc_f, pc_n)[0]


    tc_n = pd.melt(malm_n.loc[:,['tc_11', 'tc_12', 'tc_13', 'tc_14', 'tc_15',
                                 'tc_16', 'tc_17', 'tc_18']])
    tc_n=tc_n.iloc[:,1].dropna()
    
    tc_f = pd.melt(malm_f.loc[:,['tc_11', 'tc_12', 'tc_13', 'tc_14', 'tc_15',
                                 'tc_16', 'tc_17', 'tc_18']])
    tc_f=tc_f.iloc[:,1].dropna()
    
    tc_ttest = stats.ttest_ind(tc_f, tc_n)[0]


    ec_n = pd.melt(malm_n.loc[:,['ec_11', 'ec_12', 'ec_13', 'ec_14', 'ec_15',
                                 'ec_16', 'ec_17', 'ec_18']])
    ec_n=ec_n.iloc[:,1].dropna()
    
    ec_f = pd.melt(malm_f.loc[:,['ec_11', 'ec_12', 'ec_13', 'ec_14', 'ec_15',
                                 'ec_16', 'ec_17', 'ec_18']])
    ec_f=ec_f.iloc[:,1].dropna()
    
    ec_ttest = stats.ttest_ind(ec_f, ec_n)[0]
    
    ttest = [pc_ttest, tc_ttest, ec_ttest]
    
    ttest = pd.DataFrame(data=ttest)
    
    ttest.index = ['tfpch', 'techch', 'effch']
    return ttest

def malm_ttest_sectors(eff_chem, df_chem, eff_elec, df_elec, eff_mac, df_mac):
    
    '''t-statistic for pc, tc, ec of 3 sectors'''
    
    ttest_chem = malm_ttest_period(eff_chem, df_chem)
    ttest_elec = malm_ttest_period(eff_elec, df_elec)
    ttest_mac = malm_ttest_period(eff_mac, df_mac)
    ttest = pd.concat([ttest_chem, ttest_elec, ttest_mac], axis=0)
    ttest.columns= ['Values']
    ttest['sector'] =['Chemical', 'Chemical', 'Chemical',
                      'Electronic', 'Electronic', 'Electronic',
                      'Machinery', 'Machinery', 'Machinery']
    return ttest

def visualization_ttest_sectors(ttest):
    
    '''using entire data during the period
       for all 3 sectors'''
       
    ax= sns.barplot(x= ttest.index , y=ttest['Values'], hue=ttest['sector'])
    ax.axhline(1.645, linestyle='--', color='grey', linewidth=2)
    ax.axhline(0, color='black', linewidth=2)
    ax.axhline(-1.645, linestyle='--', color='grey', linewidth=2)
    plt.xlabel('Malmquist Indices')
    plt.ylabel('T-Statistic')
    plt.title('Malmquist indices differences for laggard and leader firms')
    plt.show()
    return ax

def data_malm_ttest_period(eff_chem, df):
    
    '''data contains all near and further firms in the period'''
    malm_n, malm_f, r = malm_by_efflevel(eff_chem, df)
    pc_n = pd.melt(malm_n.loc[:,['pc_11', 'pc_12', 'pc_13', 'pc_14', 'pc_15',
                                 'pc_16', 'pc_17', 'pc_18']])
    pc_n=pc_n.iloc[:,1].dropna()
    
    pc_f = pd.melt(malm_f.loc[:,['pc_11', 'pc_12', 'pc_13', 'pc_14', 'pc_15',
                                 'pc_16', 'pc_17', 'pc_18']])
    pc_f=pc_f.iloc[:,1].dropna()

    tc_n = pd.melt(malm_n.loc[:,['tc_11', 'tc_12', 'tc_13', 'tc_14', 'tc_15',
                                 'tc_16', 'tc_17', 'tc_18']])
    tc_n=tc_n.iloc[:,1].dropna()
    
    tc_f = pd.melt(malm_f.loc[:,['tc_11', 'tc_12', 'tc_13', 'tc_14', 'tc_15',
                                 'tc_16', 'tc_17', 'tc_18']])
    tc_f=tc_f.iloc[:,1].dropna()

    ec_n = pd.melt(malm_n.loc[:,['ec_11', 'ec_12', 'ec_13', 'ec_14', 'ec_15',
                                 'ec_16', 'ec_17', 'ec_18']])
    ec_n=ec_n.iloc[:,1].dropna()
    
    ec_f = pd.melt(malm_f.loc[:,['ec_11', 'ec_12', 'ec_13', 'ec_14', 'ec_15',
                                 'ec_16', 'ec_17', 'ec_18']])
    ec_f=ec_f.iloc[:,1].dropna()
    
    return pc_n, pc_f, tc_n, tc_f, ec_n, ec_f
def ttest_malmquist_1(df):
    
    """ttest if mean of pc <> 1"""
    
    arr = []
    for i in range(df.shape[1]):
        arr.append(stats.ttest_1samp(df.iloc[:,i],1, nan_policy='omit'))
    tstats = pd.DataFrame(data = arr)
    tstats.columns = ['t_stats', 'p_value']
    return tstats
        
def ttest_malmquist(df, n_matured, n_intermediate, n_young):
    
    '''run ttest for age group'''
    
    df_matured, df_intermediate, df_young = df_by_age(df, n_matured, n_intermediate, n_young)
    df_matured = df_matured.iloc[:, [1,4,7,10,13,16,19,22]]
    df_intermediate = df_intermediate.iloc[:, [1,4,7,10,13,16,19,22]]
    df_young = df_young.iloc[:, [1,4,7,10,13,16,19,22]]
    tstats_matured = ttest_malmquist_1(df_matured)
    tstats_intermediate = ttest_malmquist_1(df_intermediate)
    tstats_young = ttest_malmquist_1(df_young)
    yrs = pd.Series(data=[i for i in range(2011,2019)])
    tstats = pd.DataFrame(data=pd.concat([yrs, tstats_matured, tstats_intermediate, tstats_young], axis = 1))
    tstats.columns = ['Years', 't_Matured', 'p_Matured', 't_Intermediate', 'p_Intermediate', 't_Young', 'p_young']
    return tstats
