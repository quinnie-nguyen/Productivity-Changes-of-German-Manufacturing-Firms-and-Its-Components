# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:17:54 2020

@author: overall result
"""
import pandas as pd
import numpy as np
import Preprocessing_data as preda
import matplotlib.pyplot as plt
import seaborn as sns

'''Report for DEA model'''

title_list = ['eff_score_chem.csv', 'eff_score_elec.csv', 'eff_score_mac.csv']
eff_chem, eff_elec, eff_mac = [preda.read_eff_score(title) for title in title_list]

sales = ['sales_chem.csv', 'sales_elec.csv', 'sales_mac.csv']
sales_chem, sales_elec, sales_mac = [pd.read_csv(title, sep ='|').iloc[:,1:] for title in sales]


eff_chem_dmu = preda.eff_dmu(eff_chem)
eff_elec_dmu = preda.eff_dmu(eff_elec)
eff_mac_dmu = preda.eff_dmu(eff_mac)

"""with geometric average"""
summary_chem = preda.efficiency_sum_stats(eff_chem)
summary_elec = preda.efficiency_sum_stats(eff_elec)
summary_mac = preda.efficiency_sum_stats(eff_mac)

"""with weighted average"""
summary_chem = preda.efficiency_sum_stats_weighted(eff_chem, sales_chem)
summary_elec = preda.efficiency_sum_stats_weighted(eff_elec, sales_elec)
summary_mac = preda.efficiency_sum_stats_weighted(eff_mac, sales_mac)

preda.eff_distribution_OT(eff_chem, summary_chem, 140, 36, 21,
                          title =' Efficiency level of chemical sector over time')
preda.eff_distribution_OT(eff_elec, summary_elec, 175, 41, 17,
                          title =' Efficiency level of electronic sector over time')
preda.eff_distribution_OT(eff_mac, summary_mac, 241, 41, 23,
                          title =' Efficiency level of machinery sector over time')
preda.eff_distribution_OT_by_sector(eff_chem, eff_elec, eff_mac, 198, 237, 307)


"""for RC"""
preda.eff_distribution_OT(eff_chem, summary_chem, 140, 36, 21,
                          title =' Efficiency level of chemical sector over time_RC')
preda.eff_distribution_OT(eff_elec, summary_elec, 175, 41, 17,
                          title =' Efficiency level of electronic sector over time_RC')
preda.eff_distribution_OT(eff_mac, summary_mac, 241, 41, 23,
                          title =' Efficiency level of machinery sector over time_RC')
""""""

preda.eff_distribution_OT_by_age(eff_chem, 140,36,21,
                                 title ='Efficiency level of chemical firms by age over time')
preda.eff_distribution_OT_by_age(eff_elec, 175, 41, 17,
                                 title ='Efficiency level of electronic firms by age over time')
preda.eff_distribution_OT_by_age(eff_mac, 241, 41, 23,
                                 title ='Efficiency level of machinery firms by age over time')

"""with geometric mean"""
summary_by_age_chem = preda.eff_stats_by_age_merged(eff_chem, 140, 36, 21)
summary_by_age_elec = preda.eff_stats_by_age_merged(eff_elec, 175, 41, 17)
summary_by_age_mac = preda.eff_stats_by_age_merged(eff_mac, 241, 41, 23)

"""with weighted average"""
summary_by_age_chem = preda.eff_stats_by_age_merged_weighted(eff_chem, sales_chem, 140, 36, 21)
summary_by_age_elec = preda.eff_stats_by_age_merged_weighted(eff_elec, sales_elec, 175, 41, 17)
summary_by_age_mac = preda.eff_stats_by_age_merged_weighted(eff_mac, sales_mac, 241, 41, 23)

data_viz_eff_by_age = pd.DataFrame(data=[])
data_viz_eff_by_age['year'] = summary_by_age_chem['year']
data_viz_eff_by_age['chemical'] = summary_chem['Mean']
data_viz_eff_by_age['electronic'] = summary_elec['Mean']
data_viz_eff_by_age['machinery'] = summary_mac['Mean']
data_viz_eff_by_age['matured firmc(c)'] = summary_by_age_chem['Mean_old']
data_viz_eff_by_age['intermediate firm (c)'] = summary_by_age_chem['Mean_young']
data_viz_eff_by_age['young firm (c)'] = summary_by_age_chem['Mean_newborn']
data_viz_eff_by_age['matured firm (e)'] = summary_by_age_elec['Mean_old']
data_viz_eff_by_age['intermediate firm (e)'] = summary_by_age_elec['Mean_young']
data_viz_eff_by_age['young firm (e)'] = summary_by_age_elec['Mean_newborn']
data_viz_eff_by_age['matured firm (m)'] = summary_by_age_mac['Mean_old']
data_viz_eff_by_age['intermediate firm (m)'] = summary_by_age_mac['Mean_young']
data_viz_eff_by_age['young firm (m)'] = summary_by_age_mac['Mean_newborn']
data_viz_eff_by_age.replace(1, np.nan, inplace=True)
preda.visualize_clustered_toward_eff(data_viz_eff_by_age, 'among sectors')
preda.visualize_clustered_toward_eff(data_viz_eff_by_age, 'chemical')
preda.visualize_clustered_toward_eff(data_viz_eff_by_age, 'electronic')
preda.visualize_clustered_toward_eff(data_viz_eff_by_age, 'machinery')

from time import strftime 
writer = pd.ExcelWriter(strftime('Report DEA %Y-%m-%d.xlsx'))

eff_chem_dmu.to_excel(writer, 'eff_firms_chem')
eff_elec_dmu.to_excel(writer, 'eff_firms_elec')
eff_mac_dmu.to_excel(writer, 'eff_firms_mac')

summary_chem.to_excel(writer, 'eff_summary_stats_chem')
summary_elec.to_excel(writer, 'eff_summary_stats_elec')
summary_mac.to_excel(writer, 'eff_summary_stats_mac')

summary_by_age_chem.to_excel(writer, 'eff_score_by_age_chem')
summary_by_age_elec.to_excel(writer, 'eff_score_by_age_elec')
summary_by_age_mac.to_excel(writer, 'eff_score_by_age_mac')

#dataframe contains all eff_scores of 3 sectors
manufacturing = pd.concat([eff_chem, eff_elec, eff_mac], axis = 0)
manufacturing.drop('firm_id', axis = 1, inplace=True)

#boxplot of eff_score across sectors
preda.eff_distribution_OT_by_sector(manufacturing)

#manufacturing.to_excel(writer, 'eff_overall')
writer.save()

'''Report Malmquist index (df)'''

df_titles = [['malmquist_vrs_chem.csv', 140, 36, 21], 
           ['malmquist_vrs_elec.csv', 175, 41, 17],
           ['malmquist_vrs_mac.csv', 241, 41, 23]]
'''dataframe including pure technical efficiency change and scale efficiency change'''
df_chem_f, df_elec_f, df_mac_f = [preda.read_malmquist(title, n_matured, n_intermediate, n_young) 
                            for title, n_matured, n_intermediate, n_young in df_titles]
df_chem, df_elec, df_mac = [preda.read_malmquist(title, n_matured, n_intermediate, n_young) 
                            for title, n_matured, n_intermediate, n_young in df_titles]

'''only PTCH and TCCH'''
for i in range(11,19):
    df_chem = df_chem.drop(f'sec_{i}', axis = 1)
    df_elec = df_elec.drop(f'sec_{i}', axis = 1)
    df_mac = df_mac.drop(f'sec_{i}', axis = 1)

names = ['dmu',
         'pc_11', 'ec_11', 'tc_11',
         'pc_12', 'ec_12', 'tc_12',
         'pc_13', 'ec_13', 'tc_13',
         'pc_14', 'ec_14', 'tc_14',
         'pc_15', 'ec_15', 'tc_15',
         'pc_16', 'ec_16', 'tc_16',
         'pc_17', 'ec_17', 'tc_17',
         'pc_18', 'ec_18', 'tc_18', 'age']
df_chem.columns = names
df_elec.columns = names
df_mac.columns = names
'''ids of firm that grow during 2010-2018'''

preda.growth_dmu(df_chem,197)
preda.growth_dmu(df_elec,233)
preda.growth_dmu(df_mac,305)


df_list = [[df_chem, 140, 36, 21],
           [df_elec, 175, 41, 17],
           [df_mac, 241, 41, 23]]


'''growth of productivity of firms '''

productivity_growth_chem = preda.prod_growth_1018(df_chem, 140, 36, 21)
young_chem = [0,5,9,12,17,21,21,21]
productivity_growth_chem['%ygrowth'] = productivity_growth_chem['young_growth'] / young_chem

productivity_growth_elec = preda.prod_growth_1018(df_elec, 175, 41, 17)
young_elec = [1,2,7,10,13,17,17,17]
productivity_growth_elec['%ygrowth'] = productivity_growth_elec['young_growth'] / young_elec

productivity_growth_mac = preda.prod_growth_1018(df_mac, 241, 41, 23)
young_mac = [1,2,7,10,18,23,23,23]
productivity_growth_mac['%ygrowth'] = productivity_growth_mac['young_growth'] / young_mac

'''the number and percentage of firms that have productivity growth mainly or only from
efficiency change'''

ec_driven_chem = preda.source_pc_sector_ec(df_chem_f, 140, 36, 21)
ec_driven_elec = preda.source_pc_sector_ec(df_elec_f, 175, 41, 17)
ec_driven_mac = preda.source_pc_sector_ec(df_mac_f, 241, 41, 23)


from time import strftime 
writer_m = pd.ExcelWriter(strftime('Report MALMQUIST %Y-%m-%d.xlsx'))
productivity_growth_chem.to_excel(writer_m, 'prod_growth_chem')
productivity_growth_elec.to_excel(writer_m, 'prod_growth_elec')
productivity_growth_mac.to_excel(writer_m, 'prod_growth_mac')

ec_driven_chem.to_excel(writer_m, 'ec_driven_chem')
ec_driven_elec.to_excel(writer_m, 'ec_driven_elec')
ec_driven_mac.to_excel(writer_m, 'ec_driven_mac')

tc_driven_chem = preda.source_pc_sector_tc(df_chem, 140, 36, 21)
tc_driven_elec = preda.source_pc_sector_tc(df_elec, 175, 41, 17)
tc_driven_mac = preda.source_pc_sector_tc(df_mac, 241, 41, 23)

tc_driven_chem.to_excel(writer_m, 'tc_driven_chem')
tc_driven_elec.to_excel(writer_m, 'tc_driven_elec')
tc_driven_mac.to_excel(writer_m, 'tc_driven_mac')



"""one-tailed t-test to see whether there is significant difference in
pc, tc, ec of laggard and leader firms
    grey line: p_value = 0.005
    red line: p_value = 0.01
    green: p_value = 0.025
    using data each year seperately"""

preda.visualization_ttest_years(eff_chem, df_chem, 'Malmquist indices differences for laggard and leader firms in chemical sector')

preda.visualization_ttest_years(eff_elec, df_elec, 'Malmquist indices differences for laggard and leader firms in electronic sector')

preda.visualization_ttest_years(eff_mac, df_mac, 'Malmquist indices differences for laggard and leader firms in machinery sector')

ttest = preda.malm_ttest_sectors(eff_chem, df_chem, eff_elec, df_elec, eff_mac, df_mac)

"""one-tailed t-test for the same purpose, but using the entire data 
during the period rather than do ttest for each year seperately"""

preda.visualization_ttest_sectors(ttest)

"""how do Malmquist index and its components change over time?"""

'''with geometric mean'''
malm_change_chem = preda.avg_change_bygroup(df_chem, 140, 36, 21)
malm_change_chem.replace(0,np.nan,inplace=True)
malm_change_elec = preda.avg_change_bygroup(df_elec, 175, 41, 17)
malm_change_mac = preda.avg_change_bygroup(df_mac, 241, 41, 23)

'''with weighted average'''
malm_change_chem = preda.weighted_avg_change_bygroup(df_chem_f, sales_chem, 140, 36, 21)
malm_change_elec = preda.weighted_avg_change_bygroup(df_elec_f, sales_elec, 175, 41, 17)
malm_change_mac = preda.weighted_avg_change_bygroup(df_mac_f, sales_mac, 241, 41, 23)


preda.visualize_change_by_group(malm_change_chem, 'young', 'The evolution in MI, EC, and TC during 2011-2018 of young firms in chemical sector')
preda.visualize_change_by_group(malm_change_chem, 'intermediate', 'The evolution in MI, EC, and TC during 2011-2018 of intermediate firms in chemical sector')
preda.visualize_change_by_group(malm_change_chem, 'matured', 'The evolution in MI, EC, and TC during 2011-2018 of matured firms in chemical sector')

preda.visualize_change_by_component(malm_change_chem, "MI", "Total factor productivity change of chemical sector (2010-2018)")
preda.visualize_change_by_component(malm_change_chem, "EC")
preda.visualize_change_by_component(malm_change_chem, "TC")

preda.visualize_change_by_group(malm_change_elec, 'young', 'The evolution in MI, EC, and TC during 2011-2018 of young firms in electronic sector')
preda.visualize_change_by_group(malm_change_elec, 'intermediate', 'The evolution in MI, EC, and TC during 2011-2018 of intermediate firms in electronic sector')
preda.visualize_change_by_group(malm_change_elec, 'matured', 'The evolution in MI, EC, and TC during 2011-2018 of matured firms in electronic sector')

preda.visualize_change_by_component(malm_change_elec, "MI", "Total factor productivity change of electronic sector (2010-2018)")
preda.visualize_change_by_component(malm_change_elec, "EC")
preda.visualize_change_by_component(malm_change_elec, "TC")


preda.visualize_change_by_group(malm_change_mac, 'young', 'The evolution in MI, EC, and TC during 2011-2018 of young firms in machinery sector')
preda.visualize_change_by_group(malm_change_mac, 'intermediate', 'The evolution in MI, EC, and TC during 2011-2018 of intermediate firms in machinery sector')
preda.visualize_change_by_group(malm_change_mac, 'matured', 'The evolution in MI, EC, and TC during 2011-2018 of matured firms in machinery sector')


preda.visualize_change_by_component(malm_change_mac, "MI", "Total factor productivity change of machinery sector (2010-2018)")
preda.visualize_change_by_component(malm_change_mac, "EC")
preda.visualize_change_by_component(malm_change_mac, "TC")


"""significant or insignificant of non-zero pc"""

tstast_chem = preda.ttest_malmquist(df_chem, 140, 36, 21)
tstast_elec = preda.ttest_malmquist(df_elec, 175, 41, 17)
tstast_mac = preda.ttest_malmquist(df_mac, 241, 41, 23)


from time import strftime 
writer_m = pd.ExcelWriter(strftime('Report MALMQUIST %Y-%m-%d.xlsx'))
productivity_growth_chem.to_excel(writer_m, 'prod_growth_chem')
productivity_growth_elec.to_excel(writer_m, 'prod_growth_elec')
productivity_growth_mac.to_excel(writer_m, 'prod_growth_mac')

ec_driven_chem.to_excel(writer_m, 'ec_driven_chem')
ec_driven_elec.to_excel(writer_m, 'ec_driven_elec')
ec_driven_mac.to_excel(writer_m, 'ec_driven_mac')

tc_driven_chem = preda.source_pc_sector_tc(df_chem, 140, 36, 21)
tc_driven_elec = preda.source_pc_sector_tc(df_elec, 175, 41, 17)
tc_driven_mac = preda.source_pc_sector_tc(df_mac, 241, 41, 23)

tc_driven_chem.to_excel(writer_m, 'tc_driven_chem')
tc_driven_elec.to_excel(writer_m, 'tc_driven_elec')
tc_driven_mac.to_excel(writer_m, 'tc_driven_mac')

malm_change_chem.to_excel(writer_m, 'malm_change_chem')
malm_change_elec.to_excel(writer_m, 'malm_change_elec')
malm_change_mac.to_excel(writer_m, 'malm_change_mac')

tstast_chem.to_excel(writer_m, 'tfpc_ttest_chem')
tstast_elec.to_excel(writer_m, 'tfpc_ttest_elec')
tstast_mac.to_excel(writer_m, 'tfpc_ttest_mac')

writer_m.save()
