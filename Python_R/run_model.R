####
setwd('D:/Uni Jena/data_thesis/data_running_R_backwardclean')
library(data.table)
library(deaR)
chem_total <- read.csv('mac_malm.csv', sep = '|')
chem_total[is.na(chem_total)] <- 0
# data for eff scores
chem_18 <- read_data(chem_total[chem_total$year==2018, c('firm', "fixed_assets", "employees", "mat_cost", "sales")], ni=3, no=1)
chem_17 <- read_data(chem_total[chem_total$year==2017, c('firm', "fixed_assets", "employees", "mat_cost", "sales")], ni=3, no=1)
chem_16 <- read_data(chem_total[chem_total$year==2016, c('firm', "fixed_assets", "employees", "mat_cost", "sales")], ni=3, no=1)
chem_15 <- read_data(chem_total[chem_total$year==2015, c('firm', "fixed_assets", "employees", "mat_cost", "sales")], ni=3, no=1)
chem_14 <- read_data(chem_total[chem_total$year==2014, c('firm', "fixed_assets", "employees", "mat_cost", "sales")], ni=3, no=1)
chem_13 <- read_data(chem_total[chem_total$year==2013, c('firm', "fixed_assets", "employees", "mat_cost", "sales")], ni=3, no=1)
chem_12 <- read_data(chem_total[chem_total$year==2012, c('firm', "fixed_assets", "employees", "mat_cost", "sales")], ni=3, no=1)
chem_11 <- read_data(chem_total[chem_total$year==2011, c('firm', "fixed_assets", "employees", "mat_cost", "sales")], ni=3, no=1)
chem_10 <- read_data(chem_total[chem_total$year==2010, c('firm', "fixed_assets", "employees", "mat_cost", "sales")], ni=3, no=1)

chem_18 <- read_data(chem_total[chem_total$year==2018, c('firm', "fixed_assets", "emp_cost", "mat_cost", "sales")], ni=3, no=1)
chem_17 <- read_data(chem_total[chem_total$year==2017, c('firm', "fixed_assets", "emp_cost", "mat_cost", "sales")], ni=3, no=1)
chem_16 <- read_data(chem_total[chem_total$year==2016, c('firm', "fixed_assets", "emp_cost", "mat_cost", "sales")], ni=3, no=1)
chem_15 <- read_data(chem_total[chem_total$year==2015, c('firm', "fixed_assets", "emp_cost", "mat_cost", "sales")], ni=3, no=1)
chem_14 <- read_data(chem_total[chem_total$year==2014, c('firm', "fixed_assets", "emp_cost", "mat_cost", "sales")], ni=3, no=1)
chem_13 <- read_data(chem_total[chem_total$year==2013, c('firm', "fixed_assets", "emp_cost", "mat_cost", "sales")], ni=3, no=1)
chem_12 <- read_data(chem_total[chem_total$year==2012, c('firm', "fixed_assets", "emp_cost", "mat_cost", "sales")], ni=3, no=1)
chem_11 <- read_data(chem_total[chem_total$year==2011, c('firm', "fixed_assets", "emp_cost", "mat_cost", "sales")], ni=3, no=1)
chem_10 <- read_data(chem_total[chem_total$year==2010, c('firm', "fixed_assets", "emp_cost", "mat_cost", "sales")], ni=3, no=1)


# costant return to scale
eff_18 <- model_basic(chem_18, orientation = 'io',rts = 'crs')
eff_scores_18 <- data.frame(efficiencies(eff_18))
eff_17 <- model_basic(chem_17, orientation = 'io',rts = 'crs')
eff_scores_17 <- data.frame(efficiencies(eff_17))
eff_16 <- model_basic(chem_16, orientation = 'io',rts = 'crs')
eff_scores_16 <- data.frame(efficiencies(eff_16))
eff_15 <- model_basic(chem_15, orientation = 'io',rts = 'crs')
eff_scores_15 <- data.frame(efficiencies(eff_15))
eff_14 <- model_basic(chem_14, orientation = 'io',rts = 'crs')
eff_scores_14 <- data.frame(efficiencies(eff_14))
eff_13 <- model_basic(chem_13, orientation = 'io',rts = 'crs')
eff_scores_13 <- data.frame(efficiencies(eff_13))
eff_12 <- model_basic(chem_12, orientation = 'io',rts = 'crs')
eff_scores_12 <- data.frame(efficiencies(eff_12))
eff_11 <- model_basic(chem_11, orientation = 'io',rts = 'crs')
eff_scores_11 <- data.frame(efficiencies(eff_11))
eff_10 <- model_basic(chem_10, orientation = 'io',rts = 'crs')
eff_scores_10 <- data.frame(efficiencies(eff_10))


#variable return to scale
eff_18 <- model_basic(chem_18, orientation = 'io',rts = 'vrs')
eff_scores_18 <- data.frame(efficiencies(eff_18))
eff_17 <- model_basic(chem_17, orientation = 'io',rts = 'vrs')
eff_scores_17 <- data.frame(efficiencies(eff_17))
eff_16 <- model_basic(chem_16, orientation = 'io',rts = 'vrs')
eff_scores_16 <- data.frame(efficiencies(eff_16))
eff_15 <- model_basic(chem_15, orientation = 'io',rts = 'vrs')
eff_scores_15 <- data.frame(efficiencies(eff_15))
eff_14 <- model_basic(chem_14, orientation = 'io',rts = 'vrs')
eff_scores_14 <- data.frame(efficiencies(eff_14))
eff_13 <- model_basic(chem_13, orientation = 'io',rts = 'vrs')
eff_scores_13 <- data.frame(efficiencies(eff_13))
eff_12 <- model_basic(chem_12, orientation = 'io',rts = 'vrs')
eff_scores_12 <- data.frame(efficiencies(eff_12))
eff_11 <- model_basic(chem_11, orientation = 'io',rts = 'vrs')
eff_scores_11 <- data.frame(efficiencies(eff_11))
eff_10 <- model_basic(chem_10, orientation = 'io',rts = 'vrs')
eff_scores_10 <- data.frame(efficiencies(eff_10))

#dataframe of efficiency score result
dmu <- chem_total$firm[chem_total$year==2018]
eff_score_chem <- data.frame(cbind(dmu, eff_scores_10, eff_scores_11, eff_scores_12, eff_scores_13, eff_scores_14,
                                   eff_scores_15, eff_scores_16, eff_scores_17, eff_scores_18))
names(eff_score_chem) <- c('firm_id', 'eff_scores_10', 'eff_scores_11', 'eff_scores_12', 'eff_scores_13', 'eff_scores_14',
                           'eff_scores_15', 'eff_scores_16', 'eff_scores_17', 'eff_scores_18' )
write.csv(eff_score_chem,file = 'eff_score_mac.csv')





# data for malmquist index
chemmal <- read_malmquist(chem_total[c('firm', 'year', 'fixed_assets', 'employees', 'mat_cost', 'sales')],percol = 2, arrangement = 'vertical', inputs=3:5, outputs=6)

chemmal <- read_malmquist(chem_total[c('firm', 'year', 'fixed_assets', 'emp_cost', 'mat_cost', 'sales')],percol = 2, arrangement = 'vertical', inputs=3:5, outputs=6)

#constant return to scale
malmquist_chem <- malmquist_index(chemmal, orientation = 'io', rts = 'crs')

#set tc_vrs=True
malmquist_chem <- malmquist_index(chemmal, orientation = 'io', rts = 'crs', tc_vrs = TRUE)

chem_ec <- data.frame(t(malmquist_chem$ec))
names(chem_ec) <- c('ec_11','ec_12','ec_13','ec_14','ec_15','ec_16','ec_17','ec_18')

chem_pc <- data.frame(t(malmquist_chem$mi))
names(chem_pc) <- c('pc_11','pc_12','pc_13','pc_14','pc_15','pc_16','pc_17','pc_18')
chem_tc <- data.frame(t(malmquist_chem$tc))
names(chem_tc) <- c('tc_11','tc_12','tc_13','tc_14','tc_15','tc_16','tc_17','tc_18')
malmindex_chem <- data.frame(cbind(dmu,chem_pc,chem_ec,chem_tc))
malmindex_chem <- malmindex_chem[,c('dmu','pc_11','ec_11','tc_11',
                                    'pc_12','ec_12','tc_12',
                                    'pc_13','ec_13','tc_13',
                                    'pc_14','ec_14','tc_14',
                                    'pc_15','ec_15','tc_15',
                                    'pc_16','ec_16','tc_16',
                                    'pc_17','ec_17','tc_17',
                                    'pc_18','ec_18','tc_18')]


#pec, sec
malmquist_chem <- malmquist_index(chemmal, orientation = 'io', rts = 'vrs')

chem_pc <- data.frame(t(malmquist_chem$mi))
names(chem_pc) <- c('pc_11','pc_12','pc_13','pc_14','pc_15','pc_16','pc_17','pc_18')

chem_pec <- data.frame(t(malmquist_chem$pech))
names(chem_pec) <- c('pec_11','pec_12','pec_13','pec_14','pec_15','pec_16','pec_17','pec_18')

chem_tc <- data.frame(t(malmquist_chem$tc))
names(chem_tc) <- c('tc_11','tc_12','tc_13','tc_14','tc_15','tc_16','tc_17','tc_18')

chem_sec <- data.frame(t(malmquist_chem$sech))
names(chem_sec) <- c('sec_11','sec_12','sec_13','sec_14','sec_15','sec_16','sec_17','sec_18')

malmindex_chem <- data.frame(cbind(dmu, chem_pc, chem_pec, chem_tc, chem_sec))
malmindex_chem <- malmindex_chem[,c('dmu','pc_11','pec_11','tc_11','sec_11',
                                    'pc_12','pec_12','tc_12','sec_12',
                                    'pc_13','pec_13','tc_13','sec_13',
                                    'pc_14','pec_14','tc_14','sec_14',
                                    'pc_15','pec_15','tc_15','sec_15',
                                    'pc_16','pec_16','tc_16','sec_16',
                                    'pc_17','pec_17','tc_17','sec_17',
                                    'pc_18','pec_18','tc_18','sec_18')]

##############
write.csv(malmindex_chem, file = 'malmquist_tovrs_mac.csv')
write.csv(malmindex_chem, file = 'malmquist_vrs_mac.csv')
