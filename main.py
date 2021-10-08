#!/usr/bin/env python
# @Time    : 10/7/2021 22:11
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
import pandas as pd
import pickle
from CCHP_SDP import SDP_Model
# demands into dataframe
infile_states = open('sampling/markov_states', 'rb')
markov_states = pickle.load(infile_states)
infile_states.close()
infile_transition = open('sampling/markov_transition', 'rb')
markov_transition = pickle.load(infile_transition)
infile_transition.close()

path = r'sampling\cchp_data.xlsx'
df_HD = pd.read_excel(path, sheet_name='heating_demand', header=0, index_col=0)
df_CD = pd.read_excel(path, sheet_name='heating_demand', header=0, index_col=0)
df_ED = pd.read_excel(path, sheet_name='electricity_demand', header=0, index_col=0)
df_EP = pd.read_excel(path, sheet_name='electricity_price', header=0, index_col=0)
df_Temp = pd.read_excel(path, sheet_name='temperature', header=0, index_col=0)
df_Sc = pd.read_excel(path, sheet_name='possibility', header=0, index_col=0)
df_StHt = pd.read_excel(path, sheet_name='storage_heating', header=0, index_col=0)
df_StEl = pd.read_excel(path, sheet_name='storage_electricity', header=0, index_col=0)

cchp_sdp = SDP_Model()
# cchp_sdp.add_subsystem()
cchp_sdp.set_para_var()
cchp_sdp.create_model()
# cchp_sdp.sdp.pprint()
cchp_sdp.solve_sdp(markov_states, markov_transition, df_Temp, df_EP, verbosity=0)

gen_elec = {'PM': 6.0, 'ORC': 2.5}
gen_heat = {'ABH': 5, 'EH': 5, 'Bo': 5}
con_fuel = ['PM', 'Bo']
con_waste = ['ORC', 'ABH']
storage = {'El': 5, 'Ht': 5}
cchp = SDP_Model(gen_elec, gen_heat, con_fuel, con_waste, storage)