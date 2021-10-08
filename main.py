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
df_EP = pd.read_excel(path, sheet_name='electricity_price', header=0, index_col=0)
# df_Temp = pd.read_excel(path, sheet_name='temperature', header=0, index_col=0)

cchp_sdp = SDP_Model()
cchp_sdp.set_para_var()
cchp_sdp.create_model()
cchp_sdp.solve_sdp(markov_states, markov_transition, df_EP, verbosity=0)
print('end')