#!/usr/bin/env python
# @Time    : 10/7/2021 22:11
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
import pandas as pd
import pickle
from CCHP_SDP import SDP_Model
from util import record_cache, record_performance


def main(filename):
    # Load demands
    infile_states = open('sampling/markov_states', 'rb')
    markov_states = pickle.load(infile_states)
    infile_states.close()
    infile_transition = open('sampling/markov_transition', 'rb')
    markov_transition = pickle.load(infile_transition)
    infile_transition.close()
    df_EP = pd.read_excel('sampling/cchp_data.xlsx', sheet_name='electricity_price', header=0, index_col=0)
    # df_Temp = pd.read_excel('sampling/chp_data.xlsx', sheet_name='temperature', header=0, index_col=0)
    # Creat model
    cchp_sdp = SDP_Model()
    cchp_sdp.set_para_var()
    cchp_sdp.create_model()
    st_now = cchp_sdp.solve_sdp(markov_states, markov_transition, df_EP, verbosity=0)
    # Record results
    record_cache(filename + '_detailed', cchp_sdp)
    record_performance(filename + '_overall', cchp_sdp, st_now)


if __name__ == '__main__':
    # main('24hr_2h0e0c')
    infile_cache = open('result/24hr_2h0e0c_detailed_cost', 'rb')
    cache_detailed = pickle.load(infile_cache)
    infile_cache.close()