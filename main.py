#!/usr/bin/env python
# @Time    : 10/7/2021 22:11
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
import pandas as pd
import pickle
from CCHP import CCHP_Model
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
    # Creat model
    cchp = CCHP_Model()
    cchp.add_subsystem()
    cchp_sdp = SDP_Model()
    cchp_sdp.set_para_var(cchp)
    cchp_sdp.create_model()
    st_now, time_average = cchp_sdp.solve_sdp(cchp, markov_states, markov_transition, verbosity=0)
    # Record results
    record_cache(filename + '_detailed', cchp, cchp_sdp)
    record_performance(filename + '_overall', cchp, cchp_sdp, st_now, time_average)


if __name__ == '__main__':
    main('24hr_2h0e0c')
    # infile_cache = open('result/24hr_10h0e0c_detailed_cache', 'rb')
    # cache_detailed = pickle.load(infile_cache)
    # infile_cache.close()