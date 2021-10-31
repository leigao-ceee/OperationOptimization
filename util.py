#!/usr/bin/env python
# @Time    : 10/10/2021 18:56
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
import os
import pickle


def record_cache(filename, cchp_model, sdp_model):
    outfile_cost = open('result/' + filename + '_cost', 'wb')
    pickle.dump(sdp_model.sdp.cost, outfile_cost)
    outfile_cost.close()
    outfile_operation = open('result/' + filename + '_cache', 'wb')
    system_cache = {}
    for sys in cchp_model.conv_system:
        system_cache.update({sys.name + str(sys.idx) + '_beta': sys.cache_beta})
        system_cache.update({sys.name + str(sys.idx) + '_alpha': sys.cache_alpha})
    for sys in cchp_model.strg_system:
        system_cache.update({sys.name + str(sys.idx): sys.cache_state})
    pickle.dump(system_cache, outfile_operation)
    outfile_operation.close()
    print('Finish record cache data')


def record_performance(file_in, cchp_model, sdp_model, st_now, time_count):
    if os.path.isfile('result/' + file_in):
        print("Output file for parameter already exists.")
        file_temp = 'result/' + file_in + ".old"
        print("Existing file has been copied to:", file_temp)
    file = open('result/' + file_in + '.txt', 'w')
    file.write("%%%%%%%%  Thermal System Information  %%%%%%%% \n")
    for sys in cchp_model.conv_system:
        file.write(sys.name + str(sys.idx) + ' capacity: {0}, ramp : {1} '.format(sys.capacity, sys.ramp) + '\n')
    file.write("%%%%%%%%  Storage System Information  %%%%%%%% \n")
    for sys in cchp_model.strg_system:
        file.write(sys.name + str(sys.idx) + ' capacity: {0}, ramp: {1}.'.format(sys.capacity, sys.ramp) + '\n')
    file.write("%%%%%%%%      Solver Information      %%%%%%%% \n")

    file.write("%%%%%%%%   Performance Information    %%%%%%%% \n")
    file.write('The first stage cost is {0} '.format(sdp_model.sdp.cost[0, 0][st_now]) + '\n')
    file.write('Average time consumption is {0} '.format(time_count) + '\n')
    file.close()
    print('Finish record performance data')