#!/usr/bin/env python
# @Time    : 10/10/2021 18:56
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
import os
import pickle


def record_cache(filename, model):
    outfile_cost = open('result/' + filename + '_cost', 'wb')
    pickle.dump(model.sdp.cost, outfile_cost)
    outfile_cost.close()
    outfile_operation = open('result/' + filename + '_cache', 'wb')
    system_cache = {}
    for sys in model.conv_system:
        system_cache.update({sys.name + str(sys.idx) + '_beta': sys.cache_beta})
        system_cache.update({sys.name + str(sys.idx) + '_alpha': sys.cache_alpha})
    for sys in model.strg_system:
        system_cache.update({sys.name + str(sys.idx) + '_beta': sys.cache_state})
    pickle.dump(system_cache, outfile_operation)
    outfile_operation.close()
    print('Finish record cache data')


def record_performance(file_in, model, st_now):
    if os.path.isfile(file_in):
        print("Output file for parameter already exists.")
        file_temp = file_in + ".old"
        print("Existing file has been copied to:", file_temp)
    file = open(file_in, 'w')
    file.write("%%%%%%%%  Thermal System Information  %%%%%%%% \n")
    for sys in model.conv_system:
        file.write(sys.name + str(sys.idx) + ' capacity = ' + str(sys.capacity) + '\n')
    file.write("%%%%%%%%  Storage System Information  %%%%%%%% \n")
    for sys in model.strg_system:
        file.write(sys.name + str(sys.idx) + ' capacity: {0} and ramp: {1}.'.format(sys.capacity, sys.capacity) + '\n')
    file.write("%%%%%%%%      Solver Information      %%%%%%%% \n")

    file.write("%%%%%%%%   Performance Information    %%%%%%%% \n")
    file.write('The first stage cost is '.format(model.sdp.cost[0, 0][st_now]) + '\n')
    file.close()
    print('Finish record performance data')