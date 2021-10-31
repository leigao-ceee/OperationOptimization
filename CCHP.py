#!/usr/bin/env python
# @Time    : 10/31/2021 12:35
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
import numpy as np


class ThermalSystem(object):
    instances = []

    def __init__(self, name=None, idx=0, capacity=None, ramp=None, pdata=None, qdata=None):
        """
        :param
        pdata: system performance data list six parameters ax^2+by^2+cxy+dx+ey+f

        """
        self.__class__.instances.append(self)
        self.name = name
        self.idx = idx
        self.capacity = capacity
        self.ramp = ramp
        self.pdata = pdata
        self.qdata = qdata
        self.cache_beta = None
        self.cache_alpha = None

    def generate_cache(self, stage, storages, scen):
        self.cache_beta = np.zeros([stage, scen])
        self.cache_alpha = np.zeros([stage, scen])
        for i in range(len(storages)):
            self.cache_beta = np.expand_dims(self.cache_beta, axis=-1)
            self.cache_beta = np.repeat(self.cache_beta, len(storages[i]), axis=-1)
            self.cache_alpha = np.expand_dims(self.cache_alpha, axis=-1)
            self.cache_alpha = np.repeat(self.cache_alpha, len(storages[i]), axis=-1)


class StorageSystem(object):
    instances = []

    def __init__(self, name=None, idx=0, capacity=None, pdata=None, ramp=None):
        """
        :param
        pdata: system performance data list six parameters ax^2+by^2+cxy+dx+ey+f

        """
        self.__class__.instances.append(self)
        self.name = name
        self.idx = idx
        self.capacity = capacity
        self.ramp = ramp
        self.pdata = pdata
        self.state = list(range(ramp))
        self.cache_state = None

    def generate_cache(self, stage, storages, scen):
        self.cache_state = np.zeros([stage, scen])
        for i in range(len(storages)):
            self.cache_state = np.expand_dims(self.cache_state, axis=-1)
            self.cache_state = np.repeat(self.cache_state, len(storages[i]), axis=-1)


class CCHP_Model(object):
    def __init__(self):
        # energy flow type: f2e(fuel to electricity), h2e; f2h, e2h, h2h; e2c, h2c
        self.f2e = []
        self.h2e = []
        self.f2h = []
        self.h2h = []
        self.e2h = []
        self.h2c = []
        self.e2c = []
        self.s2h = []
        self.s2e = []
        self.s2c = []
        self.conv_system = []
        self.strg_system = []
        self.fuel_system = []
        self.wast_system = []
        self.elec_out = []
        self.heat_out = []
        self.cool_out = []
        # self.add_subsystem()

    def add_subsystem(self, subsystem=None):
        if not subsystem:
            # turbine: cy zheng, 2014； the capacity of turbien means the input fuel to turbine
            self.f2e.append(ThermalSystem(name='turbine', idx=0, capacity=2000., ramp=[0.2, 1.],
                                          pdata=[0.1283, -0.6592, 0.7945, 0.003],
                                          qdata=[-0.7098, 1.5206, -1.1191, 0.835]))
            # ORC: Xuan Wang, 2016
            self.h2e.append(ThermalSystem(name='rankine', idx=0, capacity=1500., ramp=[0.1, 1.],
                                          pdata=[.31788, -.75303, .66103, -.09595]))
            # https://jmpcoblog.com/hvac-blog/energy-efficient-hot-water-boiler-plant-design-part-2-golden-rules-of-condensing-boiler-technology
            self.f2h.append(ThermalSystem(name='boiler', idx=0, capacity=800., ramp=[0.1, 1.],
                                          pdata=[1.5819, -3.2913, 2.3124, .23454]))
            # self.h2h.append(ThermalSystem(name='absorb', idx=0, capacity=5., ramp=[0.1, 1.],
            #                               pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.e2h.append(ThermalSystem(name='heater', idx=0, capacity=800., ramp=[0.1, 1.],
                                          pdata=[0, 0, 0, 0.95]))
            # ABC: cy zheng, 2014
            self.h2c.append(ThermalSystem(name='absorb', idx=0, capacity=1000., ramp=[0.1, 1.],
                                          pdata=[0, -0.6181, 0.8669, 0.4724]))
            # cao tao phd theis [149]
            self.e2c.append(ThermalSystem(name='vcc', idx=0, capacity=1000., ramp=[0.1, 1.],
                                          pdata=[7.6816, -16.0917, 11.4564, 1.0403]))
            self.s2h.append(StorageSystem(name='storage_h', idx=0, capacity=1000., ramp=2,
                                          pdata=[0, 0]))
            # self.s2e.append(StorageSystem(name='storage_e', idx=0, capacity=1000., ramp=10,
            #                               pdata=[0, 0]))
            # self.s2c.append(StorageSystem(name='storage_c', idx=0, capacity=500., ramp=1,
            #                               pdata=[0, 0]))
        else:
            for sys in subsystem:
                if sys.name == 'turbine':
                    self.f2e.append(sys)
                elif sys.name == 'rankine':
                    self.h2e.append(sys)
                elif sys.name == 'boiler':
                    self.f2h.append(sys)
                elif sys.name == 'heater':
                    self.e2h.append(sys)
                elif sys.name == 'absorb':
                    self.h2c.append(sys)
                elif sys.name == 'vcc':
                    self.e2c.append(sys)
                elif sys.name == 'storage_h':
                    self.s2h.append(sys)
                elif sys.name == 'storage_e':
                    self.s2e.append(sys)
                elif sys.name == 'storage_c':
                    self.s2c.append(sys)
                else:
                    print('Wrong subsystem type')
        # self.conv_system = self.f2e + self.h2e + self.f2h + self.h2h + self.e2h + self.h2c + self.e2c
        self.conv_system = self.f2e + self.h2e + self.f2h + self.e2h + self.h2c + self.e2c
        self.strg_system = self.s2h + self.s2e + self.s2c
        self.fuel_system = self.f2e + self.f2h
        self.wast_system = self.h2e + self.h2h
        # self.heat_out = self.f2h + self.h2h + self.e2h
        self.heat_out = self.f2h + self.e2h
        self.cool_out = self.h2c + self.e2c
        self.elec_out = self.f2e + self.h2e