#!/usr/bin/env python
# @Time    : 4/11/2021 16:15
# @Author  : Lei Gao
# @Email    : leigao@umd.edu

import pyomo.environ as pyo
import pandas as pd
import numpy as np
import itertools
import time
import pickle
from math import inf


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

    def generate_cache(self, stage, scen):
        self.cache_beta = [[0 for _ in range(scen)] for _ in range(stage + 1)]
        self.cache_alpha = [[0 for _ in range(scen)] for _ in range(stage + 1)]

    # def eff_constraint(self, sdp):
    #     def _eff_constrain(model):
    #         if self.name == 'turbine':
    #             model.alpha = pyo.Var(within=pyo.Binary, initialize=1)
    #             model.beta = pyo.Var(within=pyo.NonNegativeReals, bounds=(self.ramp[0], self.ramp[1]), initialize=0)
    #             sdp.ein[self.name] = pyo.Var(within=pyo.NonNegativeReals, initialize=0)
    #             return sdp.alpha[self.name] == (self.pdata[0] * sdp.beta[self.name] ** 2 +
    #                                             self.pdata[1] * sdp.p0 ** 2 +
    #                                             self.pdata[2] * sdp.beta[self.name] * sdp.p0 +
    #                                             self.pdata[3] * sdp.beta[self.name] +
    #                                             self.pdata[4] * sdp.p0 +
    #                                             self.pdata[5]) * sdp.ein[self.name] / self.capacity


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
        self.state = list(range(ramp + 1))
        self.cache_state = None

    def generate_cache(self, stage, scen):
        self.cache_state = [[0 for _ in range(scen)] for _ in range(stage + 1)]


class CCHP_Model(object):
    def __init__(self):
        # energy flow type: f2e(fuel to electricity), h2e; f2h, e2h, h2h; e2c, h2c
        self.sdp = pyo.ConcreteModel()
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
        self.add_subsystem()

    def add_subsystem(self, subsystem=None):
        if not subsystem:
            self.f2e.append(ThermalSystem(name='turbine', idx=0, capacity=5., ramp=[0.2, 1.],
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708],
                                          qdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.h2e.append(ThermalSystem(name='rankine', idx=0, capacity=5., ramp=[0.1, 1.],
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.f2h.append(ThermalSystem(name='boiler', idx=0, capacity=5., ramp=[0.1, 1.],
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            # self.h2h.append(ThermalSystem(name='absorb', idx=0, capacity=5., ramp=[0.1, 1.],
            #                               pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.e2h.append(ThermalSystem(name='heater', idx=0, capacity=5., ramp=[0.1, 1.],
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.h2c.append(ThermalSystem(name='absorb', idx=0, capacity=5., ramp=[0.1, 1.],
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.e2c.append(ThermalSystem(name='vcc', idx=0, capacity=5., ramp=[0.1, 1.],
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.s2h.append(StorageSystem(name='storage_h', idx=0, capacity=5., ramp=2,
                                          pdata=[0, 0]))
            self.s2e.append(StorageSystem(name='storage_e', idx=0, capacity=5., ramp=2,
                                          pdata=[0, 0]))
            self.s2c.append(StorageSystem(name='storage_c', idx=0, capacity=5., ramp=2,
                                          pdata=[0, 0]))
        else:
            pass
        # self.conv_system = self.f2e + self.h2e + self.f2h + self.h2h + self.e2h + self.h2c + self.e2c
        self.conv_system = self.f2e + self.h2e + self.f2h + self.e2h + self.h2c + self.e2c
        self.strg_system = self.s2h + self.s2e + self.s2c
        self.fuel_system = self.f2e + self.f2h
        self.wast_system = self.h2e + self.h2h
        # self.heat_out = self.f2h + self.h2h + self.e2h
        self.heat_out = self.f2h + self.e2h
        self.cool_out = self.h2c + self.e2c
        self.elec_out = self.f2e + self.h2e

    def set_para_var(self):
        # SET DECLARATION
        # ## subsystems index
        _all_sys_name = [ss.name for ss in self.conv_system] + [ss.name for ss in self.strg_system]
        self.sdp.all_systems = pyo.Set(initialize=_all_sys_name)
        self.sdp.conv_systems = pyo.Set(initialize=[ss.name for ss in self.conv_system])
        self.sdp.strg_systems = pyo.Set(initialize=[ss.name for ss in self.strg_system])
        self.sdp.grid_systems = pyo.Set(initialize=[ss.name for ss in self.e2h])
        self.sdp.fuel_systems = pyo.Set(initialize=[ss.name for ss in self.fuel_system])
        self.sdp.wast_systems = pyo.Set(initialize=[ss.name for ss in self.wast_system])
        self.sdp.heat_out = pyo.Set(initialize=[ss.name for ss in self.heat_out])
        self.sdp.cool_out = pyo.Set(initialize=[ss.name for ss in self.cool_out])
        self.sdp.elec_out = pyo.Set(initialize=[ss.name for ss in self.elec_out])
        self.sdp.heat_strg = pyo.Set(initialize=[ss.name for ss in self.s2h])
        self.sdp.cool_strg = pyo.Set(initialize=[ss.name for ss in self.s2c])
        self.sdp.elec_strg = pyo.Set(initialize=[ss.name for ss in self.s2e])
        self.sdp.PM_systems = pyo.Set(initialize=[ss.name for ss in self.f2e])
        # PARAMETER DECLARATION
        # ## efficiency
        self.sdp.para_eff = pyo.Param(self.sdp.conv_systems, pyo.RangeSet(0, 5), initialize=0., mutable=True)
        self.sdp.para_waste = pyo.Param(self.sdp.PM_systems, pyo.RangeSet(0, 5), initialize=0., mutable=True)
        self.sdp.para_eff_strg = pyo.Param(self.sdp.strg_systems, pyo.RangeSet(0, 1), initialize=0., mutable=True)
        # ## capacity
        self.sdp.para_cp = pyo.Param(self.sdp.all_systems, initialize=0., mutable=True)
        # ## operation limitation
        self.sdp.para_ramp_u = pyo.Param(self.sdp.conv_systems, initialize=0., mutable=True)
        self.sdp.para_ramp_l = pyo.Param(self.sdp.conv_systems, initialize=0., mutable=True)
        self.sdp.para_strg_u = pyo.Param(self.sdp.strg_systems, within=pyo.Integers, initialize=0, mutable=True)
        self.sdp.para_strg_l = pyo.Param(self.sdp.strg_systems, within=pyo.Integers, initialize=0)
        # ## current storage state
        self.sdp.beta_strg_now = pyo.Param(self.sdp.strg_systems, initialize=0, mutable=True)
        self.sdp.beta_strg_next = pyo.Param(self.sdp.strg_systems, initialize=0, mutable=True)
        # ## parameters of system
        self.sdp.para_T = pyo.Param(initialize=0., mutable=True)
        self.sdp.elec_demand = pyo.Param(initialize=0., mutable=True)
        self.sdp.heat_demand = pyo.Param(initialize=0., mutable=True)
        self.sdp.cool_demand = pyo.Param(initialize=0., mutable=True)
        self.sdp.cost_fuel = pyo.Param(initialize=0., mutable=True)
        self.sdp.cost_elec = pyo.Param(initialize=0., mutable=True)
        # ## set conversion and storage systems efficiency correlation and capacity
        for system in self.conv_system:
            self.sdp.para_cp[system.name] = system.capacity
            self.sdp.para_ramp_l[system.name] = system.ramp[0]
            self.sdp.para_ramp_u[system.name] = system.ramp[1]
            for i in range(6):
                self.sdp.para_eff[system.name, i] = system.pdata[i]
                if system.name in [ss.name for ss in self.f2e]:
                    self.sdp.para_waste[system.name, i] = system.qdata[i]
        for system in self.strg_system:
            self.sdp.para_cp[system.name] = system.capacity
            self.sdp.para_strg_u[system.name] = system.ramp
            for i in range(2):
                self.sdp.para_eff_strg[system.name, i] = system.pdata[i]

        # ## for setting the variable upper and lower range
        def _beta_range(model, name):
            return model.para_ramp_l[name], model.para_ramp_u[name]

        def _beta_range_strg(model, name):
            return model.para_strg_l[name], model.para_strg_u[name]

        self.sdp.beta = pyo.Var(self.sdp.conv_systems, within=pyo.NonNegativeReals, bounds=_beta_range, initialize=1.)
        # self.sdp.beta_strg = pyo.Var(self.sdp.strg_systems, within=pyo.Integers, bounds=_beta_range_strg, initialize=1)
        self.sdp.alpha = pyo.Var(self.sdp.all_systems, within=pyo.Binary, initialize=1)
        self.sdp.M = pyo.Var(self.sdp.conv_systems, bounds=(0, 1.0), initialize=0.)
        self.sdp.energy_in = pyo.Var(self.sdp.conv_systems, within=pyo.NonNegativeReals, initialize=0.)
        self.sdp.waste_out = pyo.Var(self.sdp.PM_systems, within=pyo.NonNegativeReals, initialize=0.)
        self.sdp.energy_out = pyo.Var(self.sdp.conv_systems, within=pyo.NonNegativeReals, initialize=0.)
        self.sdp.grid = pyo.Var(within=pyo.NonNegativeReals, initialize=0.)

    def create_model(self):
        def _efficiency(model, subsystem):
            return model.M[subsystem] == (model.para_eff[subsystem, 0] * model.beta[subsystem] ** 2 +
                                          model.para_eff[subsystem, 1] * model.para_T ** 2 +
                                          model.para_eff[subsystem, 2] * model.beta[subsystem] * model.para_T +
                                          model.para_eff[subsystem, 3] * model.beta[subsystem] +
                                          model.para_eff[subsystem, 4] * model.para_T +
                                          model.para_eff[subsystem, 5]) * model.energy_in[subsystem] / \
                   model.para_cp[subsystem]

        # def _efficiency_strg(model, subsystem):
        #     return model.eff[subsystem] == (model.para_eff_strg[subsystem, 0] * model.beta[subsystem] +
        #                                   model.para_eff_strg[subsystem, 1] * model.para_T +
        #                                   model.para_eff_strg[subsystem, 2]) * model.energy_i[subsystem] / \
        #            model.para_cp[subsystem]

        def _waste_out(model, subsystem):
            return (model.para_waste[subsystem, 0] * model.beta[subsystem] ** 2 +
                    model.para_waste[subsystem, 1] * model.para_T ** 2 +
                    model.para_waste[subsystem, 2] * model.beta[subsystem] * model.para_T +
                    model.para_waste[subsystem, 3] * model.beta[subsystem] +
                    model.para_waste[subsystem, 4] * model.para_T +
                    model.para_waste[subsystem, 5]) * model.energy_in[subsystem] == model.waste_out[subsystem]

        def _waste_in(model):
            return sum(model.waste_out[pm] for pm in model.PM_systems) >= \
                   sum(model.energy_in[sys] for sys in model.wast_systems)

        def _M1(model, subsystem):
            return model.M[subsystem] >= model.alpha[subsystem] * model.para_ramp_l[subsystem]

        def _M2(model, subsystem):
            return model.M[subsystem] <= model.alpha[subsystem] * model.para_ramp_u[subsystem]

        def _M3(model, subsystem):
            return model.M[subsystem] >= model.beta[subsystem] - \
                   (model.para_ramp_u[subsystem] - model.alpha[subsystem]) * model.para_ramp_u[subsystem]

        def _M4(model, subsystem):
            return model.M[subsystem] <= model.beta[subsystem] - \
                   (model.para_ramp_u[subsystem] - model.alpha[subsystem]) * model.para_ramp_l[subsystem]

        def _M5(model, subsystem):
            return model.M[subsystem] <= model.beta[subsystem] + \
                   (model.para_ramp_u[subsystem] - model.alpha[subsystem]) * model.para_ramp_u[subsystem]

        def _demand_heat(model):
            return sum(model.M[sys_heat] * model.para_cp[sys_heat] for sys_heat in model.heat_out) \
                   + (sum(model.para_cp[strg_heat] * (model.beta_strg_next[strg_heat] - model.beta_strg_now[strg_heat])
                          / model.para_strg_u[strg_heat] for strg_heat in model.heat_strg)) >= model.heat_demand

        def _demand_cool(model):
            return sum(model.M[sys_cool] * model.para_cp[sys_cool] for sys_cool in model.cool_out) \
                   + (sum(model.para_cp[strg_cool] * (model.beta_strg_next[strg_cool] - model.beta_strg_now[strg_cool])
                          / model.para_strg_u[strg_cool] for strg_cool in model.cool_strg)) >= model.cool_demand

        def _demand_elec(model):
            return sum(model.M[sys_elec] * model.para_cp[sys_elec] for sys_elec in model.elec_out) \
                   + (sum(model.para_cp[strg_elec] * (model.beta_strg_next[strg_elec] - model.beta_strg_now[strg_elec])
                          / model.para_strg_u[strg_elec] for strg_elec in model.elec_strg)) >= model.elec_demand

        def _obj(model):
            return model.cost_fuel * sum(model.energy_in[ss] for ss in model.fuel_systems) * 3600 + \
                   model.cost_elec * (sum(model.energy_in[ss] for ss in model.grid_systems) + model.grid)

        self.sdp.st_eff = pyo.Constraint(self.sdp.conv_systems, rule=_efficiency)
        self.sdp.st_wast_out = pyo.Constraint(self.sdp.PM_systems, rule=_waste_out)
        self.sdp.st_wast_in = pyo.Constraint(self.sdp.PM_systems, rule=_waste_in)
        self.sdp.st_heat = pyo.Constraint(rule=_demand_heat)
        self.sdp.st_cool = pyo.Constraint(rule=_demand_cool)
        self.sdp.st_elec = pyo.Constraint(rule=_demand_elec)
        self.sdp.st_M1 = pyo.Constraint(self.sdp.conv_systems, rule=_M1)
        self.sdp.st_M2 = pyo.Constraint(self.sdp.conv_systems, rule=_M2)
        self.sdp.st_M3 = pyo.Constraint(self.sdp.conv_systems, rule=_M3)
        self.sdp.st_M4 = pyo.Constraint(self.sdp.conv_systems, rule=_M4)
        self.sdp.st_M5 = pyo.Constraint(self.sdp.conv_systems, rule=_M5)
        self.sdp.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    def _assign_state(self, heat_state, cool_state, elec_state, mode):
        if mode == 'now':
            for idx, strg_heat in enumerate(self.sdp.heat_strg):
                self.sdp.beta_strg_now[strg_heat] = heat_state[idx]
            for idx, strg_cool in enumerate(self.sdp.cool_strg):
                self.sdp.beta_strg_now[strg_cool] = cool_state[idx]
            for idx, strg_elec in enumerate(self.sdp.elec_strg):
                self.sdp.beta_strg_now[strg_elec] = elec_state[idx]
        elif mode == 'next':
            for idx, strg_heat in enumerate(self.sdp.heat_strg):
                self.sdp.beta_strg_next[strg_heat] = heat_state[idx]
            for idx, strg_cool in enumerate(self.sdp.cool_strg):
                self.sdp.beta_strg_next[strg_cool] = cool_state[idx]
            for idx, strg_elec in enumerate(self.sdp.elec_strg):
                self.sdp.beta_strg_next[strg_elec] = elec_state[idx]
        else:
            print('Something wrong')

    def _stage_scen_para(self, temp, demands, cost_elec):
        self.sdp.para_T = temp
        self.sdp.heat_demand = demands[0]
        self.sdp.cool_demand = demands[1]
        self.sdp.elec_demand = demands[2]
        self.sdp.cost_elec = cost_elec

    def solve_sdp(self, markov_demands, markov_prob, data_temp, data_cost, data_prob, solver_info=None, verbosity=0):
        """:param
        stream_solver = False  # True prints solver output to screen
        keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
        """
        if solver_info is None:
            solver_info = {'solver': "couenne", 'solver_io': None, 'keepfiles': True, 'stream_solver': False}
        opt = pyo.SolverFactory(solver_info['solver'], solver_io=solver_info['solver_io'])
        stage_number = len(markov_demands)
        scen_number = len(markov_demands[-1])
        storages = [strg.state for strg in self.strg_system]
        cost = np.zeros([stage_number+1, len(storages[0]), len(storages[1]), len(storages[2]), scen_number])

        for system in self.conv_system + self.strg_system:
            system.generate_cache(stage_number, scen_number)

        for t in range(stage_number - 1, 0, -1):
            print('Solve the {0}th stage problem'.format(str(t)))
            for st_h, st_c, st_e in itertools.product(*storages):
                self._assign_state([st_h], [st_c], [st_e], 'now')
                start = time.time()
                for sc in range(scen_number):
                    self._stage_scen_para(data_temp[1][t], markov_demands[t - 1][sc], data_cost[1][t])
                    # cost[t, st_h, st_c, st_e, sc] = -inf
                    cost_now = +inf
                    for st_h_next, st_c_next, st_e_next in itertools.product(*storages):
                        self._assign_state([st_h_next], [st_c_next], [st_e_next], 'next')
                        results = opt.solve(self.sdp, keepfiles=solver_info['keepfiles'], tee=solver_info['stream_solver'])
                        obj = pyo.value(self.sdp.obj)
                        cost_temp = cost[t+1, st_h_next, st_c_next, st_e_next, sc] + obj
                        if cost_temp <= cost_now:
                            cost_now = cost_temp
                    cost[t, st_h, st_c, st_e, sc] = cost_now
                for sc_now in range(scen_number):
                    cost[t, st_h, st_c, st_e, sc_now] = sum(cost[t, st_h, st_c, st_e, sc] * markov_prob[t][sc_now][sc]
                                                            for sc in range(scen_number))
                print('    The best combination of scen{0} is StHt: {1} and StEl: {2} with cost of {3:4f}'.
                  format(sc, df_StHt[1][ii], df_StEl[1][jj], f[t, i, j, sc]))
            cost[t, i, j] = sum(df_Sc[s][1] * f[t, i, j, s] for s in range(1, df_Sc.size + 1))
            end = time.time()
            print('  The cost is: {0:.4} and used time is: {1:.2} for StHt {2} and StEl {3}'.
                  format(cost[t, i, j], end - start, df_StHt[1][i], df_StEl[1][j]))
            print('########################\n')


# demands into dataframe
infile_states = open('sampling/markov_states', 'rb')
markov_states = pickle.load(infile_states)
infile_states.close()
infile_transition = open('sampling/markov_transition', 'rb')
markov_transition = pickle.load(infile_transition)
infile_transition.close()

path = r'D:\Lei\work\CombinedSystemOperation\CCHPmodel_ceee_office\cchp_data.xlsx'
df_HD = pd.read_excel(path, sheet_name='heating_demand', header=0, index_col=0)
df_CD = pd.read_excel(path, sheet_name='heating_demand', header=0, index_col=0)
df_ED = pd.read_excel(path, sheet_name='electricity_demand', header=0, index_col=0)
df_EP = pd.read_excel(path, sheet_name='electricity_price', header=0, index_col=0)
df_Temp = pd.read_excel(path, sheet_name='temperature', header=0, index_col=0)
df_Sc = pd.read_excel(path, sheet_name='possibility', header=0, index_col=0)
df_StHt = pd.read_excel(path, sheet_name='storage_heating', header=0, index_col=0)
df_StEl = pd.read_excel(path, sheet_name='storage_electricity', header=0, index_col=0)

cchp_sdp = CCHP_Model()
# cchp_sdp.add_subsystem()
cchp_sdp.set_para_var()
cchp_sdp.create_model()
# cchp_sdp.sdp.pprint()
cchp_sdp.solve_sdp(markov_states, markov_transition, df_Temp, df_EP, df_Sc, verbosity=0)

gen_elec = {'PM': 6.0, 'ORC': 2.5}
gen_heat = {'ABH': 5, 'EH': 5, 'Bo': 5}
con_fuel = ['PM', 'Bo']
con_waste = ['ORC', 'ABH']
storage = {'El': 5, 'Ht': 5}
cchp = CCHP_Model(gen_elec, gen_heat, con_fuel, con_waste, storage)
