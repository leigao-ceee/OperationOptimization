#!/usr/bin/env python
# @Time    : 4/11/2021 16:15
# @Author  : Lei Gao
# @Email    : leigao@umd.edu

import pyomo.environ as pyo
import pandas as pd
import numpy as np
import itertools
import time


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
        self.cache_beta = [[0 for _ in range(scen)] for _ in range(stage+1)]
        self.cache_alpha = [[0 for _ in range(scen)] for _ in range(stage + 1)]

    def eff_constraint(self, sdp):
        sdp.alpha[self.name] = pyo.Var(within=pyo.Binary, initialize=1)
        sdp.beta[self.name] = pyo.Var(within=pyo.NonNegativeReals, bounds=(self.ramp[0], self.ramp[1]), initialize=0)
        sdp.ein[self.name] = pyo.Var(within=pyo.NonNegativeReals, initialize=0)
        return sdp.alpha[self.name] == (self.pdata[0] * sdp.beta[self.name] ** 2 +
                                        self.pdata[1] * sdp.p0 ** 2 +
                                        self.pdata[2] * sdp.beta[self.name] * sdp.p0 +
                                        self.pdata[3] * sdp.beta[self.name] +
                                        self.pdata[4] * sdp.p0 +
                                        self.pdata[5]) * sdp.ein[self.name] / self.capacity


class StorageSystem(object):
    instances = []

    def __init__(self, name=None, idx=0, capacity=None, ramp=None, pdata=None):
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
        self.state = list(range(ramp+1))
        self.cache_state = None

    def generate_cache(self, stage, scen):
        self.cache_state = [[0 for _ in range(scen)] for _ in range(stage + 1)]

class CCHP_Model(object):
    def __init__(self):
        # etype: energy flow type: f2e(fuel to electricity), h2e; f2h, e2h, h2h; e2c, h2c
        self.sdp = pyo.ConcreteModel()
        self.f2e = []
        self.h2e = []
        self.f2h = []
        self.h2h = []
        self.e2h = []
        self.s2h = []
        self.s2e = []
        self.fuel_system = []
        self.conv_system = []
        self.elec_system = []
        self.heat_system = []
        self.wast_system = []
        self.strg_system = []
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
            self.h2h.append(ThermalSystem(name='absorb', idx=0, capacity=5., ramp=[0.1, 1.],
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.e2h.append(ThermalSystem(name='heater', idx=0, capacity=5., ramp=[0.1, 1.],
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.s2h.append(StorageSystem(name='storage_h', idx=0, capacity=5., ramp=5,
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
            self.s2e.append(StorageSystem(name='storage_e', idx=0, capacity=5., ramp=5,
                                          pdata=[0, 0, 0, 0, -0.001357, 0.161708]))
        else:
            pass
        self.conv_system = self.f2e + self.h2e + self.f2h + self.h2h + self.e2h
        self.fuel_system = self.f2e + self.f2h
        self.elec_system = self.f2e + self.h2e
        self.heat_system = self.f2h + self.h2h + self.e2h
        self.wast_system = self.h2e + self.h2h
        self.strg_system = self.s2h + self.s2e

    def set_para_var(self):
        # SET DECLARATION
        # ## subsystems index
        _all_sys_name = [ss.name for ss in self.conv_system] + [ss.name for ss in self.strg_system]
        self.sdp.all_systems = pyo.Set(initialize=_all_sys_name)
        self.sdp.conv_systems = pyo.Set(initialize=[ss.name for ss in self.conv_system])
        PM_name = [ss.name for ss in self.f2e]
        self.sdp.fuel_systems = pyo.Set(initialize=[ss.name for ss in self.fuel_system])
        self.sdp.grid_systems = pyo.Set(initialize=[ss.name for ss in self.e2h])
        self.sdp.PM_systems = pyo.Set(initialize=PM_name)
        self.sdp.elec_systems = pyo.Set(initialize=[ss.name for ss in self.elec_system])
        self.sdp.heat_systems = pyo.Set(initialize=[ss.name for ss in self.heat_system])
        self.sdp.wast_systems = pyo.Set(initialize=[ss.name for ss in self.wast_system])
        self.sdp.strg_systems = pyo.Set(initialize=[ss.name for ss in self.strg_system])
        self.sdp.elec_systems_strg = pyo.Set(initialize=[ss.name for ss in self.s2e])
        self.sdp.heat_systems_strg = pyo.Set(initialize=[ss.name for ss in self.s2h])
        # PARAMETER DECLARATION
        # ## capacity
        self.sdp.para_eff = pyo.Param(self.sdp.conv_systems, pyo.RangeSet(0, 5), initialize=0., mutable=True)
        self.sdp.para_waste = pyo.Param(self.sdp.PM_systems, pyo.RangeSet(0, 5), initialize=0., mutable=True)
        self.sdp.para_eff_strg = pyo.Param(self.sdp.strg_systems, pyo.RangeSet(0, 2), initialize=0., mutable=True)
        self.sdp.para_cp = pyo.Param(self.sdp.all_systems, initialize=0., mutable=True)
        self.sdp.para_ramp_u = pyo.Param(self.sdp.conv_systems, initialize=0., mutable=True)
        self.sdp.para_ramp_l = pyo.Param(self.sdp.conv_systems, initialize=0., mutable=True)
        self.sdp.para_strg_u = pyo.Param(self.sdp.strg_systems, within=pyo.Integers, initialize=0, mutable=True)
        self.sdp.para_strg_l = pyo.Param(self.sdp.strg_systems, within=pyo.Integers, initialize=0)
        self.sdp.beta_strg_load = pyo.Param(self.sdp.strg_systems, initialize=0, mutable=True)
        self.sdp.para_T = pyo.Param(initialize=0., mutable=True)
        self.sdp.elec_demand = pyo.Param(initialize=0., mutable=True)
        self.sdp.heat_demand = pyo.Param(initialize=0., mutable=True)
        self.sdp.cost_fuel = pyo.Param(initialize=0., mutable=True)
        self.sdp.cost_elec = pyo.Param(initialize=0., mutable=True)
        for system in ThermalSystem.instances:
            self.sdp.para_cp[system.name] = system.capacity
            self.sdp.para_ramp_l[system.name] = system.ramp[0]
            self.sdp.para_ramp_u[system.name] = system.ramp[1]
            for i in range(6):
                self.sdp.para_eff[system.name, i] = system.pdata[i]
                if system.name in PM_name:
                    self.sdp.para_waste[system.name, i] = system.qdata[i]
        for system in StorageSystem.instances:
            self.sdp.para_cp[system.name] = system.capacity
            self.sdp.para_strg_u[system.name] = system.ramp
            for i in range(3):
                self.sdp.para_eff_strg[system.name, i] = system.pdata[i]

        def _beta_range(model, name):
            return model.para_ramp_l[name], model.para_ramp_u[name]

        def _beta_range_strg(model, name):
            return model.para_strg_l[name], model.para_strg_u[name]
        self.sdp.beta = pyo.Var(self.sdp.conv_systems, within=pyo.NonNegativeReals, bounds=_beta_range, initialize=1.)
        self.sdp.beta_strg = pyo.Var(self.sdp.strg_systems, within=pyo.Integers, bounds=_beta_range_strg, initialize=1)
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
            return sum(model.M[sys_heat] * model.para_cp[sys_heat] for sys_heat in model.heat_systems) \
                   + (sum(model.para_cp[strg_heat] * (model.beta_strg[strg_heat] - model.beta_strg_load[strg_heat]) /
                          model.para_strg_u[strg_heat] for strg_heat in model.heat_systems_strg)) >= model.heat_demand

        def _demand_elec(model):
            return sum(model.M[sys_elec] * model.para_cp[sys_elec] for sys_elec in model.elec_systems) \
                   + (sum(model.para_cp[strg_elec] * (model.beta_strg[strg_elec] - model.beta_strg_load[strg_elec]) /
                          model.para_strg_u[strg_elec] for strg_elec in model.elec_systems_strg)) >= model.elec_demand

        def _obj(model):
            return model.cost_fuel * sum(model.energy_in[ss] for ss in model.fuel_systems) * 3600 + \
                   model.cost_elec * (sum(model.energy_in[ss] for ss in model.grid_systems) + model.grid)

        self.sdp.st_eff = pyo.Constraint(self.sdp.conv_systems, rule=_efficiency)
        self.sdp.st_wast_out = pyo.Constraint(self.sdp.PM_systems, rule=_waste_out)
        self.sdp.st_wast_in = pyo.Constraint(self.sdp.PM_systems, rule=_waste_in)
        self.sdp.st_heat = pyo.Constraint(rule=_demand_heat)
        self.sdp.st_elec = pyo.Constraint(rule=_demand_elec)
        self.sdp.st_M1 = pyo.Constraint(self.sdp.conv_systems, rule=_M1)
        self.sdp.st_M2 = pyo.Constraint(self.sdp.conv_systems, rule=_M2)
        self.sdp.st_M3 = pyo.Constraint(self.sdp.conv_systems, rule=_M3)
        self.sdp.st_M4 = pyo.Constraint(self.sdp.conv_systems, rule=_M4)
        self.sdp.st_M5 = pyo.Constraint(self.sdp.conv_systems, rule=_M5)
        self.sdp.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    def _assign_state(self, heat_state, elec_state):
        for idx, strg_heat in enumerate(self.sdp.heat_systems_strg):
            self.sdp.beta_strg_load[strg_heat] = heat_state[idx]
        for idx, strg_elec in enumerate(self.sdp.heat_systems_strg):
            self.sdp.beta_strg_load[strg_elec] = elec_state[idx]

    def _stage_scen_para(self, temp, heat_demand, elec_demand, cost_elec):
        self.sdp.para_T = temp
        self.sdp.heat_demand = heat_demand
        self.sdp.elec_demand = elec_demand
        self.sdp.cost_elec = cost_elec

    def solve_sdp(self, data_demand, data_temp, data_cost, data_prob, solver_info={}, verbosity=0):
        """:param
        stream_solver = False  # True prints solver output to screen
        keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
        """
        if not solver_info:
            solver_info['solver'] = "couenne"
            solver_info['solver_io'] = None
            solver_info['keepfiles'] = False
            solver_info['stream_solver'] = False
        opt = pyo.SolverFactory(solver_info['solver'], solver_io=solver_info['solver_io'])
        stage_number = len(data_demand)
        scen_number = len(data_prob)
        for system in self.conv_system + self.strg_system:
            system.generate_cache(stage_number, scen_number)
        for t in range(stage_number, 0, -1):
            print('Solve the {0}th stage problem'.format(str(t)))
            temp = data_temp[t]
            cost_elec = data_cost[t]
            storages = [strg.state for strg in self.strg_system]
            for heat_state, elec_state in itertools.product(*storages):
                self.sdp._assign_state([heat_state], [elec_state])
                start = time.time()
                for sc in range(scen_number):
                    self.sdp._stage_scen_para(temp, data_demand['heat'][t][sc], data_demand['elec'][t][sc], cost_elec)
                    results = opt.solve(m, keepfiles=solver_info['keepfiles'], tee=solver_info['stream_solver'])
                    obj = pyo.value(m.obj)
                    heat_idx, heat_idx = storages[0].index(self.sdp.beta_strg[0]), storages[1].index(self.sdp.beta_strg[1])
                    f[t, i, j, sc] = cost[t + 1, heat_idx, heat_idx] + obj
                    print('    The best combination of scen{0} is StHt: {1} and StEl: {2} with cost of {3:4f}'.
                            format(sc, df_StHt[1][ii], df_StEl[1][jj], f[t, i, j, sc]))
                cost[t, i, j] = sum(df_Sc[s][1] * f[t, i, j, s] for s in range(1, df_Sc.size + 1))
                end = time.time()
                print('  The cost is: {0:.4} and used time is: {1:.2} for StHt {2} and StEl {3}'.
                      format(cost[t, i, j], end - start, df_StHt[1][i], df_StEl[1][j]))
            print('########################\n')

# demands into dataframe
path = r'D:\Lei\work\CombinedSystemOperation\CCHPmodel_ceee_office\cchp_data.xlsx'
df_HD = pd.read_excel(path, sheet_name='heating_demand', header=0, index_col=0)
df_ED = pd.read_excel(path, sheet_name='electricity_demand', header=0, index_col=0)
df_EP = pd.read_excel(path, sheet_name='electricity_price', header=0, index_col=0)
df_Temp = pd.read_excel(path, sheet_name='temperature', header=0, index_col=0)
df_Sc = pd.read_excel(path, sheet_name='possibility', header=0, index_col=0)
df_StHt = pd.read_excel(path, sheet_name='storage_heating', header=0, index_col=0)
df_StEl = pd.read_excel(path, sheet_name='storage_electricity', header=0, index_col=0)


cchp_sdp = CCHP_Model()
cchp_sdp.set_para_var()
cchp_sdp.create_model()
cchp_sdp.sdp.pprint()
cchp_sdp.solve_sdp([df_HD, df_ED], df_Temp, df_EP, df_Sc, verbosity=0)
p = 1


gen_elec = {'PM': 6.0, 'ORC': 2.5}
gen_heat = {'ABH': 5, 'EH': 5, 'Bo': 5}
con_fuel = ['PM', 'Bo']
con_waste = ['ORC', 'ABH']
storage = {'El': 5, 'Ht': 5}
cchp = CCHP_Model(gen_elec, gen_heat, con_fuel, con_waste, storage)


def creat_sdp():
    # MODEL CONSTRUCTION
    model = pyo.ConcreteModel()
    # SET DECLARATION
    # ## subsystems index
    model.SbSy = pyo.Set(initialize=['PM', 'ORC', 'Gr', 'ABH', 'EH', 'Bo'])
    model.SSEl = pyo.Set(initialize=['PM', 'ORC'])
    model.SSHt = pyo.Set(initialize=['ABH', 'EH', 'Bo'])
    model.SSFl = pyo.Set(initialize=['PM', 'Bo'])
    model.SSWH = pyo.Set(initialize=['ORC', 'ABH'])
    model.SSSt = pyo.Set(initialize=['El', 'Ht'])
    # PARAMETER DECLARATION
    # ## capacity
    model.CpEl = pyo.Param(model.SSEl, initialize={'PM': 6.0, 'ORC': 2.5})
    model.CpHt = pyo.Param(model.SSHt, initialize={'ABH': 5, 'EH': 5, 'Bo': 5})
    model.CpSt = pyo.Param(model.SSSt, initialize={'El': 5, 'Ht': 5})
    # ## demand, probability and temperature, storage state
    model.DeHt = pyo.Param(initialize=0, mutable=True)
    model.DeEl = pyo.Param(initialize=0, mutable=True)
    model.Temp = pyo.Param(initialize=0, mutable=True)
    model.CtEl = pyo.Param(initialize=0, mutable=True)
    model.CtFl = pyo.Param(initialize=1.1763e-5)  # fuel unit price ($/kJ)
    model.HtSS = pyo.Param(initialize=0, mutable=True)
    model.ElSS = pyo.Param(initialize=0, mutable=True)
    # VARIABLES DECLARATION
    # on-off and partial load variables
    model.ApEl = pyo.Var(model.SSEl, within=pyo.Binary, initialize=1)  # 2
    model.ApHt = pyo.Var(model.SSHt, within=pyo.Binary, initialize=1)  # 3
    model.BtEl = pyo.Var(model.SSEl, bounds=(0.1, 1), initialize=0)  # 2
    model.BtHt = pyo.Var(model.SSHt, bounds=(0.1, 1), initialize=0)  # 3
    model.BtSt = pyo.Var(model.SSSt, within=pyo.Integers, bounds=(0, 5), initialize=0)  # 2
    # energy input and output
    model.Elec = pyo.Var(model.SSEl, within=pyo.NonNegativeReals, initialize=0)  # 2
    model.Heat = pyo.Var(model.SSHt, within=pyo.NonNegativeReals, initialize=0)  # 3
    model.ElGd = pyo.Var(within=pyo.NonNegativeReals, initialize=0)  # from grid for electricity demand
    model.ElEH = pyo.Var(within=pyo.NonNegativeReals, initialize=0)  # from grid for vapor compression heat pump
    model.HtWH = pyo.Var(model.SSWH, within=pyo.NonNegativeReals, initialize=0)  # 2
    # primary energy
    model.Fuel = pyo.Var(model.SSFl, within=pyo.NonNegativeReals, initialize=0)  # 2
    # subsystem efficiency
    # model.EfSS = Var(model.SbSy, bounds=(0, 5), initialize=0)  # 5
    # additional variable
    model.zEl = pyo.Var(model.SSEl, bounds=(0, 1), initialize=0)  # 2
    model.zHt = pyo.Var(model.SSHt, bounds=(0, 1), initialize=0)  # 3

    # CONSTRAINS DECLARATION
    # energy balance of prime mover
    def _pm(m):
        return m.zEl['PM'] * m.CpEl['PM'] == m.Fuel['PM'] * (-0.001357 * m.Temp + 0.161708)

    model.pm_st = pyo.Constraint(rule=_pm)

    # waste heat from PM
    def _wh(m):
        return m.Fuel['PM'] * (0.0474 - 0.000303 * m.Temp + 0.3866 * m.BtEl['PM'] - 2.8e-6 * m.Temp ** 2
                               + 0.0001041 * m.Temp * m.BtEl['PM'] - 0.2182 * m.BtEl['PM'] ** 2) >= \
               sum(m.HtWH[SSWH] for SSWH in m.SSWH)

    model.rh_st = pyo.Constraint(rule=_wh)

    # energy balance of absorption heat pump
    def _abh(m):
        return m.zHt['ABH'] * m.CpHt['ABH'] == m.HtWH['ABH'] * (1.45 + 0.007737 * m.Temp - 0.04782 * m.BtHt['ABH'] -
                                                                0.0002651 * m.Temp ** 2 + 0.006368 * m.Temp * m.BtHt[
                                                                    'ABH'])

    model.abh_st = pyo.Constraint(rule=_abh)

    # energy balance of ORC
    def _orc(m):
        return m.zEl['ORC'] * m.CpEl['ORC'] == m.HtWH['ORC'] * (-0.001357 * m.Temp + 0.161708)

    model.orc_st = pyo.Constraint(rule=_orc)

    # energy balance of electric heat pump
    def _eh(m):
        return m.zHt['EH'] * m.CpHt['EH'] == m.ElEH * (3.142 + 0.1087 * m.Temp + 0.1208 * m.BtHt['EH'] +
                                                       0.001161 * m.Temp ** 2 - 0.03463 * m.Temp * m.BtHt['EH'])

    model.eh_st = pyo.Constraint(rule=_eh)

    # energy balance of boiler
    def _bo(m):
        return m.zHt['Bo'] * m.CpHt['Bo'] == m.Fuel['Bo'] * (1.572 * m.BtHt['Bo'] / (0.1745 + 1.744 * m.BtHt['Bo']))

    model.bo_st = pyo.Constraint(rule=_bo)

    # demand
    def _dm_ht(m):
        return sum(m.zHt[SSHt] * m.CpHt[SSHt] for SSHt in m.SSHt) \
               + (model.HtSS - model.BtSt['Ht'] / m.CpSt['Ht']) * m.CpSt['Ht'] >= m.DeHt

    model.dm_ht_st = pyo.Constraint(rule=_dm_ht)

    def _dm_el(m):
        return sum(m.zEl[SSEl] * m.CpEl[SSEl] for SSEl in m.SSEl) \
               + m.ElGd + (model.ElSS - model.BtSt['El'] / m.CpSt['El']) * m.CpSt['El'] >= m.DeEl

    model.dm_el_st = pyo.Constraint(rule=_dm_el)

    # constraints of additional variable z (heating&electricity)
    def _zht_1(m, ht):
        return m.zHt[ht] >= m.ApHt[ht] * 0.1

    model.zht_1_st = pyo.Constraint(model.SSHt, rule=_zht_1)

    def _zht_2(m, ht):
        return m.zHt[ht] <= m.ApHt[ht] * 1

    model.zht_2_st = pyo.Constraint(model.SSHt, rule=_zht_2)

    def _zht_3(m, ht):
        return m.zHt[ht] >= m.BtHt[ht] - (1 - m.ApHt[ht]) * 1

    model.zht_3_st = pyo.Constraint(model.SSHt, rule=_zht_3)

    def _zht_4(m, ht):
        return m.zHt[ht] <= m.BtHt[ht] - (1 - m.ApHt[ht]) * 0.1

    model.zht_4_st = pyo.Constraint(model.SSHt, rule=_zht_4)

    def _zht_5(m, ht):
        return m.zHt[ht] <= m.BtHt[ht] + (1 - m.ApHt[ht]) * 1

    model.zht_5_st = pyo.Constraint(model.SSHt, rule=_zht_5)

    def _zel_1(m, el):
        return m.zEl[el] >= m.ApEl[el] * 0.1

    model.zel_1_st = pyo.Constraint(model.SSEl, rule=_zel_1)

    def _zel_2(m, el):
        return m.zEl[el] <= m.ApEl[el] * 1

    model.zel_2_st = pyo.Constraint(model.SSEl, rule=_zel_2)

    def _zel_3(m, el):
        return m.zEl[el] >= m.BtEl[el] - (1 - m.ApEl[el]) * 1

    model.zel_3_st = pyo.Constraint(model.SSEl, rule=_zel_3)

    def _zel_4(m, el):
        return m.zEl[el] <= m.BtEl[el] - (1 - m.ApEl[el]) * 0.1

    model.zel_4_st = pyo.Constraint(model.SSEl, rule=_zel_4)

    def _zel_5(m, el):
        return m.zEl[el] <= m.BtEl[el] + (1 - m.ApEl[el]) * 1

    model.zel_5_st = pyo.Constraint(model.SSEl, rule=_zel_5)

    # OBJECTIVE DECLARATION
    def _obj(m):
        return m.CtFl * sum(m.Fuel[SSFl] for SSFl in m.SSFl) * 3600 + m.CtEl * (m.ElGd + m.ElEH)

    model.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    return model


def assign_param(model, s, t, i, j):
    model.DeHt = df_HD[s][t]
    model.DeEl = df_ED[1][t]
    model.Temp = df_Temp[s][t]
    model.CtEl = df_EP[1][t]
    model.HtSS = df_StHt[1][i]
    model.ElSS = df_StEl[1][j]


"""
solver_manager = SolverManagerFactory('neos')
opt = SolverFactory('cbc')
# available NLP solvers from neos: knitro; conopt; l-bfgs-b; lancelot; loqo; mosek; snopt
"""
solver = "couenne"
solver_io = None
opt = pyo.SolverFactory(solver, solver_io=solver_io)
# opt.options['expect_infeasible_problem'] = 'no'
"""
opt.options['acceptable_tol'] = 0.001
opt.options['ms_enable'] = 1
opt.options['ms_maxsolves'] = 0
opt.options['par_numthreads'] = 12
opt.options['ms_savetol'] = 0.0001
opt.options['ms_num_to_save'] = 3
opt.options['ms_maxtime_cpu'] = 600
"""
stream_solver = False  # True prints solver output to screen
keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
# store temparory data [stage, state, scenario]
f = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_st_ht = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_st_el = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_al_pm = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_bt_pm = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_al_orc = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_bt_orc = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_al_abh = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_bt_abh = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_al_bo = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_bt_bo = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_al_eh = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cache_bt_eh = np.zeros([len(df_HD) + 1, df_StHt.size + 1, df_StEl.size + 1, df_Sc.size + 1])
cost = np.zeros([len(df_HD) + 2, df_StHt.size + 1, df_StEl.size + 1])

# for t in range(len(df_HD), 0, -1):
for t in range(24, 0, -1):
    print('Solve the {0}th stage problem'.format(str(t)))
    for i, j in itertools.product(range(1, df_StHt.size + 1), range(1, df_StEl.size + 1)):
        start = time.time()
        for sc in range(1, df_Sc.size + 1):
            m = creat_sdp()
            assign_param(m, sc, t, i, j)
            results = opt.solve(m, keepfiles=keepfiles, tee=stream_solver)
            obj = pyo.value(m.obj)
            ii, jj = int(pyo.value(m.BtSt['Ht'])) + 1, int(pyo.value(m.BtSt['El'])) + 1
            f[t, i, j, sc] = cost[t + 1, ii, jj] + obj
            print('    The best combination of scen{0} is StHt: {1} and StEl: {2} with cost of {3:4f}'.format(sc,
                                                                                                              df_StHt[
                                                                                                                  1][
                                                                                                                  ii],
                                                                                                              df_StEl[
                                                                                                                  1][
                                                                                                                  jj],
                                                                                                              f[
                                                                                                                  t, i, j, sc]))
        cost[t, i, j] = sum(df_Sc[s][1] * f[t, i, j, s] for s in range(1, df_Sc.size + 1))
        end = time.time()
        print('  The cost is: {0:.4} and used time is: {1:.2} for StHt {2} and StEl {3}'.
              format(cost[t, i, j], end - start, df_StHt[1][i], df_StEl[1][j]))
    print('########################\n')
