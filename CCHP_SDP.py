#!/usr/bin/env python
# @Time    : 4/11/2021 16:15
# @Author  : Lei Gao
# @Email    : leigao@umd.edu

from pyomo.environ import *
import pandas as pd
import numpy as np
import itertools
import time


# demands into dataframe
path = r'D:\Lei\work\CombinedSystemOperation\CCHPmodel_ceee_office\cchp_data.xlsx'
df_HD = pd.read_excel(path, sheet_name='heating_demand', header=0, index_col=0)
df_ED = pd.read_excel(path, sheet_name='electricity_demand', header=0, index_col=0)
df_EP = pd.read_excel(path, sheet_name='electricity_price', header=0, index_col=0)
df_Temp = pd.read_excel(path, sheet_name='temperature', header=0, index_col=0)
df_Sc = pd.read_excel(path, sheet_name='possibility', header=0, index_col=0)
df_StHt = pd.read_excel(path, sheet_name='storage_heating', header=0, index_col=0)
df_StEl = pd.read_excel(path, sheet_name='storage_electricity', header=0, index_col=0)


def subprob(t, i, j, s):
    # MODEL CONSTRUCTION
    model = ConcreteModel()

    # SET DECLARATION
    # ## subsystems index
    model.SbSy = Set(initialize=['PM', 'ORC', 'Gr', 'ABH', 'EH', 'Bo'])
    model.SSEl = Set(initialize=['PM', 'ORC'])
    model.SSHt = Set(initialize=['ABH', 'EH', 'Bo'])
    model.SSFl = Set(initialize=['PM', 'Bo'])
    model.SSWH = Set(initialize=['ORC', 'ABH'])
    model.SSSt = Set(initialize=['El', 'Ht'])
    # PARAMETER DECLARATION
    # ## capacity
    model.CpEl = Param(model.SSEl, initialize={'PM': 6.0, 'ORC': 2.5})
    model.CpHt = Param(model.SSHt, initialize={'ABH': 5, 'EH': 5, 'Bo': 5})
    model.CpSt = Param(model.SSSt, initialize={'El': 5, 'Ht': 5})
    # ## demand, probability and temperature
    model.DeHt = Param(initialize=0, mutable=True)
    model.DeEl = Param(initialize=0, mutable=True)
    model.Temp = Param(initialize=0, mutable=True)
    model.CtEl = Param(initialize=0, mutable=True)
    model.CtFl = Param(initialize=1.1763e-5)  # fuel unit price ($/kJ)
    # ## assign values
    model.DeHt = df_HD[s][t]
    model.DeEl = df_ED[1][t]
    model.Temp = df_Temp[s][t]
    model.CtEl = df_EP[1][t]

    # VARIABLES DECLARATION
    # on-off and partial load variables
    nnr = NonNegativeReals
    model.ApEl = Var(model.SSEl, within=Binary, initialize=1)  # 2
    model.ApHt = Var(model.SSHt, within=Binary, initialize=1)  # 3
    model.BtEl = Var(model.SSEl, bounds=(0.1, 1), initialize=0)  # 2
    model.BtHt = Var(model.SSHt, bounds=(0.1, 1), initialize=0)  # 3
    model.BtSt = Var(model.SSSt, within=Integers, bounds=(0, 5), initialize=0)  # 2
    # energy input and output
    model.Elec = Var(model.SSEl, within=nnr, initialize=0)  # 2
    model.Heat = Var(model.SSHt, within=nnr, initialize=0)  # 3
    model.ElGd = Var(within=nnr, initialize=0)  # from grid for electricity demand
    model.ElEH = Var(within=nnr, initialize=0)  # from grid for vapor compression heat pump
    model.HtWH = Var(model.SSWH, within=nnr, initialize=0)  # 2
    # primary energy
    model.Fuel = Var(model.SSFl, within=nnr, initialize=0)  # 2
    model.FlTt = Var(within=nnr, initialize=0)  # total fuel consumption
    # subsystem efficiency
    # model.EfSS = Var(model.SbSy, bounds=(0, 5), initialize=0)  # 5
    # additional variable
    model.zEl = Var(model.SSEl, bounds=(0, 1), initialize=0)  # 2
    model.zHt = Var(model.SSHt, bounds=(0, 1), initialize=0)  # 3

    # CONSTRAINS DECLARATION
    # energy balance of prime mover
    def _pm(m):
        return m.zEl['PM'] * m.CpEl['PM'] == m.Fuel['PM'] * (-0.001357 * m.Temp + 0.161708)
    model.pm_st = Constraint(rule=_pm)

    # energy balance of absorption heat pump
    def _abh(m):
        return m.zHt['ABH'] * m.CpHt['ABH'] == m.HtWH['ABH'] * (1.45 + 0.007737 * m.Temp - 0.04782 * m.BtHt['ABH'] -
                                                                0.0002651 * m.Temp ** 2 + 0.006368 * m.Temp * m.BtHt['ABH'])
    model.abh_st = Constraint(rule=_abh)

    # energy balance of ORC
    def _orc(m):
        return m.zEl['ORC'] * m.CpEl['ORC'] == m.HtWH['ORC'] * (-0.001357 * m.Temp + 0.161708)
    model.orc_st = Constraint(rule=_orc)

    # energy balance of electric heat pump
    def _eh(m):
        return m.zHt['EH'] * m.CpHt['EH'] == m.ElEH * (3.142 + 0.1087 * m.Temp + 0.1208 * m.BtHt['EH'] +
                                                       0.001161 * m.Temp ** 2- 0.03463 * m.Temp * m.BtHt['EH'])
    model.eh_st = Constraint(rule=_eh)

    # energy balance of boiler
    def _bo(m):
        return m.zHt['Bo'] * m.CpHt['Bo'] == m.Fuel['Bo'] * (1.572 * m.BtHt['Bo'] / (0.1745 + 1.744 * m.BtHt['Bo']))
    model.bo_st = Constraint(rule=_bo)

    # waste heat from PM
    def _wh(m):
        return m.Fuel['PM'] * (0.0474 - 0.000303 * m.Temp + 0.3866 * m.BtEl['PM'] - 2.8e-6 * m.Temp ** 2
                               + 0.0001041 * m.Temp * m.BtEl['PM'] - 0.2182 * m.BtEl['PM'] ** 2) >= \
               sum(m.HtWH[SSWH] for SSWH in m.SSWH)
    model.rh_st = Constraint(rule=_wh)

    # demand
    def _dm_ht(m):
        return sum(m.zHt[SSHt] * m.CpHt[SSHt] for SSHt in m.SSHt) \
               + (df_StHt[1][i] - model.BtSt['Ht']/m.CpSt['Ht']) * m.CpSt['Ht'] >= m.DeHt
    model.dm_ht_st = Constraint(rule=_dm_ht)

    def _dm_el(m):
        return sum(m.zEl[SSEl] * m.CpEl[SSEl] for SSEl in m.SSEl) \
               + m.ElGd + (df_StEl[1][j] - model.BtSt['El']/m.CpSt['El']) * m.CpSt['El'] >= m.DeEl
    model.dm_el_st = Constraint(rule=_dm_el)

    # constraints of additional variable z (heating&electricity)
    def _zht_1(m, ht):
        return m.zHt[ht] >= m.ApHt[ht] * 0.1
    model.zht_1_st = Constraint(model.SSHt, rule=_zht_1)

    def _zht_2(m, ht):
        return m.zHt[ht] <= m.ApHt[ht] * 1
    model.zht_2_st = Constraint(model.SSHt, rule=_zht_2)

    def _zht_3(m, ht):
        return m.zHt[ht] >= m.BtHt[ht] - (1 - m.ApHt[ht]) * 1
    model.zht_3_st = Constraint(model.SSHt, rule=_zht_3)

    def _zht_4(m, ht):
        return m.zHt[ht] <= m.BtHt[ht] - (1 - m.ApHt[ht]) * 0.1
    model.zht_4_st = Constraint(model.SSHt, rule=_zht_4)

    def _zht_5(m, ht):
        return m.zHt[ht] <= m.BtHt[ht] + (1 - m.ApHt[ht]) * 1
    model.zht_5_st = Constraint(model.SSHt, rule=_zht_5)

    def _zel_1(m, el):
        return m.zEl[el] >= m.ApEl[el] * 0.1
    model.zel_1_st = Constraint(model.SSEl, rule=_zel_1)

    def _zel_2(m, el):
        return m.zEl[el] <= m.ApEl[el] * 1
    model.zel_2_st = Constraint(model.SSEl, rule=_zel_2)

    def _zel_3(m, el):
        return m.zEl[el] >= m.BtEl[el] - (1 - m.ApEl[el]) * 1
    model.zel_3_st = Constraint(model.SSEl, rule=_zel_3)

    def _zel_4(m, el):
        return m.zEl[el] <= m.BtEl[el] - (1 - m.ApEl[el]) * 0.1
    model.zel_4_st = Constraint(model.SSEl, rule=_zel_4)

    def _zel_5(m, el):
        return m.zEl[el] <= m.BtEl[el] + (1 - m.ApEl[el]) * 1
    model.zel_5_st = Constraint(model.SSEl, rule=_zel_5)

    # OBJECTIVE DECLARATION
    def _obj(m):
        return m.CtFl * sum(m.Fuel[SSFl] for SSFl in m.SSFl) * 3600 + m.CtEl * (m.ElGd + m.ElEH)
    model.obj = Objective(rule=_obj, sense=minimize)

    return model

"""
solver_manager = SolverManagerFactory('neos')
opt = SolverFactory('cbc')
# available NLP solvers from neos: knitro; conopt; l-bfgs-b; lancelot; loqo; mosek; snopt
"""
solver = "couenne"
solver_io = None
opt = SolverFactory(solver, solver_io=solver_io)
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
            m = subprob(t, i, j, sc)
            results = opt.solve(m, keepfiles=keepfiles, tee=stream_solver)
            obj = value(m.obj)
            ii, jj = int(m.BtSt['Ht'].value), int(m.BtSt['El'].value)
            # print(m.BtSt['Ht'].value)
            # print(m.BtSt['El'].value)
            fix_cost = cost[t+1, ii, jj]
            f[t, i, j, sc] = fix_cost + obj
            # print('  The best combination of scen{0} is StHt: {1} and StEl: {2}'.format(sc, m.BtSt['Ht'].value, m.BtSt['El'].value))
        cost[t, i, j] = sum(df_Sc[s][1] * f[t, i, j, s] for s in range(1, df_Sc.size + 1))
        end = time.time()
        print('  The cost is: {0:.4} and used time is: {1:.2} for StHt {2} and StEl {3}'.
              format(cost[t, i, j], end - start, df_StHt[1][i], df_StEl[1][j]))
    print('########################\n')