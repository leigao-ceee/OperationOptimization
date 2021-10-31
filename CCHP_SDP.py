#!/usr/bin/env python
# @Time    : 4/11/2021 16:15
# @Author  : Lei Gao
# @Email    : leigao@umd.edu

import pyomo.environ as pyo
import numpy as np
import itertools
import time
from math import inf


class SDP_Model(object):
    def __init__(self):
        # energy flow type: f2e(fuel to electricity), h2e; f2h, e2h, h2h; e2c, h2c
        self.sdp = pyo.ConcreteModel()

    def set_para_var(self, cchp):
        # SET DECLARATION
        # ## subsystems index
        _all_sys_name = [ss.name for ss in cchp.conv_system] + [ss.name for ss in cchp.strg_system]
        self.sdp.all_systems = pyo.Set(initialize=_all_sys_name)
        self.sdp.conv_systems = pyo.Set(initialize=[ss.name for ss in cchp.conv_system])
        self.sdp.strg_systems = pyo.Set(initialize=[ss.name for ss in cchp.strg_system])
        self.sdp.grid_systems = pyo.Set(initialize=[ss.name for ss in cchp.e2h])
        self.sdp.fuel_systems = pyo.Set(initialize=[ss.name for ss in cchp.fuel_system])
        self.sdp.wast_systems = pyo.Set(initialize=[ss.name for ss in cchp.wast_system])
        self.sdp.heat_out = pyo.Set(initialize=[ss.name for ss in cchp.heat_out])
        self.sdp.cool_out = pyo.Set(initialize=[ss.name for ss in cchp.cool_out])
        self.sdp.elec_out = pyo.Set(initialize=[ss.name for ss in cchp.elec_out])
        self.sdp.heat_strg = pyo.Set(initialize=[ss.name for ss in cchp.s2h])
        self.sdp.cool_strg = pyo.Set(initialize=[ss.name for ss in cchp.s2c])
        self.sdp.elec_strg = pyo.Set(initialize=[ss.name for ss in cchp.s2e])
        self.sdp.PM_systems = pyo.Set(initialize=[ss.name for ss in cchp.f2e])
        # PARAMETER DECLARATION
        # ## efficiency
        self.sdp.para_eff = pyo.Param(self.sdp.conv_systems, pyo.RangeSet(0, 3), initialize=0., mutable=True)
        self.sdp.para_waste = pyo.Param(self.sdp.PM_systems, pyo.RangeSet(0, 3), initialize=0., mutable=True)
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
        self.sdp.cost_fuel = pyo.Param(initialize=6.389e-6, mutable=True)  # fuel unit price ($/kJ)
        self.sdp.cost_elec = pyo.Param(initialize=11.21, mutable=True)  # electricity unit price (c/kJ)
        # ## set conversion and storage systems efficiency correlation and capacity
        for system in cchp.conv_system:
            self.sdp.para_cp[system.name] = system.capacity
            self.sdp.para_ramp_l[system.name] = system.ramp[0]
            self.sdp.para_ramp_u[system.name] = system.ramp[1]
            for i in range(4):
                self.sdp.para_eff[system.name, i] = system.pdata[i]
                if system.name in [ss.name for ss in cchp.f2e]:
                    self.sdp.para_waste[system.name, i] = system.qdata[i]
        for system in cchp.strg_system:
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
        self.sdp.grid = pyo.Var(within=pyo.NonNegativeReals, initialize=0.)  # none-negative for forbidden sale to grid

    def create_model(self):
        def _efficiency(model, subsystem):
            # return model.M[subsystem] == (model.para_eff[subsystem, 0] * model.beta[subsystem] ** 2 +
            #                               model.para_eff[subsystem, 1] * model.para_T ** 2 +
            #                               model.para_eff[subsystem, 2] * model.beta[subsystem] * model.para_T +
            #                               model.para_eff[subsystem, 3] * model.beta[subsystem] +
            #                               model.para_eff[subsystem, 4] * model.para_T +
            #                               model.para_eff[subsystem, 5]) * model.energy_in[subsystem] / \
            #        model.para_cp[subsystem]
            return model.M[subsystem] == (model.para_eff[subsystem, 0] * model.beta[subsystem] ** 3 +
                                          model.para_eff[subsystem, 1] * model.beta[subsystem] ** 2 +
                                          model.para_eff[subsystem, 2] * model.beta[subsystem] +
                                          model.para_eff[subsystem, 3]) * model.energy_in[subsystem] / \
                   model.para_cp[subsystem]

        # def _efficiency_strg(model, subsystem):
        #     return model.eff[subsystem] == (model.para_eff_strg[subsystem, 0] * model.beta[subsystem] +
        #                                   model.para_eff_strg[subsystem, 1] * model.para_T +
        #                                   model.para_eff_strg[subsystem, 2]) * model.energy_i[subsystem] / \
        #            model.para_cp[subsystem]

        def _waste_out(model, subsystem):
            # return (model.para_waste[subsystem, 0] * model.beta[subsystem] ** 2 +
            #         model.para_waste[subsystem, 1] * model.para_T ** 2 +
            #         model.para_waste[subsystem, 2] * model.beta[subsystem] * model.para_T +
            #         model.para_waste[subsystem, 3] * model.beta[subsystem] +
            #         model.para_waste[subsystem, 4] * model.para_T +
            #         model.para_waste[subsystem, 5]) * model.energy_in[subsystem] == model.waste_out[subsystem]
            return (model.para_waste[subsystem, 0] * model.beta[subsystem] ** 3 +
                    model.para_waste[subsystem, 1] * model.beta[subsystem] ** 2 +
                    model.para_waste[subsystem, 2] * model.beta[subsystem] +
                    model.para_waste[subsystem, 3]) * model.energy_in[subsystem] == model.waste_out[subsystem]

        def _waste_in(model):
            return sum(model.waste_out[pm] for pm in model.PM_systems) >= \
                   sum(model.energy_in[sys] for sys in model.wast_systems)

        def _M1(model, subsystem):
            return model.M[subsystem] >= model.alpha[subsystem] * model.para_ramp_l[subsystem]

        def _M2(model, subsystem):
            return model.M[subsystem] <= model.alpha[subsystem] * model.para_ramp_u[subsystem]

        def _M3(model, subsystem):
            return model.M[subsystem] >= model.beta[subsystem] - \
                   (1 - model.alpha[subsystem]) * model.para_ramp_u[subsystem]

        def _M4(model, subsystem):
            return model.M[subsystem] <= model.beta[subsystem] - \
                   (1 - model.alpha[subsystem]) * model.para_ramp_l[subsystem]

        def _M5(model, subsystem):
            return model.M[subsystem] <= model.beta[subsystem] + \
                   (1 - model.alpha[subsystem]) * model.para_ramp_u[subsystem]

        def _demand_heat(model):
            return sum(model.M[sys_heat] * model.para_cp[sys_heat] for sys_heat in model.heat_out) \
                   + (sum(model.para_cp[strg_heat] * (model.beta_strg_next[strg_heat] - model.beta_strg_now[strg_heat])
                          / model.para_strg_u[strg_heat] for strg_heat in model.heat_strg)) >= model.heat_demand

        def _demand_cool(model):
            return sum(model.M[sys_cool] * model.para_cp[sys_cool] for sys_cool in model.cool_out) \
                   + (sum(model.para_cp[strg_cool] * (model.beta_strg_next[strg_cool] - model.beta_strg_now[strg_cool])
                          / model.para_strg_u[strg_cool] for strg_cool in model.cool_strg)) >= model.cool_demand

        def _demand_elec(model):
            return sum(model.M[sys_elec] * model.para_cp[sys_elec] for sys_elec in model.elec_out) + model.grid \
                   + (sum(model.para_cp[strg_elec] * (model.beta_strg_next[strg_elec] - model.beta_strg_now[strg_elec])
                          / model.para_strg_u[strg_elec] for strg_elec in model.elec_strg)) >= model.elec_demand

        def _obj(model):
            return model.cost_fuel * sum(model.energy_in[ss] for ss in model.fuel_systems) * 3600 + \
                   model.cost_elec * (sum(model.energy_in[ss] for ss in model.grid_systems) + model.grid)/100

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

    def _assign_state(self, storage_state, mode):
        if mode == 'now':
            for idx, name in enumerate(self.sdp.strg_systems):
                self.sdp.beta_strg_now[name] = storage_state[idx]
        elif mode == 'next':
            for idx, name in enumerate(self.sdp.strg_systems):
                self.sdp.beta_strg_next[name] = storage_state[idx]
        elif mode == 'first':
            for idx, name in enumerate(self.sdp.strg_systems):
                self.sdp.beta_strg_next[name] = 0
        else:
            print('Something wrong')

    def _stage_scen_para(self, demands):
        self.sdp.elec_demand = demands[0]
        self.sdp.heat_demand = demands[1]
        self.sdp.cool_demand = demands[2]
        # self.sdp.cost_elec = cost_elec
        # self.sdp.para_T = temp

    def _subproblem(self, opt, t, sc, st_now, strg_system, conv_system):
        cost_now = +inf
        for st_next in itertools.product(*self.sdp.storages):
            self._assign_state(st_next, 'next')
            results = opt.solve(self.sdp, keepfiles=False, tee=False, report_timing=False)
            obj = pyo.value(self.sdp.obj)
            cost_temp = sum(self.sdp.cost[t + 1, s][st_next] * self.markov_prob[t + 1][sc][s]
                            for s in range(self.scen_number)) + obj
            if cost_temp <= cost_now:
                cost_now = cost_temp
                self.sdp.cost[t, sc][st_now] = cost_temp
                for idx, system in enumerate(strg_system):
                    system.cache_state[t, sc][st_now] = st_next[idx]
                for system in conv_system:
                    system.cache_beta[t, sc][st_now] = pyo.value(self.sdp.beta[system.name])
                    system.cache_alpha[t, sc][st_now] = pyo.value(self.sdp.alpha[system.name])

    def solve_sdp(self, cchp, markov_demands, markov_prob, solver_info=None, verbosity=0):
        """:param
        stream_solver = False  # True prints solver output to screen
        keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
        """
        if solver_info is None:
            solver_info = {'solver': "bonmin", 'solver_io': None, }
        opt = pyo.SolverFactory(solver_info['solver'], solver_io=solver_info['solver_io'])
        self.stage_number = len(markov_demands)
        self.scen_number = len(markov_demands[-1])
        self.sdp.storages = [strg.state for strg in cchp.strg_system]
        self.sdp.cost = np.zeros([self.stage_number + 1, self.scen_number])
        for i in range(len(self.sdp.storages)):
            self.sdp.cost = np.expand_dims(self.sdp.cost, axis=-1)
            self.sdp.cost = np.repeat(self.sdp.cost, len(self.sdp.storages[i]), axis=-1)
        markov_prob.append(np.ones([self.scen_number, self.scen_number]))
        self.markov_prob = markov_prob
        for system in cchp.strg_system + cchp.conv_system:
            system.generate_cache(self.stage_number, self.sdp.storages, self.scen_number)
        start0 = time.time()
        for t in range(self.stage_number - 1, 0, -1):
            start = time.time()
            print('Solve the {0}th stage problem'.format(str(t)))
            for st_now in itertools.product(*self.sdp.storages):
                self._assign_state(st_now, 'now')
                for sc in range(self.scen_number):
                    self._stage_scen_para(markov_demands[t][sc])
                    self._subproblem(opt, t, sc, st_now, cchp.strg_system, cchp.conv_system)
            end = time.time()
            print('  The used time for one time step is: {0:.2}'.format(end - start))
        self._assign_state([], 'first')
        self._stage_scen_para(markov_demands[0][0])
        self._subproblem(opt, 0, 0, st_now, cchp.strg_system, cchp.conv_system)
        end0 = time.time()
        state_iter = len(list(itertools.product(*self.sdp.storages)))
        time_average = (end0 - start0)/((self.stage_number-1)*self.scen_number*state_iter**2)
        return st_now, time_average