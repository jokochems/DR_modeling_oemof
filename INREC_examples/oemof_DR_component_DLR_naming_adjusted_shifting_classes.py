# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:32:12 2019

@author: Johannes Kochems, Julian Endres

Module for creating a Demand Response component.
Uses the formulation given in the PhD thesis Gils, Hans Christian (2015):
Balancing of Intermittent Renewable Power Generation by Demand Response and
Thermal Energy Storage, Stuttgart, http://dx.doi.org/10.18419/opus-6888,
accessed 16.08.2019, pp. 67-70.

The model formulation is used within the (GAMS-based) energy system model
REMix at DLR, Stuttgart.

Implementation is based on the implementation by Julian Endres from RLI
created within the WindNode project which has been taken over into the
oemof.solph.custom module.

NOTE: The terminology used in the oemof.solph.custom component is applied to
the implementation here. The original terms correspondent in the PhD thesis
are matched.

A special thank you goes to Julian Endres and the oemof developping team at RLI.
"""
import itertools

from numpy import mean

from collections import abc
from pyomo.core.base.block import SimpleBlock
from pyomo.environ import (Set, NonNegativeReals, Var, Constraint,
                           BuildAction, Expression)

from oemof.solph.network import Sink
from oemof.solph.plumbing import sequence
from oemof.solph import Investment


class SinkDR(Sink):
    r""" A special Sink component which modifies the input demand series used
    to model demand response units for load shifting and shedding.

    Parameters
    ----------
    capacity_down: int or array
        availability factor for downwards load shifts
        Corresponds to P_exist * s_flex[t] in the original terminology
    capacity_up: int
        availability factor for upwards load shifts
        Corresponds to P_exist * s_free[t] in the original terminology
    delay_time: int
        shiftig time (time until energy balance is levelled out again)
        Corresponds to t_shift in the original terminology
    shift_time: int
        duration of an upwards or downwards shift (half a shifting cycle
        if there is immediate compensation)
        Corresponds to t_interfere in the original terminology
    efficiency: float
        efficiency of a load shift
    ActivateYearLimit: boolean
        control parameter; activates constraints for year limit if set to True
    ActivateDayLimit: boolean
        control parameter; activates constraints for day limit if set to True
    cost_dsm_up:
        Variable costs for upwards load shifts
    cost_dsm_down_shift:
        Variable costs for upwards load shifts (load shifting)
    cost_dsm_down_shed:
        Variable costs for upwards load shifts (load shedding)
    n_yearLimit_shift: int
        maximum number of load shifts at full capacity per year, used to limit
        the amount of energy shifted per year. Optional parameter that is only
        needed when ActivateYearLimit is True
    n_yearLimit_shed: int
        maximum number of load sheds at full capacity per year, used to limit
        the amount of energy shedded per year. Mandatory parameter if load
        shedding is allowed by setting shed_eligibility to True
    t_dayLimit: int
        maximum duration of load shifts at full capacity per day, used to limit
        the amount of energy shifted per day. Optional parameter that is only
        needed when ActivateDayLimit is True
    demand: numeric
        original electrical demand
    shed_eligibility : :obj:`boolean`
        Boolean parameter indicating whether unit is eligible for
        load shedding
    shift_eligibility : :obj:`boolean`
        Boolean parameter indicating whether unit is eligible for
        load shifting
    addition : :obj:`boolean`
        Boolean parameter indicating whether or not to include additional
        constraint (which corresponds to Eq. 10 from Zerrahn and Schill (2015a)
    fixes : :obj:`boolean`
        Boolean parameter indicating whether or not to include additional
        fixes. These comprise prohibiting shifts which cannot be balanced within
        the optimization timeframe as well as DSM storage level roundtrip
        conditions
    """

    def __init__(self, demand, capacity_down, capacity_up,
                 delay_time, shift_time, shed_time,
                 cost_dsm_up=0, cost_dsm_down_shift=0,
                 cost_dsm_down_shed=0, efficiency=1,
                 ActivateYearLimit=False, ActivateDayLimit=False,
                 n_yearLimit_shift=None, n_yearLimit_shed=None,
                 t_dayLimit=None, addition=False, fixes=False,
                 shed_eligibility=True, shift_eligibility=True, **kwargs):
        super().__init__(**kwargs)

        self.capacity_down = sequence(capacity_down)
        self.capacity_up = sequence(capacity_up)
        self.demand = sequence(demand)

        # Introduce "flexible" delay_times: delay_time is an iterable here
        self.delay_time = [el for el in range(1, delay_time + 1)]
        self.shift_time = shift_time
        self.shed_time = shed_time
        self.cost_dsm_up = cost_dsm_up
        self.cost_dsm_down_shift = cost_dsm_down_shift
        self.cost_dsm_down_shed = cost_dsm_down_shed
        self.efficiency = efficiency

        # calculate mean values
        self.capacity_down_mean = mean(capacity_down)
        self.capacity_up_mean = mean(capacity_up)

        # Optionally include year resp. day limits for shifted / shedded energy
        # Introduction of these is controlled through boolean control parameters
        # both of which default to False
        self.ActivateYearLimit = ActivateYearLimit
        self.ActivateDayLimit = ActivateDayLimit
        self.n_yearLimit_shift = n_yearLimit_shift
        self.n_yearLimit_shed = n_yearLimit_shed
        self.t_dayLimit = t_dayLimit
        self.addition = addition
        self.fixes = fixes
        # For the sake of simplicity, shift_eligibility is always set to True at first
        self.shed_eligibility = shed_eligibility
        self.shift_eligibility = shift_eligibility

        self.investment = kwargs.get('investment')

        if self.ActivateYearLimit:
            if self.n_yearLimit_shift is None:
                raise ValueError('Parameter n_yearLimit_shift must be '
                                 'set if year limit is active.')

        if self.ActivateDayLimit:
            if self.t_dayLimit is None:
                raise ValueError('Parameter t_dayLimit must be '
                                 'set if day limit is active.')

        if self.shed_eligibility:
            if self.n_yearLimit_shed is None:
                raise ValueError('Parameter n_yearLimit_shed must be '
                                 'set if shed_eligibility is True.')

        # Check whether investment mode is active or not
        self._invest_group = isinstance(self.investment, Investment)

    def constraint_group(self):
        if self._invest_group is True:
            return SinkDRInvestmentBlock
        else:
            return SinkDRBlock


class SinkDRBlock(SimpleBlock):
    r"""Constraints for SinkDR

    **The following constraints are created:**

    .. _SinkDR-equations:

    .. math::
        &
        (1) \quad \dot{E}_{t} = demand_{t} + DSM_{t}^{up} + DSM_{t]^{balanceDown}
        - DSM_{t}^{down} + DSM_{t}^{balanceDown} \quad \forall t \in \mathbb{T} \\
        &
        (2) \quad DSM_{t}^{balanceDown} = \frac{ DSM_{t-t_{shift}}^{down}}{\eta}
        \quad \forall t \in \mathbb{T} \\
        &
        (3) \quad DSM_{t}^{balanceUp} = DSM_{t-t_{shift}}^{up} \cdot \eta
        \quad \forall t \in \mathbb{T} \\
        &
        (4) \quad DSM_{t}^{down} + DSM_{t}^{balanceUp} \leq capacity_{t}^{down}
        \quad \forall t \in \mathbb{T} \\
        &
        (5) \quad DSM_{t}^{up} + DSM_{t}^{balanceDown} \leq capacity^{up}
        \quad \forall t \in \mathbb{T} \\
        &
        (6) \quad DSM_{t}^{down} - DSM_{t}^{balanceDown} = W_{t}^{levelDown}
        - W_{t-1}^{levelRed} \quad \forall t \in \mathbb{T} \\
        &
        (7) \quad DSM_{t}^{up} - DSM_{t}^{balanceUp} = W_{t}^{levelUp}
        - W_{t-1}^{levelInc} \quad \forall t \in \mathbb{T} \\
        &
        (8) \quad W_{t}^{levelDown} \leq capacity_{t}^{down}
        \cdot \t_{interfere} \quad \forall t \in \mathbb{T} \\
        &
        (9) \quad W_{t}^{levelUp} \leq capacity_{t}^{up}
        \cdot \t_{interfere} \quad \forall t \in \mathbb{T} \\
        &
        (10) \quad \sum{t=0}^{T} DSM_{t}^{down} \leq P^{exist}
        \cdot \bar{s_{t}^{free}} \cdot \t_{interfere} \cdot n^{yearLimit}
        \quad \forall t \in \mathbb{T} \\ (optional constraint)
        &
        (11) \quad \sum{t=0}^{T} DSM_{t}^{up} \leq P^{exist}
        \cdot \bar{s_{t}^{free}} \cdot \t_{interfere} \cdot n^{yearLimit}
        \quad \forall t \in \mathbb{T} \\ (optional constraint)
        &
        (12) \quad DSM_{t}^{down} \leq P^{exist} \bar{s_{t}^{flex}}
        \cdot \t_{interfere} - \sum{t'=0}^{t_dayLimit-1} P_{t-t'}^{reduction}
        \quad \forall t \in \mathbb{T} \\ (optional constraint)
        &
        (13) \quad DSM_{t}^{up} \leq P^{exist} \bar{s_{t}^{free}}
        \cdot \t_{interfere} - \sum{t'=0}^{t_dayLimit-1} P_{t-t'}^{increase}
        \quad \forall t \in \mathbb{T} \\ (optional constraint)
        &
        (14) \quad DSM_{t}^{up} + DSM_{t}^{balanceDown}
        + DSM_{t}^{down} + DSM_{t}^{balanceUp}
        \leq max \{ capacity_{t}^{up}, capacity__{t}^{do} \}
        \quad \forall t \in \mathbb{T} \\ (optional constraint)
        &
    """
    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        if group is None:
            return None

        m = self.parent_block()

        # for all DR components get inflow from bus_elec
        for n in group:
            n.inflow = list(n.inputs)[0]

        #  ************* SETS *********************************

        # Set of DR Components
        self.DR = Set(initialize=[n for n in group])

        # Depict different delay_times per unit:
        # Do a mapping
        # Solution is based on this stack overflow issue:
        # https://stackoverflow.com/questions/59237082/variable-indexed-by-an-indexed-set-with-pyomo
        # accessed 13.08.2020
        map_DR_H = {k: v for k, v in zip([n for n in group],
                                         [n.delay_time for n in group])}

        unique_H = set(itertools.chain.from_iterable(map_DR_H.values()))
        self.H = Set(initialize=unique_H)

        self.DR_H = Set(within=self.DR * self.H,
                        initialize=[(dr, h) for dr in map_DR_H for h in map_DR_H[dr]])

        #  ************* VARIABLES *****************************

        # Variable load shift down (capacity)
        # Corresponds to P_reduction in the original terminology
        # self.dsm_do_shift = Var(self.DR, self.H, m.TIMESTEPS, initialize=0,
        self.dsm_do_shift = Var(self.DR_H, m.TIMESTEPS, initialize=0,
                                within=NonNegativeReals)

        # Variable for load shedding (capacity)
        # Corresponds to P_reduction in the original terminology
        # self.dsm_do_shed = Var(self.DR, self.H, m.TIMESTEPS, initialize=0,
        self.dsm_do_shed = Var(self.DR, m.TIMESTEPS, initialize=0,
                               within=NonNegativeReals)

        # Variable load shift up (capacity)
        # Corresponds to P_increase in the original terminology
        # self.dsm_up = Var(self.DR, self.H, m.TIMESTEPS, initialize=0,
        self.dsm_up = Var(self.DR_H, m.TIMESTEPS, initialize=0,
                          within=NonNegativeReals)

        # Variable balance load shift down through upwards shift (capacity)
        # Corresponds to P_balanceRed in the original terminology
        # self.balance_dsm_do = Var(self.DR, self.H, m.TIMESTEPS, initialize=0,
        self.balance_dsm_do = Var(self.DR_H, m.TIMESTEPS, initialize=0,
                                  within=NonNegativeReals)

        # Variable balance load shift up through downwards shift (capacity)
        # Corresponds to P_balanceInc in the original terminology
        # self.balance_dsm_up = Var(self.DR, self.H, m.TIMESTEPS, initialize=0,
        self.balance_dsm_up = Var(self.DR_H, m.TIMESTEPS, initialize=0,
                                  within=NonNegativeReals)

        # Variable fictious DR storage level for downwards load shifts (energy)
        # Corresponds to W_levelRed in the original terminology
        # self.dsm_do_level = Var(self.DR, m.TIMESTEPS, initialize=0,
        self.dsm_do_level = Var(self.DR, m.TIMESTEPS, initialize=0,
                                within=NonNegativeReals)

        # Variable fictious DR storage level for upwards load shifts (energy)
        # Corresponds to W_levelInc in the original terminology
        # self.dsm_up_level = Var(self.DR, m.TIMESTEPS, initialize=0,
        self.dsm_up_level = Var(self.DR, m.TIMESTEPS, initialize=0,
                                within=NonNegativeReals)

        #  ************* CONSTRAINTS *****************************

        def _shift_shed_vars_rule(block):
            """
            Force shifting resp. shedding variables to zero dependent
            on how boolean parameters for shift resp. shed eligibility
            are set.
            """
            for t in m.TIMESTEPS:
                for g in group:
                    for h in g.delay_time:

                        if not g.shift_eligibility:
                            # Memo: By forcing dsm_do_shift for shifting to zero, dsm up should
                            # implicitly be forced to zero as well, since otherwhise,
                            # constraints below would not hold ...
                            lhs = self.dsm_do_shift[g, h, t]
                            rhs = 0

                            block.shift_shed_vars.add((g, h, t), (lhs == rhs))

                        if not g.shed_eligibility:
                            lhs = self.dsm_do_shed[g, t]
                            rhs = 0

                            block.shift_shed_vars.add((g, h, t), (lhs == rhs))

        self.shift_shed_vars = Constraint(group, self.H, m.TIMESTEPS,
                                          noruleinit=True)
        self.shift_shed_vars_build = BuildAction(
            rule=_shift_shed_vars_rule)

        # Relation between inflow and effective Sink consumption
        def _input_output_relation_rule(block):
            """
            Relation between input data and pyomo variables.
            The actual demand after DR.
            Bus outflow == Demand +- DR (i.e. effective Sink consumption)
            """

            for t in m.TIMESTEPS:

                for g in group:

                    # outflow from bus
                    lhs = m.flow[g.inflow, g, t]

                    # Demand +- DR
                    rhs = g.demand[t] + sum(self.dsm_up[g, h, t]
                                            + self.balance_dsm_do[g, h, t]
                                            - self.dsm_do_shift[g, h, t]
                                            - self.balance_dsm_up[g, h, t]
                                            for h in g.delay_time) \
                          - self.dsm_do_shed[g, t]

                    # add constraint
                    block.input_output_relation.add((g, t), (lhs == rhs))

        self.input_output_relation = Constraint(group, m.TIMESTEPS,
                                                noruleinit=True)
        self.input_output_relation_build = BuildAction(
            rule=_input_output_relation_rule)

        # Equation 4.8
        def capacity_balance_red_rule(block):
            """
            Load reduction must be balanced by load increase within delay_time
            """
            for t in m.TIMESTEPS:
                for g in group:
                    for h in g.delay_time:

                        if g.shift_eligibility:

                            # main use case
                            if t >= h:
                                # balance load reduction
                                lhs = self.balance_dsm_do[g, h, t]

                                # load reduction (efficiency considered)
                                rhs = self.dsm_do_shift[g, h, t - h] / g.efficiency

                                # add constraint
                                block.capacity_balance_red.add((g, h, t), (lhs == rhs))

                            # no balancing for the first timestep
                            elif t == m.TIMESTEPS[1]:
                                lhs = self.balance_dsm_do[g, h, t]
                                rhs = 0

                                block.capacity_balance_red.add((g, h, t), (lhs == rhs))

                            else:
                                pass  # return(Constraint.Skip)

                        # if only shedding is possible, balancing variable can be forced to 0
                        else:
                            lhs = self.balance_dsm_do[g, h, t]
                            rhs = 0

                            block.capacity_balance_red.add((g, h, t), (lhs == rhs))

        self.capacity_balance_red = Constraint(group, self.H, m.TIMESTEPS,
                                               noruleinit=True)
        self.capacity_balance_red_build = BuildAction(
            rule=capacity_balance_red_rule)

        # Equation 4.9
        def capacity_balance_inc_rule(block):
            """
            Load increased must be balanced by load reduction within delay_time
            """
            for t in m.TIMESTEPS:
                for g in group:
                    for h in g.delay_time:

                        if g.shift_eligibility:

                            # main use case
                            if t >= h:
                                # balance load increase
                                lhs = self.balance_dsm_up[g, h, t]

                                # load increase (efficiency considered)
                                rhs = self.dsm_up[g, h, t - h] * g.efficiency

                                # add constraint
                                block.capacity_balance_inc.add((g, h, t), (lhs == rhs))

                            # no balancing for the first timestep
                            elif t == m.TIMESTEPS[1]:
                                lhs = self.balance_dsm_up[g, h, t]
                                rhs = 0

                                block.capacity_balance_inc.add((g, h, t), (lhs == rhs))

                            else:
                                pass  # return(Constraint.Skip)

                        # if only shedding is possible, balancing variable can be forced to 0
                        else:
                            lhs = self.balance_dsm_up[g, h, t]
                            rhs = 0

                            block.capacity_balance_inc.add((g, h, t), (lhs == rhs))

        self.capacity_balance_inc = Constraint(group, self.H, m.TIMESTEPS,
                                               noruleinit=True)
        self.capacity_balance_inc_build = BuildAction(
            rule=capacity_balance_inc_rule)

        # Own addition: prevent shifts which cannot be compensated
        def no_comp_red_rule(block):
            """
            Prevent downwards shifts that cannot be balanced anymore
            within the optimization timeframe
            """
            for t in m.TIMESTEPS:
                for g in group:

                    if g.fixes:
                        for h in g.delay_time:

                            if t > m.TIMESTEPS[-1] - h:
                                # no load reduction anymore (dsm_do_shift = 0)
                                lhs = self.dsm_do_shift[g, h, t]
                                rhs = 0
                                block.no_comp_red.add((g, h, t), (lhs == rhs))

                    else:
                        pass  # return(Constraint.Skip)

        self.no_comp_red = Constraint(group, self.H, m.TIMESTEPS,
                                      noruleinit=True)
        self.no_comp_red_build = BuildAction(
            rule=no_comp_red_rule)

        # Own addition: prevent shifts which cannot be compensated
        def no_comp_inc_rule(block):
            """
            Prevent upwards shifts that cannot be balanced anymore
            within the optimization timeframe
            """
            for t in m.TIMESTEPS:
                for g in group:

                    if  g.fixes:
                        for h in g.delay_time:

                            if t > m.TIMESTEPS[-1] - h:
                                # no load increase anymore (dsm_up = 0)
                                lhs = self.dsm_up[g, h, t]
                                rhs = 0
                                block.no_comp_inc.add((g, h, t), (lhs == rhs))

                    else:
                        pass  # return(Constraint.Skip)

        self.no_comp_inc = Constraint(group, self.H, m.TIMESTEPS,
                                      noruleinit=True)
        self.no_comp_inc_build = BuildAction(
            rule=no_comp_inc_rule)

        # Equation 4.11
        def availability_red_rule(block):
            """
            Load reduction must be smaller than or equal to the
            (time-dependent) capacity limit
            """

            for t in m.TIMESTEPS:
                for g in group:
                    # load reduction
                    lhs = sum(self.dsm_do_shift[g, h, t]
                              + self.balance_dsm_up[g, h, t]
                              for h in g.delay_time) \
                          + self.dsm_do_shed[g, t]

                    # upper bound
                    rhs = g.capacity_down[t]

                    # add constraint
                    block.availability_red.add((g, t), (lhs <= rhs))

        self.availability_red = Constraint(group, m.TIMESTEPS,
                                           noruleinit=True)
        self.availability_red_build = BuildAction(
            rule=availability_red_rule)

        # Equation 4.12
        def availability_inc_rule(block):
            """
            Load increase must be smaller than or equal to the
            (time-dependent) capacity limit
            """
            for t in m.TIMESTEPS:
                for g in group:
                    # load increase
                    lhs = sum(self.dsm_up[g, h, t]
                              + self.balance_dsm_do[g, h, t]
                              for h in g.delay_time)

                    # upper bound
                    rhs = g.capacity_up[t]

                    # add constraint
                    block.availability_inc.add((g, t), (lhs <= rhs))

        self.availability_inc = Constraint(group, m.TIMESTEPS,
                                           noruleinit=True)
        self.availability_inc_build = BuildAction(
            rule=availability_inc_rule)

        # Equation 4.13
        def dr_storage_red_rule(block):
            """
            Fictious demand response storage level for load reductions
            transition equation
            """
            for t in m.TIMESTEPS:
                for g in group:

                    # avoid timesteps prior to t = 0
                    if t > 0:
                        # reduction minus balancing of reductions
                        lhs = m.timeincrement[t] * sum((self.dsm_do_shift[g, h, t]
                                                        - self.balance_dsm_do[g, h, t]
                                                        * g.efficiency) for h in g.delay_time)

                        # load reduction storage level transition
                        rhs = self.dsm_do_level[g, t] - self.dsm_do_level[g, t - 1]

                        # add constraint
                        block.dr_storage_red.add((g, t), (lhs == rhs))

                    else:
                        # pass  # return(Constraint.Skip)
                        lhs = self.dsm_do_level[g, t]
                        rhs = m.timeincrement[t] * sum(self.dsm_do_shift[g, h, t]
                                                       for h in g.delay_time)
                        block.dr_storage_red.add((g, t), (lhs == rhs))

        self.dr_storage_red = Constraint(group, m.TIMESTEPS,
                                         noruleinit=True)
        self.dr_storage_red_build = BuildAction(
            rule=dr_storage_red_rule)

        # # Own addition: Storage roundtrip
        # def dr_storage_roundtrip_red_rule(block):
        #     """
        #     First and last storage level shall equal each other
        #
        #     NOTE: Seems like some redundancy is created here:
        #     If initial storage levels are set to 0 and it is
        #     demanded that there are no unbalanced shifts at the
        #     end, this roundtrip condition is not needed.
        #     """
        #     for g in group:
        #
        #         if g.fixes:
        #             # first storage level
        #             lhs = self.dsm_do_level[g, m.TIMESTEPS[1]]
        #
        #             # last storage level
        #             rhs = self.dsm_do_level[g, m.TIMESTEPS[-1]]
        #
        #             # add constraint
        #             block.dr_storage_roundtrip_red.add(g, (lhs == rhs))
        #
        #         else:
        #             pass  # return(Constraint.Skip)
        #
        # self.dr_storage_roundtrip_red = Constraint(group, noruleinit=True)
        # self.dr_storage_roundtrip_red_build = BuildAction(
        #     rule=dr_storage_roundtrip_red_rule)

        # Equation 4.14
        def dr_storage_inc_rule(block):
            """
            Fictious demand response storage level for load increase
            transition equation
            """
            for t in m.TIMESTEPS:
                for g in group:

                    # avoid timesteps prior to t = 0
                    if t > 0:
                        # increases minus balancing of reductions
                        lhs = m.timeincrement[t] * sum((self.dsm_up[g, h, t]
                                                        * g.efficiency
                                                        - self.balance_dsm_up[g, h, t])
                                                       for h in g.delay_time)

                        # load increase storage level transition
                        rhs = self.dsm_up_level[g, t] - self.dsm_up_level[g, t - 1]

                        # add constraint
                        block.dr_storage_inc.add((g, t), (lhs == rhs))

                    else:
                        # pass  # return(Constraint.Skip)
                        lhs = self.dsm_up_level[g, t]
                        rhs = m.timeincrement[t] * sum(self.dsm_up[g, h, t]
                                                       for h in g.delay_time)
                        block.dr_storage_inc.add((g, t), (lhs == rhs))

        self.dr_storage_inc = Constraint(group, m.TIMESTEPS,
                                         noruleinit=True)
        self.dr_storage_inc_build = BuildAction(
            rule=dr_storage_inc_rule)

        # # Own addition: Storage roundtrip
        # def dr_storage_roundtrip_inc_rule(block):
        #     """
        #     First and last storage level shall equal each other
        #
        #     NOTE: Seems like some redundancy is created here:
        #     If initial storage levels are set to 0 and it is
        #     demanded that there are no unbalanced shifts at the
        #     end, this roundtrip condition is not needed.
        #     """
        #     for g in group:
        #
        #         if g.fixes:
        #             # first storage level
        #             lhs = self.dsm_up_level[g, m.TIMESTEPS[1]]
        #
        #             # last storage level
        #             rhs = self.dsm_up_level[g, m.TIMESTEPS[-1]]
        #
        #             # add constraint
        #             block.dr_storage_roundtrip_inc.add(g, (lhs == rhs))
        #
        #         else:
        #             pass  # return(Constraint.Skip)
        #
        # self.dr_storage_roundtrip_inc = Constraint(group, noruleinit=True)
        # self.df_storage_roundtrip_inc_build = BuildAction(
        #     rule=dr_storage_roundtrip_inc_rule)

        # Equation 4.15
        def dr_storage_limit_red_rule(block):
            """
            Fictious demand response storage level for load reduction limit
            """
            for t in m.TIMESTEPS:
                for g in group:
                    # fictious demand response load reduction storage level
                    lhs = self.dsm_do_level[g, t]

                    # maximum (time-dependent) available shifting capacity
                    rhs = g.capacity_down_mean * g.shift_time

                    # add constraint
                    block.dr_storage_limit_red.add((g, t), (lhs <= rhs))

        self.dr_storage_limit_red = Constraint(group, m.TIMESTEPS,
                                               noruleinit=True)
        self.dr_storage_level_red_build = BuildAction(
            rule=dr_storage_limit_red_rule)

        # Equation 4.16
        def dr_storage_limit_inc_rule(block):
            """
            Fictious demand response storage level for load increase limit
            """
            for t in m.TIMESTEPS:
                for g in group:
                    # fictious demand response load reduction storage level
                    lhs = self.dsm_up_level[g, t]

                    # maximum (time-dependent) available shifting capacity
                    rhs = g.capacity_up_mean * g.shift_time

                    # add constraint
                    block.dr_storage_limit_inc.add((g, t), (lhs <= rhs))

        self.dr_storage_limit_inc = Constraint(group, m.TIMESTEPS,
                                               noruleinit=True)
        self.dr_storage_level_inc_build = BuildAction(
            rule=dr_storage_limit_inc_rule)

        # Equation 4.17' -> load shedding
        def dr_yearly_limit_shed_rule(block):
            """
            Introduce overall annual (energy) limit for load shedding resp.
            overall limit for optimization timeframe considered
            A year limit in contrast to Gils (2015) is defined a mandatory
            parameter here in order to achieve an approach comparable
            to the others.
            """
            for g in group:
                # sum of all load redutions
                lhs = sum(self.dsm_do_shed[g, t]
                          for t in m.TIMESTEPS)

                # year limit
                rhs = g.capacity_down_mean * g.shed_time \
                      * g.n_yearLimit_shed

                # add constraint
                block.dr_yearly_limit_shed.add(g, (lhs <= rhs))

        self.dr_yearly_limit_shed = Constraint(group, noruleinit=True)
        self.dr_yearly_limit_shed_build = BuildAction(
            rule=dr_yearly_limit_shed_rule)

        # ************* Optional Constraints *****************************

        # Equation 4.17
        def dr_yearly_limit_red_rule(block):
            """
            Introduce overall annual (energy) limit for load reductions resp.
            overall limit for optimization timeframe considered
            """
            for g in group:

                if g.ActivateYearLimit:
                    # sum of all load redutions
                    lhs = sum(sum(self.dsm_do_shift[g, h, t]
                                  for h in g.delay_time)
                              for t in m.TIMESTEPS)

                    # year limit
                    rhs = g.capacity_down_mean * g.shift_time \
                          * g.n_yearLimit_shift

                    # add constraint
                    block.dr_yearly_limit_red.add(g, (lhs <= rhs))

                else:
                    pass  # return(Constraint.Skip)

        self.dr_yearly_limit_red = Constraint(group, noruleinit=True)
        self.dr_yearly_limit_red_build = BuildAction(
            rule=dr_yearly_limit_red_rule)

        # Equation 4.18
        def dr_yearly_limit_inc_rule(block):
            """
            Introduce overall annual (energy) limit for load increases resp.
            overall limit for optimization timeframe considered
            """
            for g in group:

                if g.ActivateYearLimit:
                    # sum of all load increases
                    lhs = sum(sum(self.dsm_up[g, h, t]
                                  for h in g.delay_time)
                              for t in m.TIMESTEPS)

                    # year limit
                    rhs = g.capacity_up_mean * g.shift_time \
                          * g.n_yearLimit_shift

                    # add constraint
                    block.dr_yearly_limit_inc.add(g, (lhs <= rhs))

                else:
                    pass  # return(Constraint.Skip)

        self.dr_yearly_limit_inc = Constraint(group, noruleinit=True)
        self.dr_yearly_limit_inc_build = BuildAction(
            rule=dr_yearly_limit_inc_rule)

        # Equation 4.19
        def dr_daily_limit_red_rule(block):
            """
            Introduce rolling (energy) limit for load reductions
            This effectively limits DR utalization dependent on
            activations within previous hours.

            Note: This effectively limits downshift in the last
            hour of a time span to the remaining share of an
            average downshift.
            """
            for t in m.TIMESTEPS:
                for g in group:

                    if g.ActivateDayLimit:

                        # main use case
                        if t >= g.t_dayLimit:

                            # load reduction
                            lhs = sum(self.dsm_do_shift[g, h, t]
                                      for h in g.delay_time)

                            # daily limit
                            rhs = g.capacity_down_mean * g.shift_time \
                                  - sum(sum(self.dsm_do_shift[g, h, t - t_dash]
                                            for h in g.delay_time)
                                        for t_dash in range(1, int(g.t_dayLimit)+1))

                            # add constraint
                            block.dr_daily_limit_red.add((g, t), (lhs <= rhs))

                        else:
                            pass  # return(Constraint.Skip)

                    else:
                        pass  # return(Constraint.Skip)

        self.dr_daily_limit_red = Constraint(group, m.TIMESTEPS,
                                             noruleinit=True)
        self.dr_daily_limit_red_build = BuildAction(
            rule=dr_daily_limit_red_rule)

        # Equation 4.20
        def dr_daily_limit_inc_rule(block):
            """
            Introduce rolling (energy) limit for load increases
            This effectively limits DR utalization dependent on
            activations within previous hours.

            Note: This effectively limits upshift in the last
            hour of a time span to the remaining share of an
            average upshift.
            """
            for t in m.TIMESTEPS:
                for g in group:

                    if g.ActivateDayLimit:

                        # main use case
                        if t >= g.t_dayLimit:

                            # load increase
                            lhs = sum(self.dsm_up[g, h, t]
                                      for h in g.delay_time)

                            # daily limit
                            rhs = g.capacity_up_mean * g.shift_time \
                                  - sum(sum(self.dsm_up[g, h, t - t_dash]
                                            for h in g.delay_time)
                                        for t_dash in range(1, int(g.t_dayLimit)+1))

                            # add constraint
                            block.dr_daily_limit_inc.add((g, t), (lhs <= rhs))

                        else:
                            pass  # return(Constraint.Skip)

                    else:
                        pass  # return(Constraint.Skip)

        self.dr_daily_limit_inc = Constraint(group, m.TIMESTEPS,
                                             noruleinit=True)
        self.dr_daily_limit_inc_build = BuildAction(
            rule=dr_daily_limit_inc_rule)

        # Own addition (optional)
        def dr_logical_constraint_rule(block):
            """
            Similar to equation 10 from Zerrahn and Schill (2015):
            The sum of upwards and downwards shifts may not be greater than the
            (bigger) capacity limit.
            """
            for t in m.TIMESTEPS:
                for g in group:

                    if g.addition:

                        # sum of load increases and reductions
                        lhs = sum(self.dsm_up[g, h, t]
                                  + self.balance_dsm_do[g, h, t]
                                  + self.dsm_do_shift[g, h, t]
                                  + self.balance_dsm_up[g, h, t]
                                  for h in g.delay_time) \
                              + self.dsm_do_shed[g, t]

                        # maximum capacity eligibly for load shifting
                        rhs = max(g.capacity_down[t],
                                  g.capacity_up[t])

                        # add constraint
                        block.dr_logical_constraint.add((g, t), (lhs <= rhs))

                    else:
                        pass  # return(Constraint.Skip)

        self.dr_logical_constraint = Constraint(group, m.TIMESTEPS,
                                                noruleinit=True)
        self.dr_logical_constraint_build = BuildAction(
            rule=dr_logical_constraint_rule)

    # Equation 4.23
    def _objective_expression(self):
        r""" Objective expression for all DR shift units with fixed costs
        and variable costs; Equation 4.23 from Gils (2015)
        """
        m = self.parent_block()

        dr_cost = 0

        # Costs only occur from initial shifting, not from balancing
        for t in m.TIMESTEPS:
            for g in self.DR:
                dr_cost += sum(self.dsm_up[g, h, t]
                               + self.balance_dsm_do[g, h, t]
                               for h in g.delay_time) * g.cost_dsm_up
                dr_cost += sum(self.dsm_do_shift[g, h, t]
                               + self.balance_dsm_up[g, h, t]
                               for h in g.delay_time) * g.cost_dsm_down_shift \
                           + self.dsm_do_shed[g, t] * g.cost_dsm_down_shed

        self.cost = Expression(expr=dr_cost)

        return self.cost


# TODO: Add Investment possibility here
class SinkDRInvestmentBlock(SinkDRBlock):
    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        super()._create()

        if group is None:
            return None

        m = self.parent_block()

        # for all DR components get inflow from bus_elec
        for n in group:
            n.inflow = list(n.inputs)[0]

        #  ************* SETS *********************************

        self.INVESTDR = Set(initialize=[n for n in group])

        #  ************* VARIABLES *****************************

        # Define bounds for investments in demand response
        def _dr_investvar_bound_rule(block, g):
            """Rule definition to bound the invested demand response capacity `invest`.
            """
            return g.investment.minimum, g.investment.maximum

        # Investment in DR capacity
        self.invest = Var(self.INVESTDR, m.TIMESTEPS, initialize=0,
                          within=NonNegativeReals,
                          bounds=_dr_investvar_bound_rule)

        # TODO: RESUME HERE
        # ... To be continued -> See oemof.solph.components.GenericInvestmentStorageBlock
        # for the main features to be covered and include constraints from above.
        # constraints and features may probably be inherited, so there probably
        # is no need to define everything from the scractch again.

    # 17.08.2019, JK: Taken from oemof.solph.components.GenericInvestmentStorageBlock
    def _objective_expression(self):
        r""" Objective expression with fixed and investement costs.
        """
        if not hasattr(self, 'INVESTDR'):
            return 0

        investment_costs = 0

        for n in self.INVESTDR:
            if n.investment.ep_costs is not None:
                investment_costs += self.invest[n] * n.investment.ep_costs
            else:
                raise ValueError("Missing value for investment costs!")

        self.investment_costs = Expression(expr=investment_costs)

        return investment_costs