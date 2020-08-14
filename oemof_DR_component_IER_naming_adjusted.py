# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:32:12 2019

@author: Johannes Kochems, Julian Endres

Module for creating a Demand Response component.
Uses the formulation given in the PhD thesis Steurer, Martin (2017): 
Analyse von Demand Side Integration im Hinblick auf eine effiziente und 
umweltfreundliche Energieversorgung, Stuttgart, 10.18419/opus-9181,
accessed 17.08.2019, pp. 80-82.

The model formulation is used within the (GAMS-based) power market model
E2M2(s) at IER, Stuttgart.

NOTE: The PhD thesis definitely does not contain the complete modelling approach.
The equations are generally formulated but correspond to one for upwards as
well as one for downwards load adjustments.

Implementation is based on the implementation by Julian Endres from RLI
created within the WindNode project which has been taken over into the
oemof.solph.custom module.

NOTE: The terminology used in the oemof.solph.custom component is applied to
the implementation here. The original terms correspondent in the PhD thesis
are matched. A positive load adjustment corresponds to a downwards load shift since
the terms are used as in balancing power markets since the same effect could
have been achieved by increasing the power output of a generator unit.

A special thank you goes to Julian Endres and the oemof developing team at RLI.
"""

from pyomo.core.base.block import SimpleBlock
from pyomo.environ import (Set, NonNegativeReals, Var, Constraint,
                           BuildAction, Expression)

from oemof.solph.network import Sink
from oemof.solph.plumbing import sequence


class SinkDSI(Sink):
    r"""
    Demand Side Integration Sink component which modifies the input demand series.

    Parameters
    ----------
    demand: numeric
        original electrical demand
    capacity_up: int or array
        maximum DSM capacity that may be increased
        Corresponds to P_neg_max * f_v_neg[t] in the original terminology
    capacity_down: int or array
        maximum DSM capacity that may be reduced
        Corresponds to P_pos_max * f_v_pos[t] in the original terminology
    delay_time: int
        shifting time for one load shifting cycle (time until energy balance
        has to be levelled out again)
        Corresponds to d_v in the original terminology
    shift_time_up: int
        switching time for load shifting unit (time for one load
        upwards adjustment <= half a shifting cycle)
        Corresponds to d_s_neg in the original terminology
    shift_time_down: int
        switching time for load shifting unit (time for one load
        downwards adjustment <= half a shifting cycle)
        Corresponds to d_s_pos in the original terminology
    efficiency: float
        efficiency of load shifting unit
    cumulative_shift_time: int
        overall duration of shift processes during the optimization timeframe (one year)
        used to limit the overall amount of energy shifted
        Corresponds to d_kum in the original terminology
    cost_dsm_up: float
        Variable costs for upwards load shifts
    cost_dsm_down_shift: float
        Variable costs for downwards load shifts (load shifting)
    cost_dsm_down_shed: float
        Variable costs for downwards load shifts (load shedding)
    addition: boolean
        Boolean parameter indicating, whether or not to use an own additional
        constraint similar to Equation 10 from Zerrahn and Schill (2015)
    shed_eligibility : :obj:`boolean`
        Boolean parameter indicating whether unit is eligible for
        load shedding
    shift_eligibility : :obj:`boolean`
        Boolean parameter indicating whether unit is eligible for
        load shifting
    """

    def __init__(self, demand, capacity_up, capacity_down,
                 delay_time, shift_time_up, shift_time_down, shed_time,
                 cumulative_shift_time, cumulative_shed_time,
                 cost_dsm_up=0, cost_dsm_down_shift=0,
                 cost_dsm_down_shed=0, efficiency=1,
                 addition=False, shed_eligibility=True,
                 shift_eligibility=True, **kwargs):
        super().__init__(**kwargs)
        
        self.capacity_down = sequence(capacity_down)
        self.capacity_up = sequence(capacity_up)
        self.demand = sequence(demand)
        self.delay_time = delay_time
        self.shift_time_down = shift_time_down
        self.shift_time_up = shift_time_up
        self.shed_time = shed_time
        self.cumulative_shift_time = cumulative_shift_time
        self.cumulative_shed_time = cumulative_shed_time
        self.cost_dsm_up = cost_dsm_up
        self.cost_dsm_down_shift = cost_dsm_down_shift
        self.cost_dsm_down_shed = cost_dsm_down_shed
        self.efficiency = efficiency
        self.addition = addition
        self.shed_eligibility = shed_eligibility
        self.shift_eligibility = shift_eligibility

    def constraint_group(self):
        return SinkDSIBlock


class SinkDSIBlock(SimpleBlock):
    r"""Constraints for SinkDSI

        **The following constraints are created:**

        .. _SinkDSI-equations:

        .. math::
            &
            (1) \quad \dot{E}_{t} = demand_{t} + DSM_{t}^{up} - DSM_{t}^{down}
            \quad \forall t \in \mathbb{T}\\
            &
            (2) \quad  DSM_{t}^{up} \leq E_t^{up}
            \quad \forall t \in \mathbb{T}\\
            &
            (3) \quad  DSM_{t}^{down} \leq E_t^{down}}
            \quad \forall t \in \mathbb{T}\\
            &
            (4) \quad  \sum_{t}^{t + L} DSM_{t}^{down} =
            \sum_{t}^{t + L} DSM_{t}^{up} \cdot \eta
            \quad \forall t \in \mathbb{T}\\
            &
            (5) \quad  \sum_{t}^{t + L} DSM_{t}^{up}
            \leq d^{s,up} \cdot E_{t}^{up}
            \quad \forall t \in \mathbb{T}\\
            &
            (6) \quad  \sum_{t}^{t + L} DSM_{t}^{down}
            \leq d^{s,down} \cdot E_{t}^{down}
            \quad \forall t \in \mathbb{T}\\
            &
            (7) \quad  \sum_{t=0}^{T} DSM_{t}^{up}
            \leq d^{cum} \cdot E_^{up,max}\\
            &
            (8) \quad  \sum_{t=0}^{T} DSM_{t}^{down}
            \leq d^{cum} \cdot E_^{down,max}\\
            &
            (9) \quad  DSM_{t}^{down} + DSM_{t}^{up}
            \leq max \{E_t^{down}, E_t^{up}\}
            \quad \forall t \in \mathbb{T}\\ (optional constraint)
            &

        """
    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        if group is None:
            return None

        m = self.parent_block()

        # for all DSI components get inflow from bus_elec
        for n in group:
            n.inflow = list(n.inputs)[0]

        #  ************* SETS **********************************

        # Set of DSI Components
        self.DSI = Set(initialize=[n for n in group])

        #  ************* VARIABLES *****************************

        # Variable load shift down
        self.dsm_do_shift = Var(self.DSI, m.TIMESTEPS, initialize=0,
                                within=NonNegativeReals)

        # Variable for load shedding
        self.dsm_do_shed = Var(self.DSI, m.TIMESTEPS, initialize=0,
                               within=NonNegativeReals)

        # Variable load shift up
        self.dsm_up = Var(self.DSI, m.TIMESTEPS, initialize=0,
                          within=NonNegativeReals)

        #  ************* CONSTRAINTS ***************************

        def _shift_shed_vars_rule(block):
            """
            Force shifting resp. shedding variables to zero dependent
            on how boolean parameters for shift resp. shed eligibility
            are set.
            """
            for t in m.TIMESTEPS:
                for g in group:

                    if not g.shift_eligibility:
                        # Memo: By forcing dsm_do_shift for shifting to zero, dsm up should
                        # implicitly be forced to zero as well, since otherwhise,
                        # constraints below would not hold ...
                        lhs = self.dsm_do_shift[g, t]
                        rhs = 0

                        block.shift_shed_vars.add((g, t), (lhs == rhs))

                    if not g.shed_eligibility:
                        lhs = self.dsm_do_shed[g, t]
                        rhs = 0

                        block.shift_shed_vars.add((g, t), (lhs == rhs))

        self.shift_shed_vars = Constraint(group, m.TIMESTEPS,
                                          noruleinit=True)
        self.shift_shed_vars_build = BuildAction(
            rule=_shift_shed_vars_rule)

        # Relation between inflow and effective Sink consumption
        def _input_output_relation_rule(block):
            """
            Relation between input data and pyomo variables.
            The actual demand after DSI.
            Bus outflow == Demand +- DSI (i.e. effective Sink consumption)
            """

            for t in m.TIMESTEPS:
                for g in group:
                    # Sink inflow (i.e. bus outflow)
                    lhs = m.flow[g.inflow, g, t]

                    # Demand + DSI_up - DSI_down
                    rhs = g.demand[t] - self.dsm_do_shift[g, t] \
                          + self.dsm_up[g, t] - self.dsm_do_shed[g, t]

                    # add constraint
                    block.input_output_relation.add((g, t), (lhs == rhs))

        self.input_output_relation = Constraint(group, m.TIMESTEPS,
                                                noruleinit=True)
        self.input_output_relation_build = BuildAction(
            rule=_input_output_relation_rule)

        # Equation 4-2 (down)
        def dsi_availability_down_rule(block):
            """
            Equation 4-2 from Steurer 2017:
            Every downwards load shift must be smaller than or equal
            to the maximum (time-dependent) capacity limit.
            """

            for t in m.TIMESTEPS:
                for g in group:

                    # DSI down
                    lhs = self.dsm_do_shift[g, t] + self.dsm_do_shed[g, t]

                    # capacity limit
                    rhs = g.capacity_down[t]

                    # add constraint
                    block.dsi_availability_down.add((g, t), (lhs <= rhs))

        self.dsi_availability_down = Constraint(group, m.TIMESTEPS,
                                                noruleinit=True)
        self.dsi_availability_down_build = BuildAction(
            rule=dsi_availability_down_rule)

        # Equation 4-2 (up)
        def dsi_availability_up_rule(block):
            """
            Equation 4-2 from Steurer 2017:
            Every upwards load shift must be smaller than or equal
            to the maximum (time-dependent) capacity limit.
            """

            for t in m.TIMESTEPS:
                for g in group:

                    # DSI up
                    lhs = self.dsm_up[g, t]

                    # capacity limit
                    rhs = g.capacity_up[t]

                    # add constraint
                    block.dsi_availability_up.add((g, t), (lhs <= rhs))

        self.dsi_availability_up = Constraint(group, m.TIMESTEPS,
                                              noruleinit=True)
        self.dsi_availability_up_build = BuildAction(
            rule=dsi_availability_up_rule)

        # Equation 4-4
        def dsi_energy_balance_rule(block):
            """
            Equation 4-4 from Steurer 2017:
            Downwards load shifts must be equal to upwards ones within a given
            shifting time, i.e. the energy balance must be levelled out
            """

            for t in m.TIMESTEPS:
                for g in group:

                    # main use case
                    if t <= m.TIMESTEPS[-1] - g.delay_time:
                        
                        # DSI down
                        lhs = sum(self.dsm_do_shift[g, tt]
                                  for tt in range(t, t + g.delay_time + 1))

                        # DSI up
                        rhs = sum(self.dsm_up[g, tt] * g.efficiency
                                  for tt in range(t, t + g.delay_time + 1))

                        # add constraint
                        block.dsi_energy_balance.add((g, t), (lhs == rhs))

                    else:

                        # DSI down
                        lhs = sum(self.dsm_do_shift[g, tt]
                                  for tt in range(t, m.TIMESTEPS[-1] + 1))

                        # DSI up
                        rhs = sum(self.dsm_up[g, tt] * g.efficiency
                                  for tt in range(t, m.TIMESTEPS[-1] + 1))

                        # add constraint
                        block.dsi_energy_balance.add((g, t), (lhs == rhs))

        self.dsi_energy_balance = Constraint(group, m.TIMESTEPS,
                                             noruleinit=True)
        self.dsi_energy_balance_build = BuildAction(
            rule=dsi_energy_balance_rule)

        # Equation 4-5 (down)
        def dsi_shifting_limit_down_rule(block):
            """
            Equation 4-5 from Steurer 2017:
            Limit the amount of energy shifted per load shifting cycle
            for downwards adjustments.
            m.timeincrement is used to limit an amount of energy,
            not capacity.
            """

            for t in m.TIMESTEPS:
                for g in group:

                    # main use case
                    if t <= m.TIMESTEPS[-1] - g.delay_time:
                        
                        # DSI down
                        lhs = sum(self.dsm_do_shift[g, t] * m.timeincrement[t]
                                  for t in range(t, t + g.delay_time + 1))

                        # maximum energy to be shifted
                        rhs = g.shift_time_down * max(g.capacity_down[t]
                                                      for t in m.TIMESTEPS)

                        # add constraint
                        block.dsi_shifting_limit_down.add((g, t), (lhs <= rhs))

                    else:

                        # DSI down
                        lhs = sum(self.dsm_do_shift[g, t] * m.timeincrement[t]
                                  for t in range(t, m.TIMESTEPS[-1] + 1))

                        # maximum energy to be shifted
                        rhs = min(g.shift_time_down,
                                  (m.TIMESTEPS[-1] - t)+1) * max(g.capacity_down[t]
                                                      for t in m.TIMESTEPS)

                        # add constraint
                        block.dsi_shifting_limit_down.add((g, t), (lhs <= rhs))

        self.dsi_shifting_limit_down = Constraint(group, m.TIMESTEPS,
                                                  noruleinit=True)
        self.dsi_shifting_limit_down_build = BuildAction(
            rule=dsi_shifting_limit_down_rule)

        # Equation 4-5' (shedding)
        def dsi_shedding_limit_rule(block):
            """
            Similar to Equation 4-5 from Steurer 2017 (Equation 4-5'):
            Limit the amount of energy shedded.
            m.timeincrement is used to limit an amount of energy,
            not capacity.
            """

            for t in m.TIMESTEPS:
                for g in group:

                    # main use case
                    if t <= m.TIMESTEPS[-1] - g.shed_time:

                        # DSI down
                        lhs = sum(self.dsm_do_shed[g, t] * m.timeincrement[t]
                                  for t in range(t, t + g.shed_time + 1))

                        # maximum energy to be shifted
                        rhs = g.shed_time * max(g.capacity_down[t]
                                                for t in m.TIMESTEPS)

                        # add constraint
                        block.dsi_shedding_limit.add((g, t), (lhs <= rhs))

                    else:

                        # DSI down
                        lhs = sum(self.dsm_do_shed[g, t] * m.timeincrement[t]
                                  for t in range(t, m.TIMESTEPS[-1] + 1))

                        # maximum energy to be shifted
                        rhs = min(g.shed_time,
                                  (m.TIMESTEPS[-1] - t)+1) * max(g.capacity_down[t]
                                                      for t in m.TIMESTEPS)

                        # add constraint
                        block.dsi_shedding_limit.add((g, t), (lhs <= rhs))

        self.dsi_shedding_limit = Constraint(group, m.TIMESTEPS,
                                                  noruleinit=True)
        self.dsi_shedding_limit_build = BuildAction(
            rule=dsi_shedding_limit_rule)

        # Equation 4-5 (up)
        def dsi_shifting_limit_up_rule(block):
            """
            Equation 4-5 from Steurer 2017:
            Limit the amount of energy shifted per load shifting cycle for
            updwards adjustments.
            m.timeincrement is used to limit an amount of energy,
            not capacity.
            """

            for t in m.TIMESTEPS:
                for g in group:

                    # main use case
                    if t <= m.TIMESTEPS[-1] - g.delay_time:
                        
                        # DSI up
                        lhs = sum(self.dsm_up[g, t] * m.timeincrement[t]
                                  for t in range(t, t + g.delay_time + 1))

                        # maximum energy to be shifted
                        rhs = g.shift_time_up * max(g.capacity_up[t]
                                                    for t in m.TIMESTEPS)

                        # add constraint
                        block.dsi_shifting_limit_up.add((g, t), (lhs <= rhs))

                    else:

                        # DSI up
                        lhs = sum(self.dsm_up[g, t] * m.timeincrement[t]
                                  for t in range(t, m.TIMESTEPS[-1] + 1))

                        # maximum energy to be shifted
                        rhs = min(g.shift_time_up,
                                  (m.TIMESTEPS[-1] - t)+1) * max(g.capacity_up[t]
                                                      for t in m.TIMESTEPS)

                        # add constraint
                        block.dsi_shifting_limit_up.add((g, t), (lhs <= rhs))

        self.dsi_shifting_limit_up = Constraint(group, m.TIMESTEPS,
                                                noruleinit=True)
        self.dsi_shifting_limit_up_build = BuildAction(
            rule=dsi_shifting_limit_up_rule)
        
        # Equation 4-6 (down)
        def dsi_overall_energy_limit_down_rule(block):
            """
            Equation 4-6 from Steurer 2017:
            Limit the overall amount of energy to be shifted within one year resp.
            within the overall optimization timeframe.

            NOTE: Steurer 2017 says that this limit has to
            be defined for every timestep. This would lead to a highly redundant
            formulation and therefore is omitted here. Instead, one constraint
            for every simulation timeframe (year) does make much more sense.
            """
            
            for g in group:
                
                # DSI down
                lhs = sum(self.dsm_do_shift[g, t] * m.timeincrement[t]
                          for t in range(m.TIMESTEPS[1],
                                         m.TIMESTEPS[-1] + 1))

                # maximum energy to be shifted
                rhs = g.cumulative_shift_time * max(g.capacity_down[t]
                                                    for t in m.TIMESTEPS)

                # add constraint
                block.dsi_overall_energy_limit_down.add((g), (lhs <= rhs))

        self.dsi_overall_energy_limit_down = Constraint(group, noruleinit=True)
        self.dsi_overall_energy_limit_down_build = BuildAction(
            rule=dsi_overall_energy_limit_down_rule)

        # Equation 4-6' (shedding)
        def dsi_overall_shedding_limit_rule(block):
            """
            Similar to Equation 4-6 from Steurer 2017:
            Limit the overall amount of energy to be shedded within one year resp.
            within the overall optimization timeframe.

            NOTE: Steurer 2017 says that this limit has to
            be defined for every timestep. This would lead to a highly redundant
            formulation and therefore is omitted here. Instead, one constraint
            for every simulation timeframe (year) does make much more sense.
            """

            for g in group:
                # DSI down
                lhs = sum(self.dsm_do_shed[g, t] * m.timeincrement[t]
                          for t in range(m.TIMESTEPS[1],
                                         m.TIMESTEPS[-1] + 1))

                # maximum energy to be shifted
                rhs = g.cumulative_shed_time * max(g.capacity_down[t]
                                                    for t in m.TIMESTEPS)

                # add constraint
                block.dsi_overall_shedding_limit.add((g), (lhs <= rhs))

        self.dsi_overall_shedding_limit = Constraint(group, noruleinit=True)
        self.dsi_overall_shedding_limit_build = BuildAction(
            rule=dsi_overall_shedding_limit_rule)
        
        # Equation 4-6 (up)
        def dsi_overall_energy_limit_up_rule(block):
            """
            Equation 4-6 from Steurer 2017:
            Limit the overall amount of energy to be shifted within one year resp.
            within the overall optimization timeframe.

            NOTE: Steurer 2017 says that this limit has to
            be defined for every timestep. This would lead to a highly redundant
            formulation and therefore is omitted here. Instead, one constraint
            for every simulation timeframe (year) does make much more sense.
            """
            
            for g in group:
                
                # DSI down
                lhs = sum(self.dsm_up[g, t]
                          for t in range(m.TIMESTEPS[1],
                                         m.TIMESTEPS[-1] + 1))

                # maximum energy to be shifted
                rhs = g.cumulative_shift_time * max(g.capacity_up[t]
                                                    for t in m.TIMESTEPS)

                # add constraint
                block.dsi_overall_energy_limit_up.add((g), (lhs <= rhs))

        self.dsi_overall_energy_limit_up = Constraint(group, noruleinit=True)
        self.dsi_overall_energy_limit_up_build = BuildAction(
            rule=dsi_overall_energy_limit_up_rule)

        # Own addition (optional)
        def dsi_logic_rule(block):
            """
            Similar to equation 10 from Zerrahn and Schill (2015):
            The sum of upwards and downwards shifts may not be greater than the
            (bigger) capacity limit.
            """
            
            for t in m.TIMESTEPS:
                for g in group:

                    if g.addition:

                        # DSM up/down
                        lhs = self.dsm_do_shift[g, t] + self.dsm_up[g, t] \
                              + self.dsm_do_shed[g, t]

                        # max capacity at timestep t
                        rhs = max(g.capacity_down[t],
                                  g.capacity_up[t])

                        # add constraint
                        block.dsi_logic.add((g, t), (lhs <= rhs))

                    else:
                        pass  # return(Constraint.Skip)

        self.dsi_logic = Constraint(group, m.TIMESTEPS,
                                    noruleinit=True)
        self.dsi_logic_build = BuildAction(
            rule=dsi_logic_rule)

    def _objective_expression(self):
        """Adding cost terms for DSI activity to obj. function"""

        m = self.parent_block()

        dsi_cost = 0

        for t in m.TIMESTEPS:
            for g in self.DSI:
                dsi_cost += self.dsm_up[g, t] * g.cost_dsm_up
                dsi_cost += self.dsm_do_shift[g, t] * g.cost_dsm_down_shift \
                            + self.dsm_do_shed[g, t] * g.cost_dsm_down_shed

        self.cost = Expression(expr=dsi_cost)

        return self.cost

class SinkDSIInvestmentBlock(SinkDSIBlock):
    pass