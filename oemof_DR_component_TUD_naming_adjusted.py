# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:32:12 2019

@author: Johannes Kochems, Julian Endres

Module for creating a Demand Response component.
Uses the formulation given in the PhD thesis Ladwig, Theresa (2018):
Demand Side Management in Deutschland zur Systemintegration erneuerbarer
Energien, Dresden, https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa-236074,
accessed 02.05.2020, pp. 90-93. The constraints only applicable to P2X
technologies have been ommitted here.

The model formulation is used within the (GAMS-based) power market model
ELTRAMOD at chair energy economics (EE^2) of TU Dresden.

Implementation is based on the implementation by Julian Endres from RLI
created within the WindNode project which has been taken over into the
oemof.solph.custom module.

NOTE: The terminology used in the oemof.solph.custom component is applied to
the implementation here. The original terms correspondent in the PhD thesis
are matched.

A special thank you goes to Julian Endres and the oemof developing team at RLI.
"""

from pyomo.core.base.block import SimpleBlock
from pyomo.environ import (Set, NonNegativeReals, Reals, Var, Constraint,
                           BuildAction, Expression)

from oemof.solph.network import Sink
from oemof.solph.plumbing import sequence


class SinkDSM(Sink):
    r"""
    Demand Side Management Sink component which modifies the input demand series.

    Parameters
    ----------
    capacity_up: int
        Installed capacity of dsm application
        Note: Fixed value; does not account for varying availability
        Corresponds to dsm_up_max in the original terminology
    shift_time_down: int
        Time of a load reduction process
        Corresponds to t_she in the original terminology
    postpone_time: int
        Time between the end of an initial load reduction process
        and the end of the increase process for compensation resp.
        time between the start of an initial load increase process
        and the begin of a reduction process for compensation
        delay_time - shift_time_down
        Corresponds to t_shi in the original terminology
    delay_time: int
        Interval in which between :math:`DSM_{t}^{up}` and
        :math:`DSM_{t}^{down}` have to be compensated.
        Corresponds to t_bal in the original terminology
    cost_dsm_up: float
        costs associated with DSM upwards shifts
    cost_dsm_down_shift: float
        costs associated with DSM downwards shifts
    demand: numeric
        original electrical demand
    addition: boolean
        Boolean parameter indicating, whether or not to use an own additional
        constraint similar to Equation 10 from Zerrahn and Schill (2015)

    """

    def __init__(self, demand, capacity_up, capacity_down, shift_time_down,
                 postpone_time, shed_time,
                 annual_frequency_shift, daily_frequency_shift,
                 annual_frequency_shed,
                 cost_dsm_up=0, cost_dsm_down_shift=0,
                 cost_dsm_down_shed=0, addition=False,
                 shed_eligibility=True, shift_eligibility=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.capacity_up = sequence(capacity_up)
        self.demand = sequence(demand)
        self.capacity_down = sequence(capacity_down)
        self.shift_time_down = shift_time_down
        self.postpone_time = postpone_time
        self.delay_time = shift_time_down + postpone_time
        self.shed_time = shed_time
        self.annual_frequency_shift = annual_frequency_shift
        self.daily_frequency_shift = daily_frequency_shift
        self.annual_frequency_shed = annual_frequency_shed
        self.cost_dsm_up = cost_dsm_up
        self.cost_dsm_down_shift = cost_dsm_down_shift
        self.cost_dsm_down_shed = cost_dsm_down_shed
        self.addition = addition
        self.shed_eligibility = shed_eligibility
        self.shift_eligibility = shift_eligibility

    def constraint_group(self):
        return SinkDSMBlock


class SinkDSMBlock(SimpleBlock):
    r"""Constraints for SinkDSM

        **The following constraints are created:**

        .. _SinkDSM-equations:

        .. math::
            &
            (1) \quad \dot{E}_{t} = demand_{t} + DSM_{t}^{up} - DSM_{t}^{down}
            \quad \forall t \in \mathbb{T}\\
            &
            (2) \quad  DSM_{t}^{down} \leq demand_{t}
            \quad \forall t \in \mathbb{T}\\
            &
            (3) \quad  DSM_{t}^{up} \leq DSM^{up,max} - demand_{t}
            \quad \forall t \in \mathbb{T}\\
            &
            (4) \quad  DSM_{t}^{sl} = DSM_{t-1}^{sl} + DSM_{t}^{up} - DSM_{t}^{down}
            \quad \forall t \in \mathbb{T}\\
            &
            (5) \quad  DSM_{t}^{sl} = 0
            \quad \forall \t \in \mathbb{T} \mid t \mod \tau = 0\} \\
            &
            (6) \quad  \sum_{t}^{t + 23} DSM_{t}^{down}
            \leq \fraq{demand_{t}}{24} \cdot t_{she} \cdot f_{d}
            \quad \forall t \in \mathbb{T}\\
            &
            (7) \quad  DSM_{t}^{down} \leq demand_{t} - DSM_{t-1}^{down}
            \quad \forall t \in \mathbb{T}\\
            &
            (8) \quad  DSM_{t}^{down} + DSM_{t}^{up}
            \leq max \{demand_{t}, DSM^{up,max} - demand_{t} \}
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

        # for all DSM components get inflow from bus_elec
        for n in group:
            n.inflow = list(n.inputs)[0]

        #  ************* SETS **********************************

        # Set of DSM Components
        self.DSM = Set(initialize=[n for n in group])

        #  ************* VARIABLES *****************************

        # Variable load shift down
        self.dsm_do_shift = Var(self.DSM, m.TIMESTEPS, initialize=0,
                                within=NonNegativeReals)

        # Variable load shift down
        self.dsm_do_shed = Var(self.DSM, m.TIMESTEPS, initialize=0,
                               within=NonNegativeReals)

        # Variable load shift up
        self.dsm_up = Var(self.DSM, m.TIMESTEPS, initialize=0,
                         within=NonNegativeReals)

        # DSM storage level -> May take negative values
        self.dsm_sl = Var(self.DSM, m.TIMESTEPS, initialize=0,
                          within=Reals)

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
            The actual demand after DSM.
            Bus outflow == Demand +- DSM (i.e. effective Sink consumption)
            """

            for t in m.TIMESTEPS:
                for g in group:
                    # Sink inflow (i.e. bus outflow)
                    lhs = m.flow[g.inflow, g, t]

                    # Demand + DSM_up - DSM_down
                    rhs = g.demand[t] - self.dsm_do_shift[g, t] \
                          + self.dsm_up[g, t] - self.dsm_do_shed[g, t]

                    # add constraint
                    block.input_output_relation.add((g, t), (lhs == rhs))

        self.input_output_relation = Constraint(group, m.TIMESTEPS,
                                                noruleinit=True)
        self.input_output_relation_build = BuildAction(
            rule=_input_output_relation_rule)

        # Equation 4.23 (down)
        def dsm_availability_down_rule(block):
            """
            Equation 4.23 from Ladwig 2018 (modified):
            Every downwards load shift must be smaller than or equal
            to the current availability (originally demand).
            """

            for t in m.TIMESTEPS:
                for g in group:

                    # DSM down
                    lhs = self.dsm_do_shift[g, t] + self.dsm_do_shed[g, t]

                    # capacity limit
                    rhs = g.capacity_down[t]

                    # add constraint
                    block.dsm_availability_down.add((g, t), (lhs <= rhs))

        self.dsm_availability_down = Constraint(group, m.TIMESTEPS,
                                                noruleinit=True)
        self.dsm_availability_down_build = BuildAction(
            rule=dsm_availability_down_rule)

        # Equation 4.27 (up)
        def dsm_availability_up_rule(block):
            """
            Equation 4.27 from Ladwig 2018 (modified):
            Every upwards load shift must be smaller than or equal
            to the current availability
            (originally maximum capacity installed minus current demand).
            """

            for t in m.TIMESTEPS:
                for g in group:

                    # DSM up
                    lhs = self.dsm_up[g, t]

                    # capacity limit
                    rhs = g.capacity_up[t]

                    # add constraint
                    block.dsm_availability_up.add((g, t), (lhs <= rhs))

        self.dsm_availability_up = Constraint(group, m.TIMESTEPS,
                                              noruleinit=True)
        self.dsm_availability_up_build = BuildAction(
            rule=dsm_availability_up_rule)

        # Equation 4.28
        def dsm_storage_level_rule(block):
            """
            Equation 4.28 from Ladwig 2018:
            Equation for the transition of a fictious DSM storage level.
            """

            for t in m.TIMESTEPS:
                for g in group:

                    if t > 0:
                        
                        # DSM storage level
                        lhs = self.dsm_sl[g, t]

                        # DSM up
                        rhs = self.dsm_sl[g, t-1] + self.dsm_up[g, t] \
                              - self.dsm_do_shift[g, t]

                        # add constraint
                        block.dsm_storage_level.add((g, t), (lhs == rhs))

                    else:
                        lhs = self.dsm_sl[g, t]
                        rhs = self.dsm_up[g, t] - self.dsm_do_shift[g, t]
                        block.dsm_storage_level.add((g, t), (lhs == rhs))

        self.dsm_storage_level = Constraint(group, m.TIMESTEPS,
                                             noruleinit=True)
        self.dsm_storage_level_build = BuildAction(
            rule=dsm_storage_level_rule)

        # Equation 4.29 / 4.30
        def dsm_storage_balanced_rule(block):
            """
            Equation 4.29 from Ladwig 2018:
            DSM storage level is forced to zero at the end of every
            shifting intervall. Equation 4.30 can be integrated here
            since it only determines the shifting intervalls.

            NOTE: This is somewhat similar to the method 'interval'
            from :class:`oemof.solph.custom.SinkDSM`
            """

            for g in group:
                # Original formulation as in the PhD thesis
                # Problem: Limits DSM activations to first balancing intervals
                intervals = range(m.TIMESTEPS[1],
                                  int(g.annual_frequency_shift * g.delay_time),
                                  int(g.delay_time+1))

                # Adjusted, more flexible formulation
                # intervals = range(m.TIMESTEPS[1],
                #                   m.TIMESTEPS[-1] + 1,
                #                   g.delay_time+1)

                for interval in intervals:
                    lhs = self.dsm_sl[g, interval]
                    rhs = 0

                    # add constraint
                    block.dsm_storage_balanced.add((g, interval), (lhs == rhs))

                for t in range(intervals[-1],
                               m.TIMESTEPS[-1] + 1):
                    lhs = self.dsm_sl[g, t]
                    rhs = 0

                    # add constraint
                    block.dsm_storage_balanced.add((g, t), (lhs == rhs))

        self.dsm_storage_balanced = Constraint(group, m.TIMESTEPS,
                                                  noruleinit=True)
        self.dsm_storage_balanced_build = BuildAction(
            rule=dsm_storage_balanced_rule)
        
        # Equation 4.31 / 4.32
        def dsm_day_limit_rule(block):
            """
            Equation 4.31 from Ladwig 2018:
            Limit the amount of energy shifted down within one day
            Equation 4.32 can be integrated here since it only
            determines the daily intervalls.
            """

            for g in group:
                days = range(m.TIMESTEPS[1],
                             m.TIMESTEPS[-1] + 1,
                             24)

                for day in days:
                    if (day + 23) > m.TIMESTEPS[-1]:
                        timesteps = range(day,
                                          m.TIMESTEPS[-1] + 1)
                    else:
                        timesteps = range(day, day + 24)

                    # DSM down
                    lhs = sum(self.dsm_do_shift[g, t] * m.timeincrement[t]
                              for t in timesteps)

                    # maximum energy to be shifted
                    rhs = sum(g.demand[t] for t in timesteps) / 24 \
                          * g.shift_time_down * g.daily_frequency_shift

                    # add constraint
                    block.dsm_day_limit.add((g, day), (lhs <= rhs))

        self.dsm_day_limit = Constraint(group, m.TIMESTEPS,
                                                noruleinit=True)
        self.dsm_day_limit_build = BuildAction(
            rule=dsm_day_limit_rule)

        # TODO: Critically check Eq. 33s legitimation:
        # In my opinion, this does not make much sense because the available
        # potential in the consecutive hour might be much higher and is more
        # or less independent of the prior hour.
        # Furthermore, consequently limiting the activation of one unit would
        # demand to have a look at all successors.
        # Lastly, a DSM portfolio is modelled, not a single unit ...
        # Equation 4.33
        def dsm_down_limit_rule(block):
            """
            Equation 4.33 from Ladwig 2018:
            Limit downwards shift dependent on prior downwards shifts
            """

            for t in m.TIMESTEPS:
                for g in group:

                    if t > 0:
                        # DSM down
                        lhs = self.dsm_do_shift[g, t]

                        # maximum capacity for shifting
                        # rhs = g.demand[t - 1] - self.dsm_do_shift[g, t - 1]
                        rhs = g.capacity_down[t - 1] - self.dsm_do_shift[g, t - 1]

                        # add constraint
                        block.dsm_down_limit.add((g, t), (lhs <= rhs))

                    else:
                        pass  # return(Constraint.Skip)

        self.dsm_down_limit = Constraint(group, m.TIMESTEPS,
                                         noruleinit=True)
        self.dsm_down_limit_build = BuildAction(
            rule=dsm_down_limit_rule)

        # Equation 4.34
        def dsm_shed_limit_year_rule(block):
            """
            Equation 4.34 from Ladwig 2018:
            Limit the amount of energy shedded within one year /
            within the overall optimization timeframe.
            """
            for g in group:

                # DSM down
                lhs = sum(self.dsm_do_shed[g, t] * m.timeincrement[t]
                          for t in m.TIMESTEPS)

                # maximum energy for shedding
                rhs = g.annual_frequency_shed * g.shed_time \
                      * max(g.capacity_down[t] for t in m.TIMESTEPS)

                # add constraint
                block.dsm_shed_limit_year.add((g), (lhs <= rhs))

        self.dsm_shed_limit_year = Constraint(group, noruleinit=True)
        self.dsm_shed_limit_year_build = BuildAction(
            rule=dsm_shed_limit_year_rule)

        # Equation 4.35
        def dsm_shed_limit_day_rule(block):
            """
            Equation 4.35 from Ladwig 2018:
            Limit the amount of energy shedded within one day.
            """

            for g in group:
                days = range(m.TIMESTEPS[1],
                             m.TIMESTEPS[-1] + 1,
                             24)

                for day in days:
                    if (day + 23) > m.TIMESTEPS[-1]:
                        timesteps = range(day,
                                          m.TIMESTEPS[-1] + 1)
                    else:
                        timesteps = range(day, day + 24)

                    # DSM down
                    lhs = sum(self.dsm_do_shed[g, t] * m.timeincrement[t]
                              for t in timesteps)

                    # maximum energy to be shedded
                    rhs = g.shed_time * max(g.capacity_down[t]
                                            for t in timesteps)

                    # add constraint
                    block.dsm_shed_limit_day.add((g, day), (lhs <= rhs))

        self.dsm_shed_limit_day = Constraint(group, m.TIMESTEPS,
                                         noruleinit=True)
        self.dsm_shed_limit_day_build = BuildAction(
            rule=dsm_shed_limit_day_rule)

        # Own addition (optional)
        def dsm_logic_rule(block):
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

                        # maximum upper limit
                        rhs = max(g.demand[t],
                                  g.capacity_up[t] - g.demand[t])

                        # add constraint
                        block.dsm_logic.add((g, t), (lhs <= rhs))

                    else:
                        pass  # return(Constraint.Skip)

        self.dsm_logic = Constraint(group, m.TIMESTEPS,
                                    noruleinit=True)
        self.dsm_logic_build = BuildAction(
            rule=dsm_logic_rule)

    def _objective_expression(self):
        """Adding cost terms for DSM activity to obj. function"""

        m = self.parent_block()

        dsm_cost = 0

        for t in m.TIMESTEPS:
            for g in self.DSM:
                dsm_cost += self.dsm_up[g, t] * g.cost_dsm_up
                dsm_cost += self.dsm_do_shift[g, t] * g.cost_dsm_down_shift \
                            + self.dsm_do_shed[g, t] * g.cost_dsm_down_shed

        self.cost = Expression(expr=dsm_cost)

        return self.cost

class SinkDSMInvestmentBlock(SinkDSMBlock):
    pass