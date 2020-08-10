#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import oemof.solph as solph
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

import os
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

# register matplotlib converters which have been overwritten by pandas
register_matplotlib_converters()


#################################################################

def make_directory(folder_name):

    existing_folders = next(os.walk('.'))[1]
    if folder_name in existing_folders:
        print('----------------------------------------------------------')
        print('Folder "' + folder_name + '" already exists in current directory.')
        print('----------------------------------------------------------')
    else:
        path = "./" + folder_name
        os.mkdir(path)
        print('----------------------------------------------------------')
        print('Created folder "' + folder_name + '" in current directory.')
        print('----------------------------------------------------------')

def adjust_yaxis(ax, ydif, v):
    """shift axis ax by ydiff, maintaining point v at the same location"""
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny > maxy or (-miny == maxy and dy > 0):
        nminy = miny
        nmaxy = miny * (maxy + dy) / (miny + dy)
    else:
        nmaxy = maxy
        nminy = maxy * (miny + dy) / (maxy + dy)
    ax.set_ylim(nminy + v, nmaxy + v)


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2, (y1 - y2) / 2, v2)
    adjust_yaxis(ax1, (y2 - y1) / 2, v1)


def extract_results(model, approach, **kwargs):
    """ Extract data fro Pyomo Variables in DataFrames and plot for visualization.

    Extract the results from the toy model.
    A distinction for the different approaches has to be made since
    the demand response variables and the way they are handled vary.

    :param model: oemof.solph.models.Model
        The solved optimization model (including results)
    :param approach: str
        Must be one of ["DIW", "IER", "DLR", "TUD"]
    :return: df_model: pd.DataFrame
        A pd.DataFrame containing the concatenated and renamed results sequences
    """

    # ########################### Get DataFrame out of Pyomo and rename series

    # TODO: Completely revise results_extraction (see mail from Julian Endres)

    # Generators coal
    df_coal_1 = solph.views.node(model.es.results['main'], 'bus_elec')['sequences'][
        (('pp_coal_1', 'bus_elec'), 'flow')]
    df_coal_1.rename('coal1', inplace=True)

    # Generators RE
    df_wind = solph.views.node(model.es.results['main'], 'bus_elec')['sequences'][
        (('wind', 'bus_elec'), 'flow')]
    df_wind.rename('wind', inplace=True)

    df_pv = solph.views.node(model.es.results['main'], 'bus_elec')['sequences'][
        (('pv', 'bus_elec'), 'flow')]
    df_pv.rename('pv', inplace=True)

    # Shortage/Excess
    df_shortage = solph.views.node(model.es.results['main'], 'bus_elec')['sequences'][
        (('shortage_el', 'bus_elec'), 'flow')]
    df_shortage.rename('shortage', inplace=True)

    df_excess = solph.views.node(model.es.results['main'], 'bus_elec')['sequences'][
        (('bus_elec', 'excess_el'), 'flow')]
    df_excess.rename('excess', inplace=True)

    # DSM Demand
    df_demand_dsm = solph.views.node(model.es.results['main'], 'bus_elec')['sequences'][
        (('bus_elec', 'demand_dsm'), 'flow')]
    df_demand_dsm.rename('demand_dsm', inplace=True)

    # Print the sequences for the demand response unit in order to include
    # proper slicing
    # print(solph.views.node(model.es.results['main'], 'demand_dsm')[
    #            'sequences'].columns)

    df_dsm_add = None

    # DSM Variables - Get additional values dependent on approach chosen
    if approach == "DIW":
        df_dsmdo_shift = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, 1:-2].sum(axis=1)
        df_dsmdo_shed = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, -2]
        df_dsmup = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, -1]

    elif (approach == "IER") or (approach == "TUD"):
        df_dsmdo_shift = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, 2]
        df_dsmdo_shed = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, 1]
        df_dsmup = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, -1]

        if approach == "TUD":
            df_dsmsl = solph.views.node(model.es.results['main'], 'demand_dsm')[
                             'sequences'].iloc[:, -2]
            df_dsmsl.rename('dsm_sl', inplace=True)

            df_dsm_add = df_dsmsl.copy()

    elif approach == "DLR":
        use_no_shed = kwargs.get('use_no_shed', False)
        # use_shifting_classes = kwargs.get('use_no_shifting_classes', False)

        # Introduce a dict to map the idx location to the respective approach (w/wo shedding)
        dict_map_idx = {
            'with_shed':{'dsmdo_shift': [4, 6],
                         'dsmdo_shed': 5,
                         'dsmup': [3, 7],
                         'dsmdo_orig': 6,
                         'dsmup_orig': 7,
                         'dsmdo_bal': 3,
                         'dsmup_bal': 5,
                         'dsmslup': 2,
                         'dsmsldo': 1},
        }

        vals_wo_shed = [[2, 4], [], [1, 5], 4, 5, 1, 2, 6, 3]

        dict_map_idx['wo_shed'] = {k: v for k, v
                                   in zip(dict_map_idx["with_shed"].keys(),
                                          vals_wo_shed)}

        if not use_no_shed:
            key = 'with_shed'
        else:
            key = 'wo_shed'

        df_dsmdo_shift = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, dict_map_idx[key]['dsmdo_shift']].sum(axis=1)
        df_dsmdo_shed = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, dict_map_idx[key]['dsmdo_shed']]
        df_dsmup = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, dict_map_idx[key]['dsmup']].sum(axis=1)

        # Get individual dsm values as well as dsm storage levels, too
        df_dsmdo_orig = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, dict_map_idx[key]['dsmdo_orig']]
        df_dsmdo_orig.rename('dsm_do_orig', inplace=True)

        df_dsmup_orig = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, dict_map_idx[key]['dsmup_orig']]
        df_dsmup_orig.rename('dsm_up_orig', inplace=True)

        df_dsmdo_bal = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, dict_map_idx[key]['dsmdo_bal']]
        df_dsmdo_bal.rename('balance_dsm_do_orig', inplace=True)

        df_dsmup_bal = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, dict_map_idx[key]['dsmup_bal']]
        df_dsmup_bal.rename('balance_dsm_up_orig', inplace=True)

        df_dsmslup = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, dict_map_idx[key]['dsmslup']]
        df_dsmslup.rename('dsm_sl_up', inplace = True)

        df_dsmsldo = solph.views.node(model.es.results['main'], 'demand_dsm')[
                       'sequences'].iloc[:, dict_map_idx[key]['dsmsldo']]
        df_dsmsldo.rename('dsm_sl_do', inplace=True)

        df_dsm_add = pd.concat([df_dsmdo_orig, df_dsmup_orig,
                                df_dsmdo_bal, df_dsmup_bal,
                                df_dsmslup, df_dsmsldo], axis=1)

    else:
        raise ValueError("No valid value for approach. Must be one of ['DIW', 'IER', 'DLR', 'TUD']")

    df_dsmdo_shift.rename('dsm_do_shift', inplace=True)
    if not (approach == "DLR" and use_no_shed):
        df_dsmdo_shed.rename('dsm_do_shed', inplace=True)
    df_dsmup.rename('dsm_up', inplace=True)

    df_dsm_tot = df_dsmdo_shift - df_dsmup
    df_dsm_tot.rename('dsm_tot', inplace=True)

    df_dsm_acum = df_dsm_tot.cumsum()
    df_dsm_acum.rename('dsm_acum', inplace=True)


    # Demand
    df_demand_el = [_ for _ in model.NODES.data() if str(_) == 'demand_dsm'][0].demand
    df_demand_el.rename('demand_el', inplace=True)

    # Demand
    df_capup = [_ for _ in model.NODES.data() if str(_) == 'demand_dsm'][0].capacity_up
    df_capup.rename('cap_up', inplace=True)

    # Demand
    df_capdo = [_ for _ in model.NODES.data() if str(_) == 'demand_dsm'][0].capacity_down
    df_capdo.rename('cap_do', inplace=True)

    # ####### Merge into one DataFrame
    df_model = pd.concat([df_coal_1,  df_wind, df_pv, df_excess, df_shortage,
                          df_demand_dsm, df_dsmdo_shift, df_dsmdo_shed, df_dsmup,
                          df_dsm_tot, df_dsm_acum, df_demand_el, df_capup, df_capdo],
                          axis=1)

    # Add additional dsm values for certain approaches
    if df_dsm_add is not None:
        df_model = pd.concat([df_model, df_dsm_add], axis=1, sort=False)

    return df_model


def plot_dsm(df_gesamt, directory, project, days, **kwargs):

    figsize = kwargs.get('figsize', (15, 10))
    save = kwargs.get('save', False)
    approach = kwargs.get('approach', None)
    include_approach = kwargs.get('include_approach', False)
    ax1_ylim = kwargs.get('ax1_ylim', [-10, 250])
    ax2_ylim = kwargs.get('ax2_ylim', [-110, 150])

    use_no_shed = kwargs.get('use_no_shed', False)

    # ############ DATA PREPARATION FOR FIGURE #############################

    # Create Figure
    for info, slice in df_gesamt.resample(str(days)+'D'):
        
        # Generators from model
        # hierarchy for plot: wind, pv, coal, shortage
        graph_wind = slice.wind.values
        graph_pv = graph_wind + slice.pv.values
        graph_coal = graph_pv + slice.coal1.values
        graph_shortage = graph_coal + slice.shortage.values

        #################
        # first axis
        #get_ipython().run_line_magic('matplotlib', 'notebook')
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_ylim(ax1_ylim)

        # x-Axis date format
        ax1.xaxis_date()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m - %H h'))  # ('%d.%m-%H h'))
        ax1.set_xlim(info - pd.Timedelta(1, 'h'), info + pd.Timedelta(days*24+1, 'h'))
        plt.xticks(pd.date_range(start=info._date_repr, periods=days*24, freq='H'), rotation=90)

        # Demands
        # ax1.plot(range(timesteps), dsm, label='demand_DSM', color='black')
        ax1.step(slice.index, slice.demand_el.values, where='post', label='Demand', linestyle='--', color='blue')
        ax1.step(slice.index, slice.demand_dsm.values, where='post', label='Demand after DSM', color='black')

        # DSM Capacity
        ax1.step(slice.index, slice.demand_el + slice.cap_up, where='post', label='DSM Capacity', color='red', linestyle='--')
        ax1.step(slice.index, slice.demand_el - slice.cap_do, where='post', color='red', linestyle='--')

        # Generators
        # ax1.fill_between(slice.index, 0, graph_wind, step='post', label='Wind', facecolor='darkcyan', alpha=0.5)
        # ax1.fill_between(slice.index, graph_wind, graph_pv, step='post', label='PV', facecolor='gold', alpha=0.5)
        # ax1.fill_between(slice.index, graph_pv, graph_coal, step='post', label='Coal', facecolor='black', alpha=0.5)
        # ax1.fill_between(slice.index, slice.demand_dsm.values, graph_coal,
        #                  step='post',
        #                  label='Excess',
        #                  facecolor='firebrick',
        #                  hatch='/',
        #                  alpha=0.5)

        ax1.legend(bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

        # plt.xticks(range(0,timesteps,5))

        plt.grid()


        ###########################
        # Second axis
        ax2 = ax1.twinx()
        ax2.xaxis_date()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m - %H h'))  # ('%d.%m-%H h'))
        ax2.set_xlim(info - pd.Timedelta(1, 'h'), info + pd.Timedelta(days*24+1, 'h'))
        plt.xticks(pd.date_range(start=info._date_repr, periods=days*24, freq='H'), rotation=90)

        ax2.set_ylim(ax2_ylim)
        #align_yaxis(ax1, 100, ax2, 0)


        # DSM up/down

        #ax2.step(slice.index, slice.dsm_acum, where='post',
        #         label='DSM acum', alpha=0.5, color='orange')

        ax2.fill_between(slice.index, 0, -slice.dsm_do_shift,
                         step='post',
                         label='DSM_down_shift',
                         facecolor='red',
                         #hatch='.',
                         alpha=0.3)
        if not (approach == "DLR" and use_no_shed):
            ax2.fill_between(slice.index, -slice.dsm_do_shift,
                             -(slice.dsm_do_shift+slice.dsm_do_shed),
                             step='post',
                             label='DSM_down_shed',
                             facecolor='blue',
                             #hatch='.',
                             alpha=0.3)
        ax2.fill_between(slice.index, 0, slice.dsm_up,
                         step='post',
                         label='DSM_up',
                         facecolor='green',
                         #hatch='.',
                         alpha=0.3)
        ax2.fill_between(slice.index, 0, slice.dsm_acum,
                         step='post',
                         label='DSM acum',
                         facecolor=None,
                         hatch='x',
                         alpha=0.0)

        # Legend axis 2
        ax2.legend(bbox_to_anchor=(0., -0.3, 1., 0.102), loc=3, ncol=3, borderaxespad=0., mode="expand")
        ax1.set_xlabel('Time t in h')
        ax1.set_ylabel('MW')
        ax2.set_ylabel('MW')

        if approach is not None:
            plt.title(approach)

        plt.show()

        if save:
            fig.set_tight_layout(True)
            name = 'Plot_' + project + '_' + info._date_repr + '.png'
            if include_approach:
                name = 'Plot_' + project + '_' + approach + '_' + info._date_repr + '.png'
            fig.savefig(directory + 'graphics/' + name)
            plt.close()
            print(name + ' saved.')

