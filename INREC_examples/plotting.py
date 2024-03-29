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


def make_directory(folder_name):
    existing_folders = next(os.walk("."))[1]
    if folder_name in existing_folders:
        print("----------------------------------------------------------")
        print(
            'Folder "' + folder_name + '" already exists in current directory.'
        )
        print("----------------------------------------------------------")
    else:
        path = "./" + folder_name
        os.mkdir(path)
        print("----------------------------------------------------------")
        print('Created folder "' + folder_name + '" in current directory.')
        print("----------------------------------------------------------")


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
    """Extract data from Pyomo Variables in DataFrames and plot for visualization.

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
    # Determine which generation results to extract
    include_coal = kwargs.get("include_coal", True)
    include_gas = kwargs.get("include_gas", False)

    # Introduce shorcuts
    bus_elec_seqs = solph.views.node(model.es.results["main"], "bus_elec")[
        "sequences"
    ]
    dsm_seqs = solph.views.node(model.es.results["main"], "demand_dsm")[
        "sequences"
    ]

    # Generators coal
    if include_coal:
        df_coal_1 = bus_elec_seqs[(("pp_coal_1", "bus_elec"), "flow")].rename(
            "coal1", inplace=True
        )
    else:
        df_coal_1 = pd.Series(index=bus_elec_seqs.index)

    if include_gas:
        df_gas_1 = bus_elec_seqs[(("pp_gas_1", "bus_elec"), "flow")].rename(
            "gas1", inplace=True
        )
    else:
        df_gas_1 = pd.Series(index=bus_elec_seqs.index)

    # Generators RE
    df_wind = bus_elec_seqs[(("wind", "bus_elec"), "flow")].rename(
        "wind", inplace=True
    )

    df_pv = bus_elec_seqs[(("pv", "bus_elec"), "flow")].rename(
        "pv", inplace=True
    )

    # Shortage/Excess
    df_shortage = bus_elec_seqs[(("shortage_el", "bus_elec"), "flow")].rename(
        "shortage", inplace=True
    )

    df_excess = bus_elec_seqs[(("bus_elec", "excess_el"), "flow")].rename(
        "excess", inplace=True
    )

    # ---------------- Extract DSM results (all approaches) ---------------------
    # Parts of results extraction is dependent on kwargs (might be removed later)
    use_no_shed = kwargs.get("use_no_shed", False)

    # Demand after DSM
    df_demand_dsm = bus_elec_seqs[(("bus_elec", "demand_dsm"), "flow")].rename(
        "demand_dsm", inplace=True
    )

    # Downwards shifts (shifting)
    df_dsmdo_shift = (
        dsm_seqs.iloc[:, dsm_seqs.columns.str[1] == "dsm_do_shift"]
        .sum(axis=1)
        .rename("dsm_do_shift", inplace=True)
    )

    # Downwards shifts (shedding)
    if not (approach == "DLR" and use_no_shed):
        df_dsmdo_shed = (
            dsm_seqs.iloc[:, dsm_seqs.columns.str[1] == "dsm_do_shed"]
            .sum(axis=1)
            .rename("dsm_do_shed", inplace=True)
        )
    else:
        df_dsmdo_shed = pd.Series(index=dsm_seqs.index)

    # Upwards shifts
    df_dsmup = (
        dsm_seqs.iloc[:, dsm_seqs.columns.str[1] == "dsm_up"]
        .sum(axis=1)
        .rename("dsm_up", inplace=True)
    )

    # Print the sequences for the demand response unit in order to include
    # proper slicing
    # print(dsm_seqs.columns)

    df_dsm_add = None

    # Get additional DSM results dependent on approach considered
    if approach == "TUD":
        # DSM storage level
        df_dsmsl = (
            dsm_seqs.iloc[:, dsm_seqs.columns.str[1] == "dsm_sl"]
            .sum(axis=1)
            .rename("dsm_sl", inplace=True)
        )

        df_dsm_add = df_dsmsl.copy()

    elif approach == "DLR":
        # Original shift values
        df_dsmdo_orig = df_dsmdo_shift.copy().rename(
            "dsm_do_orig", inplace=True
        )
        df_dsmup_orig = df_dsmup.copy().rename("dsm_up_orig", inplace=True)

        # Balacing values
        df_dsmdo_bal = (
            dsm_seqs.iloc[:, dsm_seqs.columns.str[1] == "balance_dsm_do"]
            .sum(axis=1)
            .rename("balance_dsm_do", inplace=True)
        )
        df_dsmup_bal = (
            dsm_seqs.iloc[:, dsm_seqs.columns.str[1] == "balance_dsm_up"]
            .sum(axis=1)
            .rename("balance_dsm_up", inplace=True)
        )

        # DSM storage levels
        df_dsmsldo = (
            dsm_seqs.iloc[:, dsm_seqs.columns.str[1] == "dsm_do_level"]
            .sum(axis=1)
            .rename("dsm_sl_do", inplace=True)
        )
        df_dsmslup = (
            dsm_seqs.iloc[:, dsm_seqs.columns.str[1] == "dsm_up_level"]
            .sum(axis=1)
            .rename("dsm_sl_up", inplace=True)
        )

        df_dsmdo_shift = df_dsmdo_orig.add(df_dsmup_bal).rename(
            "dsm_do_shift", inplace=True
        )
        df_dsmup = df_dsmup_orig.add(df_dsmdo_bal).rename(
            "dsm_up", inplace=True
        )

        df_dsm_add = pd.concat(
            [
                df_dsmdo_orig,
                df_dsmup_orig,
                df_dsmdo_bal,
                df_dsmup_bal,
                df_dsmsldo,
                df_dsmslup,
            ],
            axis=1,
        )

    # Effective DSM shift (shifting only)
    df_dsm_tot = df_dsmdo_shift - df_dsmup
    df_dsm_tot.rename("dsm_tot", inplace=True)

    # DSM storage level
    df_dsm_acum = df_dsm_tot.cumsum()
    df_dsm_acum.rename("dsm_acum", inplace=True)

    # Original demand before DSM
    df_demand_el = [_ for _ in model.NODES.data() if str(_) == "demand_dsm"][
        0
    ].demand
    df_demand_el.rename("demand_el", inplace=True)

    # Capacity limit for upshift
    df_capup = [_ for _ in model.NODES.data() if str(_) == "demand_dsm"][
        0
    ].capacity_up
    df_capup.rename("cap_up", inplace=True)

    # Capacity limit for downshift
    df_capdo = [_ for _ in model.NODES.data() if str(_) == "demand_dsm"][
        0
    ].capacity_down
    df_capdo.rename("cap_do", inplace=True)

    # ####### Merge alld data into one DataFrame
    df_model = pd.concat(
        [
            df_coal_1,
            df_gas_1,
            df_wind,
            df_pv,
            df_excess,
            df_shortage,
            df_demand_dsm,
            df_dsmdo_shift,
            df_dsmdo_shed,
            df_dsmup,
            df_dsm_tot,
            df_dsm_acum,
            df_demand_el,
            df_capup,
            df_capdo,
        ],
        axis=1,
    )

    # Add additional dsm values for certain approaches
    if df_dsm_add is not None:
        df_model = pd.concat([df_model, df_dsm_add], axis=1, sort=False)

    return df_model


def plot_dsm(df_gesamt, directory, project, days, legend=False, **kwargs):
    """Create a plot of DSM activity"""
    figsize = kwargs.get("figsize", (12, 10))
    save = kwargs.get("save", False)
    approach = kwargs.get("approach", None)
    include_approach = kwargs.get("include_approach", False)
    include_generators = kwargs.get("include_generators", False)
    ax1_ylim = kwargs.get("ax1_ylim", [-10, 210])
    ax2_ylim = kwargs.get("ax2_ylim", [-110, 110])

    use_no_shed = kwargs.get("use_no_shed", False)

    # Create Figure
    for info, slice in df_gesamt.resample(str(days) + "D"):

        slice_reindexed = slice.reset_index(drop=True)
        slice_reindexed["new_index"] = list(range(1, len(slice_reindexed) + 1))
        slice_reindexed.set_index("new_index", drop=True, inplace=True)

        if include_generators:
            graph_wind = slice_reindexed.wind.values
            graph_pv = graph_wind + slice_reindexed.pv.values
            graph_coal = graph_pv + slice_reindexed.coal1.values
            graph_gas = graph_coal + slice_reindexed.gas1.values

        # first axis
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_ylim(ax1_ylim)
        plt.xticks(range(1, len(slice_reindexed) + 1), rotation=90)

        ax1.step(
            slice_reindexed.index,
            slice_reindexed.demand_el.values,
            where="post",
            label="Nachfrage vor Lastmanagement",
            linestyle="--",
            color="blue",
        )
        ax1.step(
            slice_reindexed.index,
            slice_reindexed.demand_dsm.values,
            where="post",
            label="Nachfrage nach Lastmanagement",
            color="black",
        )

        # DSM Capacity
        ax1.step(
            slice_reindexed.index,
            slice_reindexed.demand_el + slice_reindexed.cap_up,
            where="post",
            label="Leistungsgrenzen",
            color="red",
            linestyle="--",
        )
        ax1.step(
            slice_reindexed.index,
            slice_reindexed.demand_el - slice_reindexed.cap_do,
            where="post",
            color="red",
            linestyle="--",
        )

        # Generators
        if include_generators:
            ax1.fill_between(
                slice_reindexed.index,
                0,
                graph_wind,
                step="post",
                label="Wind",
                facecolor="darkcyan",
                alpha=0.5,
            )
            ax1.fill_between(
                slice_reindexed.index,
                graph_wind,
                graph_pv,
                step="post",
                label="PV",
                facecolor="gold",
                alpha=0.5,
            )
            ax1.fill_between(
                slice_reindexed.index,
                graph_pv,
                graph_coal,
                step="post",
                label="Coal",
                facecolor="black",
                alpha=0.5,
            )
            ax1.fill_between(
                slice_reindexed.index,
                graph_coal,
                graph_gas,
                step="post",
                label="Gas",
                facecolor="brown",
                alpha=0.5,
            )

        plt.grid()

        # Second axis
        ax2 = ax1.twinx()
        ax2.set_ylim(ax2_ylim)
        ax2.fill_between(
            slice_reindexed.index,
            0,
            -slice_reindexed.dsm_do_shift,
            step="post",
            label="Lastreduktion (Verschiebung)",
            facecolor="red",
            # hatch='.',
            alpha=0.3,
        )
        if not (approach == "DLR" and use_no_shed):
            ax2.fill_between(
                slice_reindexed.index,
                -slice_reindexed.dsm_do_shift,
                -(slice_reindexed.dsm_do_shift + slice_reindexed.dsm_do_shed),
                step="post",
                label="Lastreduktion (Verzicht)",
                facecolor="blue",
                alpha=0.3,
            )
        ax2.fill_between(
            slice_reindexed.index,
            0,
            slice_reindexed.dsm_up,
            step="post",
            label="Lasterhöhung",
            facecolor="green",
            alpha=0.3,
        )
        ax2.plot(
            slice_reindexed.index,
            slice_reindexed.dsm_acum,
            linestyle="none",
            markersize=8,
            marker="D",
            color="dimgrey",
            fillstyle="none",
            drawstyle="steps-post",
            label="Lastmanagementspeicherlevel",
        )

        # Legend
        if legend:
            handles, labels = [], []
            for ax_object in [ax1, ax2]:
                h, l = ax_object.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)

            _ = plt.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.0, -0.25, 1.0, 0.102),
                fancybox=True,
                shadow=False,
                ncol=3,
                mode="expand",
                borderaxespad=0.0,
            )
        ax1.set_xlabel("Zeit in h", labelpad=10)
        ax1.set_ylabel("Last in MW", labelpad=10)
        ax2.set_ylabel("Laständerung in MW", labelpad=10)

        if approach is not None:
            plt.title(approach)

        _ = ax1.margins(0, 0.05)
        _ = ax2.margins(0, 0.05)
        plt.show()

        if save:
            fig.set_tight_layout(True)
            name = "Plot_" + project + "_" + info._date_repr + ".png"
            if include_approach:
                name = (
                    "Plot_"
                    + project
                    + "_"
                    + approach
                    + "_"
                    + info._date_repr
                    + ".png"
                )
            fig.savefig(directory + "graphics/" + name, bbox_inches="tight")
            plt.close()
            print(name + " saved.")
