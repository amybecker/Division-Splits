import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
random.seed(a)
import csv
import os
from functools import partial
import json
import random
import numpy as np

import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.proposals import recom
from gerrychain.updaters import cut_edges, Tally
from gerrychain.tree import PopulatedGraph, contract_leaves_until_balanced_or_none, recursive_tree_part, predecessors, bipartition_tree
from networkx.algorithms import tree
from collections import deque, namedtuple


first_check_counties = True

def draw_graph(G, plan_assignment, unit_df, division_df, fig_name, geo_id ='GEOID10'):
    cdict = {G.nodes[i][geo_id]:plan_assignment[i] for i in plan_assignment.keys()}
    unit_df['color'] = unit_df.apply(lambda x: cdict[x[geo_id]], axis=1)
    fig,ax = plt.subplots()
    division_df.geometry.boundary.plot(color=None,edgecolor='k',linewidth = 2,ax=ax)
    unit_df.plot(column='color',ax = ax, cmap = 'tab20')
    ax.set_axis_off()
    plt.savefig(fig_name)
    plt.close()


def num_splits(partition, unit_df, geo_id ='GEOID10', division_col = "COUNTYFP10"):
    unit_df["current"] = unit_df[geo_id].map(dict(partition.assignment))
    splits = sum(unit_df.groupby(division_col)["current"].nunique() > 1)
    return splits


def cut_length(partition):
    return len(partition["cut_edges"])

def division_random_spanning_tree(graph, division_col="COUNTYFP10", low_weight = 1, high_weight = 10):
    for edge in graph.edges:
        if graph.nodes[edge[0]][division_col] == graph.nodes[edge[1]][division_col]:
            graph.edges[edge]["weight"] = low_weight + random.random()
        else:
            graph.edges[edge]["weight"] = high_weight + random.random()
    spanning_tree = tree.minimum_spanning_tree(
        graph, algorithm="kruskal", weight="weight"
    )
    return spanning_tree

def split_tree_at_division(h, choice=random.choice, division_col="COUNTYFP10"):
    root = choice([x for x in h if h.degree(x) > 1])
    # BFS predecessors for iteratively contracting leaves
    pred = predecessors(h.graph, root)

    leaves = deque(x for x in h if h.degree(x) == 1)
    while len(leaves) > 0:
        leaf = leaves.popleft()
        parent = pred[leaf]
        if h.graph.nodes[parent][division_col] != h.graph.nodes[leaf][division_col] and h.has_ideal_population(leaf):
            return h.subsets[leaf]
        # Contract the leaf:
        h.contract_node(leaf, parent)
        if h.degree(parent) == 1 and parent != root:
            leaves.append(parent)
    return None


def division_bipartition_tree(
    graph,
    pop_col,
    pop_target,
    epsilon,
    division_col="COUNTYFP10",
    node_repeats=1,
    spanning_tree=None,
    choice=random.choice, 
    attempts_before_giveup = 100):

    populations = {node: graph.nodes[node][pop_col] for node in graph}

    balanced_subtree = None
    if spanning_tree is None:
        spanning_tree = division_random_spanning_tree(graph, division_col=division_col)
    restarts = 0
    counter = 0
    while balanced_subtree is None and counter < attempts_before_giveup:
        # print(counter)
        if restarts == node_repeats:
            spanning_tree = division_random_spanning_tree(graph, division_col=division_col)
            restarts = 0
            counter +=1
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        if first_check_counties and restarts == 0:
            balanced_subtree = split_tree_at_division(h, choice=choice, division_col=division_col)
        if balanced_subtree is None:
            h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
            balanced_subtree = contract_leaves_until_balanced_or_none(h, choice=choice)
        restarts += 1

    if counter >= attempts_before_giveup:
        return set()
    return balanced_subtree



def demo():
    graph_path = "./Data/PA_VTDALL.json"
    graph = Graph.from_json(graph_path)
    k = 18
    ep = 0.05
    pop_col = "TOT_POP"

    plot_path = "./Data/VTD_FINAL"
    unit_df = gpd.read_file(plot_path)
    unit_col = "GEOID10"
    division_col = "COUNTYFP10"
    divisions = unit_df[[division_col,'geometry']].dissolve(by=division_col, aggfunc='sum')
    county_dict = pd.Series(unit_df[division_col].values,index=unit_df[unit_col]).to_dict()

    for v in graph.nodes:
        graph.nodes[v]['division'] = county_dict[v]
        graph.nodes[v][unit_col] = v

    updaters = {
    "population": Tally("TOT_POP", alias="population"),
    "cut_edges": cut_edges,
    }
    cddict =  recursive_tree_part(graph,range(k),unit_df[pop_col].sum()/k,pop_col, .01, node_repeats=1)
    initial_partition = Partition(graph, cddict, updaters)
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
    division_proposal = partial(recom, pop_col=pop_col, pop_target=ideal_population, epsilon=0.05,  method=partial(division_bipartition_tree, division_col = 'division'), node_repeats=2)

    chain = MarkovChain(
    proposal=division_proposal,
    constraints=[constraints.within_percent_of_ideal_population(initial_partition, 0.05),],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=1000)

    t=0
    snapshot = 100
    for part in chain:
        if t%snapshot == 0:
            draw_graph(graph, part.assignment, unit_df, divisions, './figs/chain_'+str(t)+'.png',geo_id =unit_col)
            print("t: ", t, ", num_splits: ",num_splits(part, unit_df, geo_id =unit_col, division_col = division_col), ", cut_length",cut_length(part))
        t += 1


if __name__ == '__main__':
    demo()