from random import random
from unittest.util import _MAX_LENGTH
import numpy as np
import random

from random import sample
import networkx as nx

from ns.flow.flow import Flow
from pkg_resources import yield_lines
import matplotlib.pyplot as plt
import simpy
from collections import defaultdict
from typing import Dict

# deprecated
# def generate_zipf(a, begin, end):
#     flow_size = np.random.zipf(a)
#     if flow_size > 100:
#         r1 = random.randint(0, begin)
#     elif flow_size > 10:
#         r1 = random.randint(0, begin*4)
#     else:
#         r1 = random.randint(0, end - 1)
#     return r1, flow_size

import random
import math


# call this function after connecting pg (and ps)
def initial_sample_priority(all_flows: Dict[int, Flow], sample_num):
    nFlows = len(all_flows)
    highFlows = sample(range(nFlows), sample_num)
    for h_flow in highFlows:
        all_flows[h_flow].pkt_gen.isHigh = True
    return set(highFlows)


def update_sample_priority(
    all_flows: Dict[int, Flow],
    sampled_set,
    unsampled_set,
    num_to_replace,
    new_sample_num,
):
    num_to_replace = max(num_to_replace, len(sampled_set) - new_sample_num)

    removed = set(sample(sampled_set, num_to_replace))
    for l_flow in removed:
        all_flows[l_flow].pkt_gen.isHigh = False
    sampled_set -= removed

    added = set(sample(unsampled_set, new_sample_num - len(sampled_set)))
    for h_flow in added:
        all_flows[h_flow].pkt_gen.isHigh = True

    unsampled_set -= added
    sampled_set |= added
    return sampled_set, unsampled_set


def generate_zipf_distribution(s, nflows, npackets):
    ranks = list(range(1, nflows + 1))

    z = [1 / (rank**s) for rank in ranks]

    total = sum(z)
    probabilities = [i / total for i in z]

    frequencies = [math.ceil(p * npackets) for p in probabilities]
    return frequencies


def headlinesPrint(str):
    print("*" * 60)
    print("\t\t", str)
    print("*" * 60)


# switch_segment[device_id][segment_idx] = {flow_id:[flow_num, ts, max_interval], }
# ret[segment_idx] = {flow_id:[flow_num, ts, max_interval], }
def merge_switch_to_segment(switch_segment, hasmax=True):
    device_true_segment_result = []
    # device_true_segment_result_len = []
    if hasmax:
        for i in range(len(switch_segment[0])):
            temp = {}
            for j in range(len(switch_segment)):
                if len(switch_segment[j][i]) == 0:
                    continue
                temp = merge_dict_with_max(temp, switch_segment[j][i])
            device_true_segment_result.append(temp)
            # device_true_segment_result_len.append(len(temp))
    else:
        for i in range(len(switch_segment[0])):
            temp = {}
            for j in range(len(switch_segment)):
                if len(switch_segment[j][i]) == 0:
                    continue
                temp = merge_dict(temp, switch_segment[j][i])
            device_true_segment_result.append(temp)
            # device_true_segment_result_len.append(len(temp))
    return device_true_segment_result


def max_dict_update(table, x, freq, cur_ts, int_result):
    assert x in table
    item = table[x]
    item[0] += freq
    if item[0] > 1:
        item[2] = max(item[2], int_result)
    item[1] = cur_ts
    return table


def merge_dict(dict1, dict2):
    for k, v in dict2.items():
        if k in dict1.keys():
            dict1[k] += v
        else:
            dict1[k] = v
    return dict1


def merge_dict_with_max(dict_1, dict_2):
    for k, v in dict_2.items():
        if k in dict_1.keys():
            dict_1[k][0] += v[0]
            if dict_1[k][0] > 1:
                dict_1[k][2] = max(dict_1[k][2], dict_2[k][2])
            dict_1[k][1] = max(dict_1[k][1], dict_2[k][1])
        else:
            dict_1[k] = [v[0], v[1], v[2]]
    return dict_1


def merge_flowtables(dict1: dict, dict2: dict):
    res = defaultdict(list)
    for k, v in dict1.items():
        if k not in res:
            res[k] = v
        else:
            res[k][0] = max(v[0], res[k][0])
            if res[k][0] > 1:
                res[k][2] = max(res[k][2], v[2])
            res[k][1] = max(res[k][1], v[1])

    for k, v in dict2.items():
        if k not in res:
            res[k] = v
        else:
            res[k][0] = max(v[0], res[k][0])
            if res[k][0] > 1:
                res[k][2] = max(res[k][2], v[2])
            res[k][1] = max(res[k][1], v[1])

    return res


def my_generate_flows(G, hosts, nflows):
    all_flows = dict()
    for flow_id in range(nflows):
        src = 0
        dst = 0
        while True:
            src, dst = sample(hosts, 2)
            if G.nodes[src]["pod"] != G.nodes[dst]["pod"]:
                break
        all_flows[flow_id] = Flow(flow_id, src, dst)
        all_simple_path = list(nx.all_shortest_paths(G, src, dst))

        all_flows[flow_id].path = sample(all_simple_path, 1)[0]
    return all_flows


class PathGenerator:
    def __init__(self, G, hosts) -> None:
        self.topo = G
        self.hosts = hosts
        self.pathDict = {}
        for src in hosts:
            for dst in hosts:
                if G.nodes[src]["pod"] != G.nodes[dst]["pod"]:
                    self.pathDict[(src, dst)] = list(
                        nx.all_shortest_paths(self.topo, src, dst)
                    )

    def generate_flows(self, nflows):
        all_flows = dict()
        for flow_id in range(nflows):
            src = 0
            dst = 0
            while True:
                src, dst = sample(self.hosts, 2)
                if self.topo.nodes[src]["pod"] != self.topo.nodes[dst]["pod"]:
                    break
            all_flows[flow_id] = Flow(flow_id, src, dst)
            all_flows[flow_id].path = sample(self.pathDict[(src, dst)], 1)[0]
        return all_flows


def printSampleFlow(ft, all_flows):
    for flow_id in sample(list(all_flows.keys()), 1):
        path = all_flows[flow_id].path
        print(path)
        for hop in path:
            switch_name = ft.nodes[hop]["device"].element_id
            res = ft.nodes[hop]["device"].query(flow_id)
            print(f"Query at switch {switch_name}: result = {res}")

        print(f"Flow {flow_id}")
        print("Packets Wait")
        print(all_flows[flow_id].pkt_sink.waits)
        print("Packet Arrivals")
        print(all_flows[flow_id].pkt_sink.arrivals)
        print("Arrival Perhop Times")
        print(all_flows[flow_id].pkt_sink.perhop_times)
        print(all_flows[flow_id].pkt_sink.packet_times)


def queryTopK(ft):
    global_topK = {}
    for node_id in ft.nodes():
        node = ft.nodes[node_id]
        if node["type"] == "switch":
            local_topK = node["device"].SKETCH.flow_table
            global_topK.update(local_topK)

    _ = sorted(global_topK.items(), key=lambda x: x[1], reverse=True)
    return _


def plot_fattree(ft):
    pos = nx.multipartite_layout(ft, subset_key="layer")
    nx.draw(ft, pos, with_labels=ft.nodes)
    plt.show()


def build_half_fattree_topo(k):
    """
    Return a half fat tree datacenter topology
    Core switches are full mesh connected with aggregation switches
    """
    # validate input arguments
    if not isinstance(k, int):
        raise TypeError("k argument must be of int type")
    if k < 1 or k % 2 == 1:
        raise ValueError("k must be a positive even integer")

    topo = nx.Graph()
    topo.name = "half_fat_tree_topology(%d)" % (k)

    # Create core nodes
    n_core = (k // 2) ** 2 // 2
    topo.add_nodes_from([v for v in range(int(n_core))], layer="core", type="switch")

    # Create aggregation and edge nodes and connect them
    # half tree has only a half number of pods
    for pod in range(k // 2):
        aggr_start_node = topo.number_of_nodes()
        aggr_end_node = aggr_start_node + k // 2
        edge_start_node = aggr_end_node
        edge_end_node = edge_start_node + k // 2
        aggr_nodes = range(aggr_start_node, aggr_end_node)
        edge_nodes = range(edge_start_node, edge_end_node)
        topo.add_nodes_from(aggr_nodes, layer="aggregation", type="switch", pod=pod)
        topo.add_nodes_from(edge_nodes, layer="edge", type="switch", pod=pod)
        topo.add_edges_from(
            [(u, v) for u in aggr_nodes for v in edge_nodes], type="aggregation_edge"
        )

    # Connect core switches to aggregation switches
    for core_node in range(n_core):
        for aggre_node in [
            v for v in topo.nodes() if topo.nodes[v]["layer"] == "aggregation"
        ]:
            topo.add_edge(core_node, aggre_node, type="core_aggregation")

    # Create hosts and connect them to edge switches
    for u in [v for v in topo.nodes() if topo.nodes[v]["layer"] == "edge"]:
        leaf_nodes = range(topo.number_of_nodes(), topo.number_of_nodes() + k // 2)
        topo.add_nodes_from(
            leaf_nodes, layer="leaf", type="host", pod=topo.nodes[u]["pod"]
        )
        topo.add_edges_from([(u, v) for v in leaf_nodes], type="edge_leaf")

    return topo
