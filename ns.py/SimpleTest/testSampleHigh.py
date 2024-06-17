import sys

sys.path.append("/root/Archived-DHT/DHT_revise")

from cmath import log
import logging
import argparse
import pickle
from typing import Dict
from functools import partial
from random import expovariate
import random
import numpy as np
import simpy
from ns.packet.sink import PacketSink
from ns.flow.flow import Flow
from ns.topos.fattree import build as build_fattree
from ns.topos.utils import generate_fib
import networkx as nx
from utils import (
    PathGenerator,
    headlinesPrint,
    merge_dict_with_max,
    merge_switch_to_segment,
)
from pprint import pprint
from ZipfPacketGenerator import ZipfPacketGenerator
from MonitorSwitch import MonitorSwitchDHT
from GlobalController import GlobalController
from MonitorData import MonitorData
import time
import sys
import utils
from collections import defaultdict


# initial the simpy env
env = simpy.Environment()


# build fat tree topolody
ft: nx.Graph = build_fattree(4)
print("Half Fat Tree({}) with {} nodes.".format(4, ft.number_of_nodes()))

# add host of fat tree topo
hosts = set()
for n in ft.nodes():
    if ft.nodes[n]["type"] == "host":
        hosts.add(n)

# generate flows
tic1 = time.perf_counter()
all_flows: Dict[int, Flow] = PathGenerator(ft, hosts).generate_flows(1000)
headlinesPrint("Generate Flows Succeed")
tic2 = time.perf_counter()
print(f"Generate Flows time: {tic2 - tic1}s")
logging.info(f"Generate Flows time: {tic2 - tic1}s")

zipf_flows = utils.generate_zipf_distribution(1.1, 1000, 10000)

size_dist = partial(expovariate, 1.0 / 100)
true_flow_size = dict()

# set all flows with packet gen and packet sink
for fid in all_flows:
    # since the flow size of time T is usually believed to be poisson distribution
    # the interval of two consecutive packets follows exp distribution
    arr_dist = partial(expovariate, 1.5)
    pg = ZipfPacketGenerator(
        env,
        f"Flow_{fid}",
        arr_dist,
        size_dist,
        flow_size=zipf_flows[fid],
        flow_id=fid,
        begin=0,
        end=10000,
    )
    true_flow_size[fid] = pg.flow_size
    ps = PacketSink(env, rec_arrivals=False, rec_flow_ids=False, rec_waits=False)
    all_flows[fid].pkt_gen = pg
    all_flows[fid].size = pg.flow_size
    all_flows[fid].pkt_sink = ps

sampled_high_flows_set = utils.initial_sample_priority(all_flows, 10)
unsampled_set = set(all_flows.keys()) - sampled_high_flows_set

cnt = 0
for fid in all_flows:
    flow = all_flows[fid]
    if flow.pkt_gen.isHigh:
        cnt += 1
        print(flow.fid, end=", ")
print()
print(cnt)

for i in range(10):
    sampled_high_flows_set, unsampled_set = utils.update_sample_priority(
        all_flows, sampled_high_flows_set, unsampled_set, 5, 10 + i
    )
    cnt = 0
    for flow in all_flows.values():
        if flow.pkt_gen.isHigh:
            cnt += 1
            print(flow.fid, end=", ")
    print()
    print(cnt)
