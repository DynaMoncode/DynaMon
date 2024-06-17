import logging
import argparse
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


def queryPathSketch(ft, flow_id: int) -> None:
    headlinesPrint(f"Query Path of Flow {flow_id}")
    print("Querying all sketches in the path of this flow")
    path = all_flows[flow_id].path
    pprint(path)
    for hop in path:
        if ft.nodes[hop]["type"] != "switch":
            continue
        switch_name = ft.nodes[hop]["device"].element_id
        res = ft.nodes[hop]["device"].query(flow_id)
        # ft.nodes[hop]['device'].print()
        print(f"Query at switch {switch_name}: result = {res}")


parser = argparse.ArgumentParser()

#! memory of switch in KB
parser.add_argument("--memory", type=float, nargs="+", default=[1, 2])

# sumax sketch memory, need to translate from --memory to --sketch_memory
parser.add_argument("--sketch_memory", type=list, default=[])
parser.add_argument("--d", type=int, default=3)  # sketch

# is_dht = false by default
#   true: dht ring
#   false: normal hash
parser.add_argument("--dht", action="store_true", dest="is_dht")
# the parameter seems useless
parser.add_argument("--do_dynamic", action="store_true", dest="do_dynamic", default="True")

# flow num
parser.add_argument("--n_flows", type=int, default=600_000)
parser.add_argument("--n_packets", type=int, default=10_000_000)
# parser.add_argument("--n_flows", type=int, default=1_000)
# parser.add_argument("--n_packets", type=int, default=10_000)

# Fattree k
parser.add_argument("--k", type=int, default=4)

# do_limit_memory = False by default
parser.add_argument("--limit_memory", action="store_true", dest="do_limit_memory")

# number of limited switch with limited memory
parser.add_argument("--memory_limit", type=list, default=[])
parser.add_argument("--sketch_memory_limit", type=list, default=[])

# memory of limited switch
parser.add_argument("--memory_limit_ratio", type=float, default=0.1)


# heavy_hitter_thres
parser.add_argument("--heavy_hitter_thres", type=int, default=1000)
# heavy_change_ratio
parser.add_argument("--heavy_change_ratio", type=int, default=0.0001)
# heavy_change_thres = ratio * flow number
parser.add_argument("--heavy_change_thres", type=int)


parser.add_argument("--pir", type=int, default=10000)  # pir
parser.add_argument("--buffer_size", type=int, default=10000000)  # buffer_size
parser.add_argument("--log_file", type=str, default="default_log.log")

# the switch number of Fattree (do not set)
parser.add_argument("--switch_num", type=int)

# parameters of flow
parser.add_argument("--flow_alpha", type=float, default=1.1)
parser.add_argument("--arr_dist_alpha", type=float, default=1.4)

parser.add_argument("--random_seed", type=int, default=45721)
parser.add_argument("--inter", type=int, default=10000)  # the interval of flip
parser.add_argument("--finish", type=int, default=100000)
parser.add_argument("--begin", type=int, default=10000)
parser.add_argument("--end", type=int, default=100000)
parser.add_argument("--mean_pkt_size", type=float, default=100.0)


args = parser.parse_args()


def args_initialization():
    ### process the arguments
    args.switch_num = int(5 * args.k**2 / 4)

    # change the memory from KB to Byte
    for i in range(len(args.memory)):
        args.memory[i] *= 1024

    for i in range(len(args.memory)):
        # sketch memory = memory since there exists only the PrioritySketch in switches
        args.sketch_memory.append(int(args.memory[i]))

        # memory limit
        args.sketch_memory_limit.append(int(args.sketch_memory[i] * float(args.memory_limit_ratio)))

    if args.do_limit_memory:
        for i in range(len(args.memory)):
            args.memory_limit.append(int(args.memory[i] * float(args.memory_limit_ratio)))

    # set this according to the number of total packets
    args.heavy_change_thres = int(args.n_packets * args.heavy_change_ratio)

    logging.info(args)
    logging.info(args.sketch_memory)


if __name__ == "__main__":
    # set the random seed
    random.seed(args.random_seed)
    np.set_printoptions(linewidth=400)
    np.random.seed(args.random_seed)

    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG, filename=args.log_file, filemode="a"
    )
    handler = logging.StreamHandler()
    handler.terminator = ""
    logging.info("\n")
    logging.info(time.asctime())
    logging.info(sys.argv)
    logging.info(
        f"n_flows: {args.n_flows}, memory:{args.memory}, is_dht:{args.is_dht}, heavy_hitter_thres:{args.heavy_hitter_thres}, heavy_change_ratio:{args.heavy_change_ratio}"
    )

    headlinesPrint("Process arguments begin")
    args_initialization()
    headlinesPrint("Process arguments end")

    # initial the simpy env
    env = simpy.Environment()

    # build a half fat tree topology
    ft: nx.Graph = utils.build_half_fattree_topo(args.k)
    print("Half Fat Tree({}) with {} nodes.".format(args.k, ft.number_of_nodes()))

    # # build fat tree topolody
    # ft: nx.Graph = build_fattree(args.k)
    # print("Fat Tree({}) with {} nodes.".format(args.k, ft.number_of_nodes()))

    # add host of fat tree topo
    hosts = set()
    for n in ft.nodes():
        if ft.nodes[n]["type"] == "host":
            hosts.add(n)

    # limit memory of some switches
    if args.do_limit_memory:
        # memory_limit_switches_1 = list(range(int(args.k**2 / 8)))
        # memory_limit_switches_2 = list(
        #     range(int(args.k**2 / 4), int(args.k**2 / 4) + int(args.k / 2))
        # )
        # memory_limit_switches = memory_limit_switches_1 + memory_limit_switches_2
        memory_limit_switches = [0, 2, 3]
    else:
        memory_limit_switches = []
    print("Memory limit switch", memory_limit_switches)
    logging.info(f"memory_limit_switch: {memory_limit_switches}")

    # generate flows
    tic1 = time.perf_counter()
    all_flows: Dict[int, Flow] = PathGenerator(ft, hosts).generate_flows(args.n_flows)
    headlinesPrint("Generate Flows Succeed")
    tic2 = time.perf_counter()
    print(f"Generate Flows time: {tic2 - tic1}s")
    logging.info(f"Generate Flows time: {tic2 - tic1}s")

    zipf_flows = utils.generate_zipf_distribution(args.flow_alpha, args.n_flows, args.n_packets)
    # packet size distribution: expovariate distribution
    # with lambda = 1.0 / mean_pkt_size, which means E(size) = mean_pkt_size
    size_dist = partial(expovariate, 1.0 / args.mean_pkt_size)
    true_flow_size = dict()
    # print(zipf_flows[:10])

    # set all flows with packet gen and packet sink
    for fid in all_flows:
        # since the flow size of time T is usually believed to be poisson distribution
        # the interval of two consecutive packets follows exp distribution
        arr_dist = partial(expovariate, args.arr_dist_alpha)
        pg = ZipfPacketGenerator(
            env,
            f"Flow_{fid}",
            arr_dist,
            size_dist,
            flow_size=zipf_flows[fid],
            flow_id=fid,
            begin=args.begin,
            end=args.end,
        )
        true_flow_size[fid] = pg.flow_size
        ps = PacketSink(env, rec_arrivals=False, rec_flow_ids=False, rec_waits=False)
        all_flows[fid].pkt_gen = pg
        all_flows[fid].size = pg.flow_size
        all_flows[fid].pkt_sink = ps

    # sample high priority flows
    MonitorData.sampled_high_flow_set = utils.initial_sample_priority(
        all_flows, int(len(all_flows) * 0.05)
    )
    MonitorData.unsampled_flow_set = set(all_flows.keys()) - MonitorData.sampled_high_flow_set

    total_packet_num = sum(true_flow_size.values())
    sortedFlow = sorted(true_flow_size.items(), key=lambda x: x[1], reverse=True)
    print(f"Total number of packets = {total_packet_num}")
    logging.info(f"Total number of packets = {total_packet_num}")
    print(sortedFlow[:10])
    logging.info(sortedFlow[:10])
    # logging.info(true_flow_size)
    # comment the following line, if you want more packets
    assert total_packet_num < 20000000

    # generate fib of each switch
    ft = generate_fib(ft, all_flows)
    n_classes_per_port = 4
    weights = {c: 1 for c in range(n_classes_per_port)}

    def flow_to_classes(f_id, n_id=0, fib=None):
        return (f_id + n_id + fib[f_id]) % n_classes_per_port

    for node_id in ft.nodes():
        node = ft.nodes[node_id]
        flow_classes = partial(flow_to_classes, n_id=node_id, fib=node["flow_to_port"])
        # normal switches
        if node_id not in memory_limit_switches:
            node["device"] = MonitorSwitchDHT(
                env,
                ft,
                args.sketch_memory,
                args.is_dht,
                args.k,
                args.pir,
                args.buffer_size,
                weights,
                "DRR",
                flow_classes,
                element_id=node_id,
                all_flows=all_flows,
            )
        # limited memory switches
        else:
            node["device"] = MonitorSwitchDHT(
                env,
                ft,
                args.sketch_memory_limit,
                args.is_dht,
                args.k,
                args.pir,
                args.buffer_size,
                weights,
                "DRR",
                flow_classes,
                element_id=node_id,
                all_flows=all_flows,
            )
        node["device"].demux.fib = node["flow_to_port"]

    for n in ft.nodes():
        node = ft.nodes[n]
        for port_number, next_hop in node["port_to_nexthop"].items():
            node["device"].ports[port_number].out = ft.nodes[next_hop]["device"]
    for target_flow_id, flow in all_flows.items():
        flow.pkt_gen.out = ft.nodes[flow.src]["device"]
        ft.nodes[flow.dst]["device"].demux.ends[target_flow_id] = flow.pkt_sink

    controller = GlobalController(
        env,
        args.do_dynamic,
        interval=args.inter,
        finish=args.finish,
        topo=ft,
        all_flows=all_flows,
    )
    MonitorData.setHashRings(args.k, 500)
    MonitorData.initialWeight(ft)
    MonitorData.setNumOfMemory(len(args.memory))

    if args.do_limit_memory:
        for index in memory_limit_switches:
            MonitorData.adjustNodeWeight(ft, index, args.memory_limit_ratio)

    print("Global weights of switches", MonitorData.global_weights)
    logging.info(f"Global weights of switches: {MonitorData.global_weights}")

    headlinesPrint("Simulation Started")
    tic1 = time.perf_counter()
    env.run(until=100001)
    tic2 = time.perf_counter()
    headlinesPrint(f"Simulation Finished at {env.now}")

    print(f"Simulation consumed time {tic2 - tic1} (s).")
    logging.info(f"Simulation consumed time {tic2 - tic1} (s).")

    headlinesPrint("Tasks")
    logging.info(f"Start tasks...")
    from Tasks import (
        queryflow,
        flow_estimate,
        heavy_hitter,
        heavy_change,
        flow_cardinality,
        flow_entropy,
        max_interval,
    )

    tic1 = time.perf_counter()
    # 1. true = {flow_id:[flow_num, ts, max_interval], }
    # 2. est_hasflow = {}
    # 3. est_hasflow_high = {}
    # 4. est_hasflow_low = {}
    # 5. device_true_segment = device_true_segment[device_id][segment_idx] = {flow_id:[flow_num, ts, max_interval], }
    # 6. result_segment
    # 7. high_sketch_segment
    # 8. low_sketch_segment
    # 9. ground_truth_segment_len[device_id][segment_idx] = the true flow number in this segment
    # 0. entropy_segment
    (
        true,
        est_hasflow,
        est_hasflow_high,
        est_hasflow_low,
        device_true_segment,
        result_segment,
        high_sketch_segment,
        low_sketch_segment,
        ground_truth_segment_len,
        # entropy_segment,
    ) = queryflow(ft, args.memory, args)

    print(f"{len(est_hasflow[0])} / {len(true)} // {len(args.memory)}")

    # device_true_segment_result[segment_idx] = {flow_id:[flow_num, ts, max_interval], }
    device_true_segment_result = merge_switch_to_segment(device_true_segment)

    (
        all_are_esti,
        all_aae_esti,
        all_are_esti_high,
        all_aae_esti_high,
        all_are_esti_low,
        all_aae_esti_low,
        all_are_hit,
        all_aae_hit,
        all_precision,
        all_recall,
        all_f1,
        all_cardinality,
        all_are_max,
        all_aae_max,
        all_entropy,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    all_pre_hit, all_recall_hit, all_f1_hit = [], [], []
    all_distri = []
    for num in range(len(args.memory)):
        # print(f"\n{num}, {args.memory[num]/1024} KB: ")

        are_esti, aae_esti = flow_estimate(true, est_hasflow[num], args.n_flows)
        are_esti_high, aae_esti_high = flow_estimate(true, est_hasflow_high[num], args.n_flows)
        are_esti_low, aae_esti_low = flow_estimate(true, est_hasflow_low[num], args.n_flows)

        # are_hit, aae_hit, pre_hit, recall_hit, f1_hit = heavy_hitter(
        #     device_true_segment_result, result_segment[num], args.heavy_hitter_thres
        # )

        # precision, recall, f1 = heavy_change(
        #     args.heavy_change_thres, device_true_segment_result, result_segment[num]
        # )

        # cardinality = flow_cardinality(
        #     high_sketch_segment[num],
        #     low_sketch_segment[num],
        #     ground_truth_segment_len,
        # )

        are_max, aae_max = max_interval(dict(true), dict(est_hasflow[num]), args.n_flows)

        # entropy = flow_entropy(device_true_segment_result, entropy_segment[num])

        all_are_esti.append(are_esti)
        all_aae_esti.append(aae_esti)
        all_are_esti_high.append(are_esti_high)
        all_aae_esti_high.append(aae_esti_high)
        all_are_esti_low.append(are_esti_low)
        all_aae_esti_low.append(aae_esti_low)
        # all_are_hit.append(are_hit)
        # all_aae_hit.append(aae_hit)
        # all_pre_hit.append(pre_hit)
        # all_recall_hit.append(recall_hit)
        # all_f1_hit.append(f1_hit)
        # all_precision.append(precision)
        # all_recall.append(recall)
        # all_f1.append(f1)
        # all_cardinality.append(cardinality)
        all_are_max.append(are_max)
        all_aae_max.append(aae_max)
        # all_entropy.append(entropy)
    tic2 = time.perf_counter()
    print(f"Calculation consumed time {tic2 - tic1} (s).")
    logging.info(f"Calculation consumed time {tic2 - tic1} (s).")

    logging.info(
        f"\n\nn_flows: {args.n_flows}, memory:{args.memory}, is_dht:{args.is_dht}, heavy_hitter_thres:{args.heavy_hitter_thres}, heavy_change_ratio:{args.heavy_change_ratio}"
    )
    logging.info(args)
    logging.info(
        f"\n\t\t\testimate:\nare:{np.round(all_are_esti,5)} \naae:{np.round(all_aae_esti,5)}"
    )
    logging.info(
        f"\n\t\t\testimate_high:\nare:{np.round(all_are_esti_high,5)} \naae:{np.round(all_aae_esti_high,5)}"
    )
    logging.info(
        f"\n\t\t\testimate_low:\nare:{np.round(all_are_esti_low,5)} \naae:{np.round(all_aae_esti_low,5)}"
    )
    # logging.info(
    #     f"\n\t\t\theavyhitter:\nare:{np.round(all_are_hit,5)} \naae:{np.round(all_aae_hit,5)}, \npre:{np.round(all_pre_hit,5)} \nrecall:{np.round(all_recall_hit,5)} \nf1:{np.round(all_f1_hit,5)}"
    # )
    # logging.info(
    #     f"\n\t\t\theavychange:\npre:{np.round(all_precision,5)} \nrecall:{np.round(all_recall,5)} \nf1:{np.round(all_f1,5)}"
    # )
    # logging.info(f"\n\t\t\tcardinality:\n{np.round(all_cardinality, 5)}")
    logging.info(
        f"\n\t\t\tmax_inter:\nare:{np.round(all_are_max,5)} \naae:{np.round(all_aae_max,5)}"
    )
    # logging.info(f"\t\t\tentropy:\n{np.round(all_entropy,5)}")
    logging.info(time.asctime())
