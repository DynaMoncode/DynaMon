import copy
import math
import logging
import array

from collections import defaultdict
import numpy as np
from typing import Dict
from utils import merge_dict_with_max, merge_switch_to_segment


# deprecated function
def ArE_withmax(true, estimate: Dict):
    are, aae = 0.0, 0.0
    if len(true) == 0 or len(estimate) == 0:
        return 0.0, 0.0
    for data in true:
        if data[0] in estimate.keys():
            tru, est = data[1][0], estimate[data[0]][0]
            dist = float(abs(est - tru))
            are += dist / tru
            aae += dist
        else:
            are += 1
            aae += float(abs(data[1][0]))
    are /= len(true)
    aae /= len(true)
    return are, aae


def ArE(true: Dict, estimate: Dict, n_flows: int, percentages = [1, 5, 10, 15, 100]):
    are_results = []
    aae_results = []
    indices = [int(n_flows * p / 100) - 1 for p in percentages]
    for idx in indices:
        are, aae = 0.0, 0.0
        if len(true) == 0 or len(estimate) == 0:
            return 0.0, 0.0
        for flowId in range(idx):
            if flowId in true:
                if flowId in estimate:
                    est, tru = estimate[flowId], true[flowId]
                    dist = float(abs(est - tru))
                    are += dist / tru
                    aae += dist
                else:
                    pass
                    # are += 1
                    # aae += float(abs(data[1]))
        are /= len(estimate)
        aae /= len(estimate)
        are_results.append(are)
        aae_results.append(aae)
    return are_results, aae_results


#! query flow
def queryflow(ft, memory, args):
    # shape = memory * -1
    all_result_flow_table_hash_id = []  # truth flowid query sketch
    all_high_result_all_has_flowid = []
    all_low_result_all_has_flowid = []
    all_flow_table_segment = []  # hashtable flowid
    all_high_sketch_segment = []  # high_sketch_segment
    all_low_sketch_segment = []  # low_sketch_segment
    # all_high_flow_table_segment = []
    # all_low_flow_table_segment = []
    # all_entropy_segment = []  # entropy_segment


    ground_truth_segment_len = []
    # the ground truth of all flows
    device_true_flow = defaultdict(list)
    # ground truth of each segment per segment per switch, device_true_segment[device_id][segment_idx]
    device_true_segment = []
    for node_id in ft.nodes():
        node = ft.nodes[node_id]
        if node["type"] == "switch":
            device_true_flow = merge_dict_with_max(device_true_flow, node["device"].ground_truth)
            device_true_segment.append(node["device"].ground_truth_segment)
            for data in node["device"].ground_truth_segment:
                ground_truth_segment_len.append(len(data))
    # _1 = sorted(device_true_flow.items(), key=lambda x: x[1], reverse=True)
    # print("device_true_flow", _1[:20])

    # # print(f"seg:{np.array(ground_truth_segment_len).reshape(args.switch_num, -1)}")
    # logging.info(f"seg:{np.array(ground_truth_segment_len).reshape(args.switch_num, -1)}")
    # # print(f"seg:{(np.array(ground_truth_segment_len).reshape(args.switch_num, -1).sum(axis=0))}")
    # logging.info(f"seg:{(np.array(ground_truth_segment_len).reshape(args.switch_num, -1).sum(axis=0))}")
    # # print(f"seg:{(np.array(ground_truth_segment_len).reshape(args.switch_num, -1).sum(axis=1))}")
    # logging.info(f"seg:{(np.array(ground_truth_segment_len).reshape(args.switch_num, -1).sum(axis=1))}")

    # flow_number_in_each_monitor
    device_true_switch_length = []
    for switch in device_true_segment:
        temp = {}
        for data in switch:
            temp = merge_dict_with_max(temp, data)
        device_true_switch_length.append(len(temp))
    # print(f"flow_number_in_each_monitor:{device_true_switch_length}")
    # print(f"the_sum_of_flow_number_in_each_number:{sum(device_true_switch_length)}")
    logging.info(f"each_monitor_flow_num: {device_true_switch_length}")

    # data collection for PrioritySketch
    for num in range(len(memory)):
        flow_table_segment = []
        high_table_segment = []
        low_table_segment = []

        high_keys_segment = []
        low_keys_segment = []

        high_sketch_segment = []
        low_sketch_segment = []

        # entropy_segment = []

        for node_id in ft.nodes():
            node = ft.nodes[node_id]
            if node["type"] == "switch":
                flow_table_segment.append([])
                high_table_segment.append([])
                low_table_segment.append([])
                high_keys_segment.append(node["device"].SKETCH[num].segmentKeysHigh)
                low_keys_segment.append(node["device"].SKETCH[num].segmentKeysLow)
                high_sketch_segment.append(node["device"].SKETCH[num].segmentHigh)
                low_sketch_segment.append(node["device"].SKETCH[num].segmentLow)
                # entropy_segment.append(node["device"].SKETCH[num].entropy_num_segment)

        for switch in flow_table_segment:
            for _ in range(10):
                switch.append(defaultdict(list))
        for switch in high_table_segment:
            for _ in range(10):
                switch.append(defaultdict(list))
        for switch in low_table_segment:
            for _ in range(10):
                switch.append(defaultdict(list))

        # high sketch
        for switch, (switch_keys, switch_sketch) in enumerate(
            zip(high_keys_segment, high_sketch_segment)
        ):
            for segment, (segment_keys, segment_sketch) in enumerate(
                zip(switch_keys, switch_sketch)
            ):
                for key in segment_keys:
                    sum_ans, ts_ans, int_ans = segment_sketch.query(key)
                    high_table_segment[switch][segment][key] = [
                        sum_ans,
                        ts_ans,
                        int_ans,
                    ]
                    flow_table_segment[switch][segment][key] = [
                        sum_ans,
                        ts_ans,
                        int_ans,
                    ]
        # low sketch
        for switch, (switch_keys, switch_sketch) in enumerate(
            zip(low_keys_segment, low_sketch_segment)
        ):
            for segment, (segment_keys, segment_sketch) in enumerate(
                zip(switch_keys, switch_sketch)
            ):
                for key in segment_keys:
                    sum_ans, ts_ans, int_ans = segment_sketch.query(key)
                    low_table_segment[switch][segment][key] = [
                        sum_ans,
                        ts_ans,
                        int_ans,
                    ]
                    flow_table_segment[switch][segment][key] = [
                        sum_ans,
                        ts_ans,
                        int_ans,
                    ]

        # merge all measurement result
        high_result_all_has_flowid = defaultdict(list)
        for switch in high_table_segment:
            for segment in switch:
                high_result_all_has_flowid = merge_dict_with_max(
                    high_result_all_has_flowid, segment
                )
        low_result_all_has_flowid = defaultdict(list)
        for switch in low_table_segment:
            for segment in switch:
                low_result_all_has_flowid = merge_dict_with_max(low_result_all_has_flowid, segment)
        result_all_has_flowid = defaultdict(list)
        for switch in flow_table_segment:
            for segment in switch:
                result_all_has_flowid = merge_dict_with_max(result_all_has_flowid, segment)

        all_flow_table_segment.append(flow_table_segment)
        # all_high_flow_table_segment.append(high_table_segment)
        # all_low_flow_table_segment.append(low_table_segment)

        all_result_flow_table_hash_id.append(result_all_has_flowid)
        all_high_result_all_has_flowid.append(high_result_all_has_flowid)
        all_low_result_all_has_flowid.append(low_result_all_has_flowid)

        # all_hash_table_keys_segment.append(high_keys_segment)
        # all_hash_table_keys_segment.append(low_keys_segment)
        all_high_sketch_segment.append(high_sketch_segment)
        all_low_sketch_segment.append(low_sketch_segment)
        # all_entropy_segment.append(entropy_segment)

    return (
        device_true_flow,  # true
        all_result_flow_table_hash_id,  # est
        all_high_result_all_has_flowid,  # est high
        all_low_result_all_has_flowid,  # est low
        device_true_segment,  # device true
        all_flow_table_segment,  # result segment
        all_high_sketch_segment,  # high sketch segment
        all_low_sketch_segment,  # low sketch segment
        np.array(ground_truth_segment_len).reshape(args.switch_num, -1),
        # all_entropy_segment,
    )


#! flow estimate
def flow_estimate(true, est, n_flows: int, percentages = [1, 5, 10, 15, 100]):
    # print(true)
    # print(est)
    true = {k: v[0] for k, v in true.items()}
    est = {k: v[0] if len(v) != 0 else 0 for k, v in dict(est).items()}
    return ArE(true, est, n_flows, percentages)


#! heavy_hitter
def heavy_hitter(true, est, threshold):
    est = merge_switch_to_segment(est)
    # [10]true   [10]est
    are, aae, pre, rec, f1 = [], [], [], [], []
    for segment in range(len(true)):
        device_true_heavy = {k: v[0] for k, v in true[segment].items() if v[0] >= threshold}
        result_true_heavy = {k: v[0] for k, v in est[segment].items() if v[0] >= threshold}
        if not device_true_heavy or not result_true_heavy:
            return 0, 0, 0, 0, 0

        _1 = sorted(device_true_heavy.items(), key=lambda x: x[1], reverse=True)
        _2 = sorted(result_true_heavy.items(), key=lambda x: x[1], reverse=True)

        _1, _2 = ArE(device_true_heavy, result_true_heavy)
        are.append(_1)
        aae.append(_2)

        intersection_len = len(list(set(device_true_heavy).intersection(set(result_true_heavy))))
        truth_len, estimate_len = len(device_true_heavy), len(result_true_heavy)
        if estimate_len == 0:
            pre_temp = 1 if truth_len == 0 else 0
        else:
            pre_temp = intersection_len / estimate_len
        if truth_len == 0:
            recall_temp = 1
        else:
            recall_temp = intersection_len / truth_len
        if (pre_temp + recall_temp) == 0:
            return 0, 0, 0, 0, 0
        f1_temp = (2 * pre_temp * recall_temp) / (pre_temp + recall_temp)

        pre.append(pre_temp)
        rec.append(recall_temp)
        f1.append(f1_temp)

    return np.mean(are), np.mean(aae), np.mean(pre), np.mean(rec), np.mean(f1)


#! heavy_change
def heavy_change(threshold, ground_truth_segment, result_segment):
    def compute_heavy_change(dict1, dict2):
        result = []
        for key in list(dict1.keys() & dict2.keys()):
            if abs(dict1[key][0] - dict2[key][0]) >= threshold:
                result.append(key)

        for key in list((dict1.keys() | dict2.keys()) ^ (dict1.keys() & dict2.keys())):
            if key in dict1:
                if abs(dict1[key][0]) >= threshold:
                    result.append(key)
            else:
                if abs(dict2[key][0]) >= threshold:
                    result.append(key)
        return result

    result_segment = merge_switch_to_segment(result_segment)

    change_ground_truth = []
    change_result_sketch = []
    for i in range(len(ground_truth_segment) - 1):
        if not ground_truth_segment[i + 1]:
            break
        change_ground_truth.append(
            compute_heavy_change(ground_truth_segment[i], ground_truth_segment[i + 1])
        )
        change_result_sketch.append(compute_heavy_change(result_segment[i], result_segment[i + 1]))

    # print(f"heavy_change_true length:{len(change_ground_truth)}")
    # logging.info(f"heavy_change_true length:{len(change_ground_truth)}")
    # print(f"heavy_change_esti length:{len(change_result_sketch)}")
    # logging.info(f"heavy_change_esti length:{len(change_result_sketch)}")

    pre, rec, f1 = [], [], []
    for truth, estimate in zip(change_ground_truth, change_result_sketch):
        if len(truth) == 0:
            continue

        intersection_len = len(list(set(truth).intersection(set(estimate))))
        truth_len, estimate_len = len(truth), len(estimate)

        if estimate_len == 0:
            pre_temp = 1 if truth_len == 0 else 0
        else:
            pre_temp = intersection_len / estimate_len
        if truth_len == 0:
            rec_temp = 1
        else:
            rec_temp = intersection_len / truth_len
        if (pre_temp + rec_temp) == 0:
            continue
        f1_temp = (2 * pre_temp * rec_temp) / (pre_temp + rec_temp)
        pre.append(pre_temp)
        rec.append(rec_temp)
        f1.append(f1_temp)

    return np.mean(pre), np.mean(rec), np.mean(f1)


#! flow_entropy
def flow_entropy(true, est):
    def get_entropy(value):
        sum = 0.0
        for k, v in value.items():
            sum += k * v
        entropy = 0
        for k, v in value.items():
            if k == 0:
                continue
            entropy += v * (int(k) / sum) * math.log(int(k) / sum)
        return -1 * entropy

    all_entropy = []
    est = merge_switch_to_segment(est, hasmax=False)

    ent_true, ent_est = [], []
    for index, (true_segment, est_segment) in enumerate(zip(true, est)):
        if index == 1:
            continue
        value_true = {}
        for k, v in true_segment.items():
            if v[0] in value_true.keys():
                value_true[v[0]] += 1
            else:
                value_true[v[0]] = 1
        entropy_true = get_entropy(value_true)

        entropy_esti = get_entropy(est_segment)
        ent_true.append(entropy_true)
        ent_est.append(entropy_esti)
        if entropy_true == 0:
            continue
        all_entropy.append(np.abs(entropy_true - entropy_esti) / entropy_true)

    return np.mean(all_entropy)


#! flow_cardinality
def flow_cardinality(sketch_segment_1, sketch_segment_2, ground_truth_segment_len):
    cardinality = []
    true_sum = []
    e_sum = []
    card_result = []
    for sumaxske_switch_1, sumaxske_switch_2, true_switch in zip(
        sketch_segment_1, sketch_segment_2, ground_truth_segment_len
    ):
        for index, (sumaxske1, sumaxske2, true) in enumerate(
            zip(sumaxske_switch_1, sumaxske_switch_2, true_switch)
        ):
            if index == 1:
                break
            # sumax sketch 1
            m = sumaxske1.m
            e = sumaxske1.get_zero_count()
            if e == 0:
                continue
            card = m * math.log(m / e)
            # sumax sketch 2
            m = sumaxske2.m
            e = sumaxske2.get_zero_count()
            if e == 0:
                continue
            card += m * math.log(m / e)

            cardinality.append(card)
            e_sum.append(e)
            if true == 0:
                continue
            card_result.append(np.abs(card - true) / true)
            true_sum.append(true)

    return np.mean(card_result)


#! max_interval
def max_interval(true, estimate, n_flows, percentages = [1, 5, 10, 15, 100]):
    are_results = []
    aae_results = []
    indices = [int(n_flows * p / 100) - 1 for p in percentages]
    for idx in indices:
        number_of_flows = idx
        are, aae = 0.0, 0.0
        sum = 0
        # 查询大流，由于流生成函数的固有特性，flowId越小的流，包数越大
        # 这里利用了这一点，直接遍历range
        for flowId in range(number_of_flows):
            if flowId in true:
                tru = true[flowId]
                if tru[0] < 2:
                    continue
                else:
                    true_max = tru[2]
                    if true_max < 2:
                        continue
                    sum += 1
                    if flowId in estimate.keys():
                        if estimate[flowId][0] < 2:
                            continue
                        are += float(abs(true_max - estimate[flowId][2])) / true_max
                        aae += float(abs(true_max - estimate[flowId][2]))
                    else:
                        are += 1
                        aae += float(abs(true_max))
        if sum == 0:
            return 0, 0
        are /= sum
        aae /= sum

        aae_results.append(aae)
        are_results.append(are)
    return are_results, aae_results
