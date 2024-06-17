# -*- coding: utf-8 -*-
import hashlib
import array
import copy

# borrowed from https://github.com/rafacarrascosa/countminsketch
import math

# -*- coding: utf-8 -*-
import hashlib
import array

# from mrac import *
from collections import defaultdict
from utils import merge_dict_with_max, max_dict_update
import MonitorData

class HalfCUsketch_withmax(object):
    def __init__(self, m, d):
        if not m or not d:
            m = 1
            d = 1
            # raise ValueError("Table size (m) and amount of hash functions (d)"
            #  " must be non-zero")
        self.m = m
        self.d = d
        self.n = 0
        self.sum_tables = []
        self.MAX = 65535
        self.ts_tables = []  # array -> timestamp _ table
        self.int_tables = []  # array -> max interval table
        self.MIN = 1e-5
        # sum_tables use 1/2 memory
        # ts_tables use 1/4 memory
        # int_tables use 1/4 memory
        for _ in range(d):  # sum table    unsigned int 4Bytes
            table = array.array("l", (0 for _ in range(m)))
            self.sum_tables.append(table)
        for _ in range(d):
            table = array.array("f", (0 for _ in range(int(m))))  # 1/2 memory
            self.ts_tables.append(table)
        for _ in range(d):
            table = array.array("f", (0 for _ in range(int(m))))  # 1/2 memory
            self.int_tables.append(table)

    def _hash(self, x):
        md5 = hashlib.md5(str(hash(x)).encode("utf-8"))
        hash_d = []
        for i in range(self.d):
            md5.update(str(i).encode("utf-8"))
            hash_d.append(int(md5.hexdigest(), 16) % self.m)
        return hash_d

    def add(self, x, value=1, y=None):
        self.n += value

        hash_d = self._hash(x)
        # sum
        while value:
            minus = value
            minp, secp = self.MAX, self.MAX
            for table, i in zip(self.sum_tables, hash_d):
                # print(table)
                if minp >= table[i]:
                    minp = table[i]
                else:
                    secp = min(secp, table[i])
                    minus = min(minus, secp - minp)

            minp = self.MAX
            for i, table in zip(range(self.d), self.sum_tables):
                minp = min(minp, table[hash_d[i]])
                if table[hash_d[i]] == minp:
                    table[hash_d[i]] += minus

            value -= minus

        # max
        for num, (ts_table, int_table, i) in enumerate(
            zip(self.ts_tables, self.int_tables, hash_d)
        ):
            last_ts = ts_table[i]
            last_int = int_table[i]
            if abs(last_ts) == 0.0:
                self.ts_tables[num][i] = y
            else:
                interval = y - last_ts
                self.ts_tables[num][i] = y
                self.int_tables[num][i] = max(last_int, interval)

    def query(self, x, only_sum=False):
        if only_sum:
            sum_ans = self.MAX
            for table, i in zip(self.sum_tables, self._hash(x)):
                sum_ans = min(sum_ans, table[i])
            return sum_ans
        else:
            sum_ans = self.MAX
            for table, i in zip(self.sum_tables, self._hash(x)):
                sum_ans = min(sum_ans, table[i])

            ts_ans, int_ans = self.MIN, self.MAX
            for ts_tables, int_tables, i in zip(
                self.ts_tables, self.int_tables, self._hash(x)
            ):
                ts_ans = max(ts_ans, ts_tables[i])
                int_ans = min(int_ans, int_tables[i])
            return sum_ans, ts_ans, int_ans

    def __getitem__(self, x):
        return self.query(x)

    def __len__(self):
        return self.n

    def get_zero_count(self):
        zero = 0.0
        for counter in self.sum_tables[0]:
            if counter == 0.0:
                zero += 1
        return int(zero)

    def __repr__(self) -> str:
        return f"SuMax Sketch with M = {self.m}, D = {self.d}"


class PrioritySketch(object):
    """
    memory in bytes
    change the parameter h2lCoefficient to change the high/low ratio
    """

    def __init__(
        self,
        memory: int,
        hFlowNum: int,
        lFlowNum: int,
        h2lCoefficient: int = 2,
        D: int = 3,
    ) -> None:
        # assign a coefficient to hFlowNum
        hRatio = hFlowNum * h2lCoefficient / (hFlowNum * h2lCoefficient + lFlowNum)
        # every counter consists of 4 + 2 + 2 = 8 Bytes
        totalM = int(memory / D / 4 / 2)
        self.highM = int(totalM * hRatio)
        self.highD = D
        self.highSketch = HalfCUsketch_withmax(self.highM, self.highD)

        self.lowM = totalM - self.highM
        self.lowD = D
        self.lowSketch = HalfCUsketch_withmax(self.lowM, self.lowD)

        self.this_segment_keys_high = set()
        self.this_segment_keys_low = set()
        
        self.segmentHigh = []
        self.segmentLow = []
        self.segmentKeysHigh = []
        self.segmentKeysLow = []

    def insert(self, x, isHigh: bool, freq, cur_ts):
        if isHigh:
            self.highSketch.add(x, freq, cur_ts)
            self.this_segment_keys_high.add(x)
        else:
            self.lowSketch.add(x, freq, cur_ts)
            self.this_segment_keys_low.add(x)

    def query(self, x, isHigh: bool):
        res = self.highSketch.query(x) if isHigh else self.lowSketch.query(x)
        return res

    def change_segment(self):
        # when a segment finishes, we do nothing using PrioritySketch
        # do some data collection
        self.segmentHigh.append(copy.deepcopy(self.highSketch))
        self.segmentLow.append(copy.deepcopy(self.lowSketch))
        self.segmentKeysHigh.append(copy.deepcopy(self.this_segment_keys_high))
        self.segmentKeysLow.append(copy.deepcopy(self.this_segment_keys_low))

    def reset_sketch(self, memory, hFlowNum, lFlowNum, h2lCoefficient=2, D=3):
        # the same process as __init__ function
        # assign a coefficient to hFlowNum
        hRatio = hFlowNum * h2lCoefficient / (hFlowNum * h2lCoefficient + lFlowNum)
        # every counter consists of 4 + 2 + 2 = 8 Bytes
        totalM = int(memory / D / 4 / 2)
        self.highM = int(totalM * hRatio)
        self.highD = D
        self.highSketch = HalfCUsketch_withmax(self.highM, self.highD)

        self.lowM = totalM - self.highM
        self.lowD = D
        self.lowSketch = HalfCUsketch_withmax(self.lowM, self.lowD)


class SuMax_withmax(object):
    def __init__(
        self, memory, h_flowtable_size, flowtable_size, hashtable_size, threshold_ratio
    ):
        # h-Flows
        self.h_flows = defaultdict(list)
        self.h_flowtable_size = h_flowtable_size

        # l-flows
        self.d = 3
        self.m = int(memory / self.d / 4 / 2)
        self.Sketch = HalfCUsketch_withmax(self.m, self.d)

        # m-flows
        self.flowtable_size = flowtable_size
        self.hashtable_size = hashtable_size
        self.flow_table = defaultdict(
            list
        )  # key : [freq, last_timestamp, max_interval]
        self.hash_table = defaultdict(list)

        self.flowtable_sum = defaultdict(list)
        self.flow_table_segment = []
        self.sketch_segment = []
        self.hash_table_keys_segment = []

        self.first_segement = True
        self.threshold = None
        self.threshold_ratio = threshold_ratio
        self.flowtable_min = 1

        # for entropy estimation
        self.entropy_num_segment = []
        self.entorpy_num = {}

    def try_h_flow_insert(self, x, freq, cur_ts):
        if x in self.h_flows:
            # update the freq
            self.h_flows[x][0] += freq
            # not the first flow
            if self.h_flows[x][0] > 1:
                self.h_flows[x][2] = max(
                    self.h_flows[x][2], cur_ts - self.h_flows[x][1]
                )
            # update timestamp
            self.h_flows[x][1] = cur_ts
            return True
        else:
            return False

    # x : the item to be inserted
    # freq: the frequency of x, freq = 1 by default
    # cur_ts: current timestamp
    def insert(self, x, freq, cur_ts):
        if x in self.flow_table:
            # update freq
            self.flow_table[x][0] += freq
            # update max interval
            if self.flow_table[x][0] > 1:
                self.flow_table[x][2] = max(
                    self.flow_table[x][2], cur_ts - self.flow_table[x][1]
                )
            # update timestamp
            self.flow_table[x][1] = cur_ts
        else:
            former_sum_result = self.Sketch.query(x, only_sum=True)

            self.Sketch.add(x, freq, cur_ts)

            sum_result, ts_result, int_result = self.Sketch.query(x)
            # first segment
            if self.first_segement:
                self.hash_table[x] = [sum_result, ts_result, int_result]

                if len(self.hash_table) > self.hashtable_size:
                    sort_hash_table = sorted(
                        self.hash_table.items(), key=lambda x: x[1][0], reverse=True
                    )
                    self.hash_table = dict(sort_hash_table[: self.hashtable_size])
            # after first segment, the threshold is set
            else:
                # sum
                if sum_result >= self.threshold[0] and sum_result <= self.threshold[-1]:
                    if sum_result in self.threshold:
                        if x in self.hash_table:
                            self.hash_table = max_dict_update(
                                self.hash_table, x, sum_result, ts_result, int_result
                            )
                        else:
                            self.hash_table[x] = [sum_result, ts_result, int_result]

                    if len(self.hash_table) > self.hashtable_size:
                        sort_hash_table = sorted(
                            self.hash_table.items(), key=lambda x: x[1][0], reverse=True
                        )
                        self.hash_table = dict(sort_hash_table[: self.hashtable_size])
            # entropy
            if former_sum_result in self.entorpy_num:
                if self.entorpy_num[former_sum_result] > 0:
                    self.entorpy_num[former_sum_result] -= 1
            if sum_result in self.entorpy_num:
                self.entorpy_num[sum_result] += 1
            else:
                self.entorpy_num[sum_result] = 1

    def change_segment(self):
        # print("changesegment")
        for k, v in self.hash_table.items():
            self.flow_table[k] = [v[0], v[1], v[2]]

        # entropy related operation
        if self.first_segement:
            pass
        else:
            for k, v in self.flow_table.items():
                if v[0] in self.entorpy_num:
                    self.entorpy_num[v[0]] += 1
                else:
                    self.entorpy_num[v[0]] = 1

        if len(self.flow_table) == 0:
            pass
        elif len(self.flow_table) > self.flowtable_size:
            # print(">>>")
            sort_flow_table = sorted(
                self.flow_table.items(), key=lambda x: x[1][0], reverse=True
            )
            self.flowtable_min = sort_flow_table[self.flowtable_size][1][0]
            self.flow_table = dict(sort_flow_table[: self.flowtable_size])
        else:
            # print("else")
            sort_flow_table = sorted(
                self.flow_table.items(), key=lambda x: x[1][0], reverse=True
            )
            self.flowtable_min = sort_flow_table[-1][1][0]
            self.flowtable_sum = merge_dict_with_max(
                self.flowtable_sum, self.flow_table
            )

        self.first_segement = True
        self.threshold = [
            math.ceil(i * self.flowtable_min) for i in self.threshold_ratio
        ]

        self.flow_table_segment.append(copy.deepcopy(self.flow_table))
        self.hash_table_keys_segment.append(self.hash_table.keys())
        self.sketch_segment.append(copy.deepcopy(self.Sketch))
        self.entropy_num_segment.append(copy.deepcopy(self.entorpy_num))

    def reset_ske(self, mem):
        self.d = 3
        self.m = int(mem / self.d / 4 / 2)
        self.Sketch = HalfCUsketch_withmax(self.m, self.d)

        # clear the freq and interval, keep the current ts domain
        for k, v in self.flow_table.items():
            self.flow_table[k] = [0, v[1], 0]

        for k, v in self.h_flows.items():
            self.h_flows[k] = [0, v[1], 0]
            if k in self.flow_table:
                self.flow_table.pop(k)

        self.hash_table.clear()
        self.entorpy_num.clear()
