import copy
from os import abort
from struct import pack
from ns.switch.switch import FairPacketSwitch
from ConsisSketch_withmax import PrioritySketch
from collections.abc import Callable
from ring import HashRing
from ns.packet.packet import Packet
import logging
from collections import defaultdict
from utils import merge_dict_with_max, max_dict_update
from typing import List
from MonitorData import MonitorData
import mmh3


class MonitorSwitchDHT(FairPacketSwitch):
    def __init__(
        self,
        env,
        topo,
        memory: int,
        is_dht: bool,
        nports: int,
        port_rate: float,
        buffer_size: int,
        weights,
        server: str,
        flow_classes: Callable = lambda x: x,
        element_id: str = "",
        debug: bool = False,
        all_flows: dict = {},
    ) -> None:
        super().__init__(
            env,
            nports,
            port_rate,
            buffer_size,
            weights,
            server,
            flow_classes,
            element_id,
            debug,
        )
        self.topo = topo
        self.global_flows = all_flows
        self.element_id = element_id
        self.memory = memory

        self.SKETCH: List[PrioritySketch] = []
        for i in range(len(self.memory)):
            self.SKETCH.append(
                PrioritySketch(
                    self.memory[i],
                    len(MonitorData.sampled_high_flow_set),
                    len(MonitorData.unsampled_flow_set),
                )
            )

        self.ground_truth = defaultdict(list)
        self.segment_ground_truth = defaultdict(list)
        self.ground_truth_segment = []

        self.valid = 1.0
        self.is_dht = is_dht
        if is_dht:
            self.dht = MonitorData.global_dht
        else:
            self.dht = lambda x: mmh3.hash(str(x), 1411, signed=False) % 5

    def put(self, packet: Packet):
        # ! host, put the packet directly
        if self.topo.nodes[self.element_id]["type"] == "host":
            self.demux.put(packet)
            return
        curLayer = self.topo.nodes[self.element_id]["layer"]
        toInsert = packet.flow_id
    
        # baseline method, just record and put the packet downstream
        if self.is_dht == False:
            MonitorData.flow_to_switches.setdefault(packet.flow_id, set())
            MonitorData.flow_to_switches[packet.flow_id].add(
                self.element_id
            )

            MonitorData.switches_to_flow.setdefault(self.element_id, set())
            MonitorData.switches_to_flow[self.element_id].add(
                packet.flow_id
            )
            toInsert = packet.flow_id

            if toInsert in self.segment_ground_truth:
                # self.hash_table = max_dict_update(self.hash_table, sum_result, ts_result, int_result)
                now = self.env.now
                self.segment_ground_truth[toInsert][0] += 1
                self.segment_ground_truth[toInsert][2] = max(
                    self.segment_ground_truth[toInsert][2],
                    now - self.segment_ground_truth[toInsert][1],
                )
                self.segment_ground_truth[toInsert][1] = now
            else:
                self.segment_ground_truth[toInsert] = [1, self.env.now, 0]

            # go through different memory settings
            for i in range(len(self.memory)):
                self.SKETCH[i].insert(toInsert, packet.isHigh, 1, self.env.now)
            self.demux.put(packet)
            return

        # First, check whether the packet has 'Forced to Record' label
        if hasattr(packet, "forceRec") == True:
            rightId = self.global_flows[packet.flow_id].path[packet.forceRec]
            if rightId == self.element_id:
                if True:
                    MonitorData.flow_to_switches.setdefault(packet.flow_id, set())
                    MonitorData.flow_to_switches[packet.flow_id].add(
                        self.element_id
                    )

                    MonitorData.switches_to_flow.setdefault(self.element_id, set())
                    MonitorData.switches_to_flow[self.element_id].add(
                        packet.flow_id
                    )
                    toInsert = packet.flow_id

                    # update the ground truth
                    if toInsert in self.segment_ground_truth:
                        # self.hash_table = max_dict_update(self.hash_table, sum_result, ts_result, int_result)
                        now = self.env.now
                        self.segment_ground_truth[toInsert][0] += 1
                        self.segment_ground_truth[toInsert][2] = max(
                            self.segment_ground_truth[toInsert][2],
                            now - self.segment_ground_truth[toInsert][1],
                        )
                        self.segment_ground_truth[toInsert][1] = now
                    else:
                        self.segment_ground_truth[toInsert] = [1, self.env.now, 0]

                    src = self.global_flows[packet.flow_id].path[0]
                    dst = self.global_flows[packet.flow_id].path[6]

                    # go through different memory settings
                    for i in range(len(self.memory)):
                        self.SKETCH[i].insert(toInsert, packet.isHigh, 1, self.env.now)

                    delattr(packet, "forceRec")
        # the normal hashing process
        else:
            src = self.global_flows[packet.flow_id].path[0]
            # print(self.global_flows[packet.flow_id].path)
            dst = self.global_flows[packet.flow_id].path[6]
            # cur = self.element_id
            # print(src, dst, cur)
            srcPod = self.topo.nodes[src]["pod"]
            dstPod = self.topo.nodes[dst]["pod"]
            # curPod = self.topo.nodes[self.element_id]['pod']
            # print(src, srcPod, dst, dstPod)
            if self.is_dht:
                selectedIndex = self.dht[(srcPod, dstPod)].hash(packet.flow_id) + 1
            else:
                selectedIndex = self.dht(packet.flow_id) + 1
            # selected switch
            selected = self.global_flows[packet.flow_id].path[selectedIndex]
            if selected != self.element_id:
                pass
            else:
                ifRecord = (
                    mmh3.hash(str(packet.flow_id), 134567, signed=False) % 1000000
                ) / 1000000 < self.valid

                if ifRecord == True:
                    MonitorData.flow_to_switches.setdefault(packet.flow_id, set())
                    MonitorData.flow_to_switches[packet.flow_id].add(
                        self.element_id
                    )

                    MonitorData.switches_to_flow.setdefault(self.element_id, set())
                    MonitorData.switches_to_flow[self.element_id].add(
                        packet.flow_id
                    )
                    toInsert = packet.flow_id

                    if toInsert in self.segment_ground_truth:
                        # self.hash_table = max_dict_update(self.hash_table, sum_result, ts_result, int_result)
                        now = self.env.now
                        self.segment_ground_truth[toInsert][0] += 1
                        self.segment_ground_truth[toInsert][2] = max(
                            self.segment_ground_truth[toInsert][2],
                            now - self.segment_ground_truth[toInsert][1],
                        )
                        self.segment_ground_truth[toInsert][1] = now
                    else:
                        self.segment_ground_truth[toInsert] = [1, self.env.now, 0]

                    # go through different memory settings
                    for i in range(len(self.memory)):
                        self.SKETCH[i].insert(toInsert, packet.isHigh, 1, self.env.now)

                else:
                    if selectedIndex == 1:
                        abort()
                        # packet.forceRec = 2
                    elif selectedIndex == 2:
                        packet.forceRec = 5
                    elif selectedIndex == 3:
                        # abort()
                        packet.forceRec = 5
                    elif selectedIndex == 4:
                        packet.forceRec = 5
                    elif selectedIndex == 5:
                        abort()

        # don't forget to put
        self.demux.put(packet)

    # phase1 to collect data
    def flip_phase1(self):
        # print(self.element_id, "this segment", self.segment_ground_truth)
        self.ground_truth = merge_dict_with_max(
            self.ground_truth, self.segment_ground_truth
        )
        self.ground_truth_segment.append(copy.deepcopy(self.segment_ground_truth))
        self.segment_ground_truth = defaultdict(list)

        for i in range(len(self.memory)):
            self.SKETCH[i].change_segment()

    # phase2 to reset sketch
    def flip_phase2(self, memory_ratio):
        for i in range(len(self.memory)):
            self.SKETCH[i].reset_sketch(
                int(self.memory[i] * memory_ratio),
                len(MonitorData.sampled_high_flow_set),
                len(MonitorData.unsampled_flow_set),
            )
