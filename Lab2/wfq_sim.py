#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     wfq_sim.py
# @author   Mengfan Li  <dreamfan916@gmail.com>
# @date     2020-12-27
#
# @brief    Simulate WFQ/FQ
#
# @remarks  Copyright (C) 2020 Mengfan Li All rights reserved.
#
# @remarks  PyCharm 2020.3 (Professional Edition)
#           Python 3.8
#           simpy 4.0.1
#           You must not remove this notice, or any other, from this software.
#


import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import simpy
from collections import deque
from math import ceil
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


class Packet(object):
    """
    Parameters:
    - flow_id: flow ID
    - ctime: creation time
    - length: length in bytes
    - v_finish: virtual finish time
    """

    def __init__(self, flow_id, ctime, length):
        self.flow_id = flow_id
        self.ctime = ctime
        self.length = length



class PacketGenerator(object):
    """Generate packets with given inter-arrival time and size distributions,
       and set the output port of the receiving entity.

    Parameters:
    - env: simpy.Environment
    - flow_id: flow ID
    - ia_dist: packet interarrival time distribution function
    - lng_dist: packet length distribution function
    """

    def __init__(self, env, flow_id, ia_dist, lng_dist, trace=False):
        self.env = env
        self.flow_id = flow_id
        self.ia_dist = ia_dist
        self.lng_dist = lng_dist
        self.trace = trace
        self.out = None
        self.pkts_sent = 0
        self.on = True
        self.action = env.process(self.run())  # start the run process when an instance is created

    def run(self):
        while True:
            yield self.env.timeout(self.ia_dist())
            self.pkts_sent += 1

            # Instantiate an object
            p = Packet(flow_id=self.flow_id, ctime=self.env.now, length=self.lng_dist())

            if self.trace:
                print(f"{p.ctime:.4E}s: packet generated with flow ID={p.flow_id:d} and length={p.length:d} [B]")
            self.out.put(p)


class RoundRobinScheduler(object):
    """Receive, process, and send out packets.

    Parameters:
    - env : simpy.Environment
    """

    def __init__(self, env, num_inputs, output_rate, trace=False):
        self.env = env
        self.num_inputs = num_inputs
        self.output_rate = output_rate
        self.trace = trace
        self.pkt_container = simpy.Container(env)  # as a counter for all packets
        self.flow_queues = []
        for i in range(num_inputs):
            self.flow_queues.append(deque())
        self.out = None
        self.last_flow_id = 0  # flow ID served during the last round


        self.vtime = 0              # virtual time
        self.last_update = 0.0      # last time when virtual tim
        self.active_set = set()     # set of non-empty queues
        self.ftimes = [0, 0, 0, 0]
        self.finish_time_list = [float('inf'), float('inf'), float('inf'), float('inf')]
        # self.weights = [1, 2, 3, 4]  # WFQ
        self.weights = [1,1,1,1]   # FQ


        self.action = env.process(self.run())  # start the run process when an instance is created

    """
    Send
    """
    def run(self):
        while True:
            yield self.pkt_container.get(1)

            """
            # select queue
            """
            for flow_id in range(self.num_inputs):
                if len(self.flow_queues[flow_id]) > 0:
                    pkt = self.flow_queues[flow_id][0]  # get from the left of a deque
                    self.finish_time_list[flow_id] = pkt.v_finish

            flow_id = self.finish_time_list.index(min(self.finish_time_list))

            pkt = self.flow_queues[flow_id].popleft()  # Propose a package to send

            if len(self.flow_queues[flow_id]) == 0:
                self.active_set.remove(flow_id)

            # reset the virtual time
            if not self.active_set:
                self.vtime = 0
                self.ftimes = [0, 0, 0, 0]

            for i in range(4):
                self.finish_time_list[i] = float('inf')

            # send packet
            yield self.env.timeout(pkt.length / self.output_rate)  # both based on bytes
            self.out.put(pkt)

    """
    receive
    """
    def put(self, pkt):
        flow_id = pkt.flow_id  # Receive the generated package

        #update virtual time
        now = self.env.now
        sum = 0.0
        if self.active_set:
            for flow_id_temp in self.active_set:
                sum += self.weights[flow_id_temp]

            self.vtime += (now - self.last_update) * self.output_rate / sum
        self.last_update = now

        """
        WFQ need to divide self.weights[flow_id]
        FQ because the weight are all 1, so we don't need to divide weight
        """
        self.ftimes[flow_id] = max(self.vtime, self.ftimes[flow_id]) + pkt.length / self.weights[flow_id]
        pkt.v_finish = self.ftimes[flow_id]  # The virtual completion time of the current package
        self.active_set.add(flow_id)


        self.flow_queues[flow_id].append(pkt)  # The current packet enters the queue

        self.pkt_container.put(1)



class FlowDemux(object):
    """ Demultiplex packet flows based on flow_id.

    Parameters:
    - env : simpy.Environment
    - outs : list of output ports
    """

    def __init__(self, outs=None, default=None):
        self.outs = outs
        self.default = default
        self.pkts_rcvd = 0

    def put(self, pkt):
        self.pkts_rcvd += 1
        flow_id = pkt.flow_id
        if flow_id < len(self.outs):
            self.outs[flow_id].put(pkt)
        else:
            if self.default:
                self.default.put(pkt)


class PacketSink(object):
    """Receives packets and display delay information.

    Parameters:
    - env : simpy.Environment
    - flow_id: flow ID
    - mi: measurement interval in seconds
    - trace: Boolean
    """

    def __init__(self, env, flow_id, mi, trace=False):
        self.store = simpy.Store(env)
        self.env = env
        self.flow_id = flow_id
        self.mi = mi
        self.trace = trace
        self.bytes_rcvd = 0
        self.last_mt = 0  # last measurement time
        self.times = []
        self.throughputs = []
        self.action = env.process(self.run())  # start the run process when an instance is created

    def run(self):
        self.env.process(self.update_stats())
        while True:
            pkt = (yield self.store.get())
            now = self.env.now
            self.bytes_rcvd += pkt.length
            if self.trace:
                print(f"{now:.4E}s: packet received with flow ID={pkt.flow_id:d} and length={pkt.length:d} [B]")

    def update_stats(self):
        while True:
            yield self.env.timeout(self.mi)
            now = self.env.now
            self.times.extend([self.last_mt, now])
            throughput = self.bytes_rcvd / self.mi
            self.throughputs.extend([throughput] * 2)
            self.bytes_rcvd = 0
            self.last_mt = now
            if self.trace:
                print(f"{now:.4E}s: throughput[{self.flow_id:d}]={throughput:} [B/s]")

    def put(self, pkt):
        self.store.put(pkt)


def run_simulation(sim_time, random_seed, mi, trace=True):
    """Runs a simulation and returns statistics."""
    random.seed(random_seed)
    env = simpy.Environment()

    # simulation parameters per the Lab assignment 2 document
    num_flows = 4
    rate = 1000  # in B/s
    mean_pkt_ia_time = [1, 2, 4, 3]  # in seconds
    pkt_length = [i * 1000 for i in [2, 4, 8, 6]]  # in bytes

    # start processes and run
    router = RoundRobinScheduler(env=env, num_inputs=num_flows, output_rate=rate, trace=trace)
    generators = [None] * 4
    for i in range(4):
        generators[i] = PacketGenerator(
            env=env,
            ia_dist=lambda flow_id=i: random.expovariate(1.0 / mean_pkt_ia_time[flow_id]),
            lng_dist=lambda flow_id=i: pkt_length[flow_id],
            flow_id=i,
            trace=trace
        )
        generators[i].out = router
    sinks = [PacketSink(env=env, flow_id=i, mi=mi, trace=trace) for i in range(4)]
    demux = FlowDemux(sinks)
    router.out = demux
    env.run(until=sim_time)

    # post-process simulation results
    times = [None] * 4
    flow_throughputs = [None] * 4
    for i in range(4):
        times[i] = sinks[i].times
        flow_throughputs[i] = sinks[i].throughputs
    return times, flow_throughputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-T",
        "--sim_time",
        help="time to end the simulation in seconds; default is 3600.5",
        default=3600.5,
        type=float)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="seed for random number generation; default is 1234567",
        default=1234567,
        type=int)

    parser.add_argument('--trace', dest='trace', action='store_true')
    parser.add_argument('--no-trace', dest='trace', action='store_false')
    parser.set_defaults(trace=True)
    args = parser.parse_args()

    # set variables using command-line arguments
    sim_time = args.sim_time
    random_seed = args.random_seed
    trace = args.trace

    # obtain flow throughputs for two different measurement intervals
    mis = [60, 600]
    times = [None] * 2
    flow_throughputs = [None] * 2
    for i in range(2):
        times[i], flow_throughputs[i] = run_simulation(sim_time=sim_time,
                                                       random_seed=random_seed,
                                                       mi=mis[i],
                                                       trace=trace)


    # plot flow throughputs
    fig, axs = plt.subplots(2, 1, figsize=(11.7, 8.3), sharex=True, sharey=True)  # A4 landscape
    labels = ['A', 'B', 'C', 'D']
    ymin = 0
    ymax = ceil(max(np.array(flow_throughputs[0]).max(), np.array(flow_throughputs[1]).max()) / 100) * 100 + 50
    plts = [None] * 2
    for i in range(2):
        for j in range(4):
            axs[i].plot(times[i][j], flow_throughputs[i][j], label=labels[j])
            axs[i].set_xlim([0, 3650])  # to provide space for legend
            axs[i].set_ylim([ymin, ymax])
        axs[i].xaxis.set_major_locator(MultipleLocator(mis[i]))
        axs[i].xaxis.set_minor_locator(MultipleLocator(mis[i] / 2))
        axs[i].grid(which='minor', alpha=0.2)
        axs[i].grid(which='major', alpha=0.5)
        axs[i].legend(loc='center right')
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Flow Throughput [B/s]', fontsize=12, labelpad=15)
    fig.tight_layout()
    plt.title('Round-Robin Scheduler')
    plt.savefig('rr_scheduling_v2.pdf')
    plt.show()
    plt.close()
