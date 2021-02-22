#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     mm1k.py
# @author   Mengfan Li  <dreamfan916@gmail.com>
# @date     2020-11-26
#
# @brief    Simulate M/M/1/K queueing system
#
# @remarks  Copyright (C) 2020 Mengfan Li All rights reserved.
#
# @remarks  PyCharm 2020.2.3 (Community Edition)
#           Python 3.8
#           simpy 4.0.1
#           You must not remove this notice, or any other, from this software.
#


import argparse
import numpy as np
import random
import simpy
import matplotlib.pyplot as plt

# define the queue number and the block probability list to save the probability
queue_number = 0
block_number = 0
blocking_probabilities_list = [[], [], []]


def source(env, mean_ia_time, mean_srv_time, server, num_packets, system_capacity,
           block_times, trace):
    """Generates packets with exponential interarrival time."""
    for i in range(num_packets):
        ia_time = random.expovariate(1.0 / mean_ia_time)
        srv_time = random.expovariate(1.0 / mean_srv_time)
        pkt = packet(env, 'Packet-%d' % i, server, srv_time, system_capacity,
                     block_times, trace)
        env.process(pkt)
        yield env.timeout(ia_time)


def packet(env, name, server, service_time, system_capacity, block_times, trace):
    """Requests a server, is served for a given service_time, and leaves the server."""
    arrv_time = env.now

    global queue_number, block_number

    # queue_number will automatically increment by default every time one packet is transfered
    queue_number += 1

    """
    Part K of MM1K, K is System capacity,    

    When the number of services in the queue plus the number of waits is less than K, 
    can also enter the queue;
    if equal to K, the queue will be blocked; 
    if greater than K, the queue will no longer be allowed to enter

    """
    if queue_number <= system_capacity:
        if trace:
            print('t=%.4Es: %s arrived, ' % (arrv_time, name))

        with server.request() as request:
            yield request

            yield env.timeout(service_time)

            """"
            If queue_number == system_capacity,the queue is blocked at this time. 

            it is MM1 model, the queue can not continue until the first service is completed. 
            the block_time = service_time 

            """
            if queue_number == system_capacity:
                block_time = service_time
                block_times.append(block_time)

            # When the service is complete, the queue releases one
            queue_number -= 1

            if trace:
                print('t=%.4Es: %s served for %.4Es' % (env.now, name, service_time))

    #  queue_number > system_capacity, discard the waiting element directly outside the queue
    else:
        queue_number -= 1
        block_number += 1

        if trace:
            print('t=%.4Es: %s loss_packet_num' % (env.now, name))


def run_simulation(mean_ia_time, mean_srv_time, num_packets, system_capacity,
                   random_seed, trace=True):
    """
    Runs a simulation and returns statistics
    """

    print('M/M/1/K queue\n')
    random.seed(random_seed)
    env = simpy.Environment()

    # start processes and run
    block_times = []

    server = simpy.Resource(env, capacity=1)
    env.process(source(env, mean_ia_time, mean_srv_time, server, num_packets, system_capacity,
                       block_times, trace))
    env.run()

    # return statistics
    return block_number / num_packets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-A",
        "--arrival_rate",
        help="Arrival rate [s]; default is 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "-S",
        "--mean_srv_time",
        help="mean packet service time [s]; default is 0.1",
        default=1 / 100,
        type=float)
    parser.add_argument(
        "-N",
        "--num_packets",
        help="number of packets to generate; default is 1000",
        default=1000,
        type=int)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="seed for random number generation; default is 1234",
        default=1234,
        type=int)
    parser.add_argument(
        "-K",
        "--system_capacity",
        help="number of System capacity; default is 1",
        default=1,
        type=int)

    parser.add_argument('--trace', dest='trace', action='store_true')
    parser.add_argument('--no-trace', dest='trace', action='store_false')
    parser.set_defaults(trace=True)
    args = parser.parse_args()

    # set variables using command-line arguments
    arrival_rate = args.arrival_rate
    mean_srv_time = args.mean_srv_time
    num_packets = args.num_packets
    random_seed = args.random_seed
    system_capacity = args.system_capacity
    trace = args.trace

    # run a simulation
    system_capacity_list = [10, 20, 50]
    for i in range(len(system_capacity_list)):
        for arrival_rate in np.linspace(5, 95, 19):
            blocking_probabilities = run_simulation(np.reciprocal(arrival_rate), mean_srv_time,
                                                    num_packets, system_capacity_list[i], random_seed, trace=True)

            blocking_probabilities_list[i].append((blocking_probabilities))
            block_number = 0



"""
Paint 
"""

system_capacity_list = [10, 20, 50]
temp_array = [[], [], []]
counter = 0

for N in system_capacity_list:
    for Rho in np.linspace(0.05, 0.95, 19):
        P = ((np.power(Rho, N)) - (np.power(Rho, N + 1))) / (1 - (np.power(Rho, N + 1)))

        temp_array[counter].append(P)
    counter += 1

plt.figure()
x_axis = np.linspace(5, 95, 19)
plt.plot(x_axis, temp_array[0], color='blue', label='K = 10')
plt.plot(x_axis, temp_array[1], color='red', label='K = 20')
plt.plot(x_axis, temp_array[2], color='green', label='K = 50')
plt.scatter(x_axis,blocking_probabilities_list[0], marker='o')
plt.scatter(x_axis,blocking_probabilities_list[1], marker='*')
plt.scatter(x_axis,blocking_probabilities_list[2], marker='^')
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
plt.title('Formula and simulation result')
plt.xlabel("arrival rate")
plt.ylabel('block probability')
plt.legend()
plt.show()

temp_array = np.array(temp_array)
blocking_probabilities_list = np.array(blocking_probabilities_list)
gap = abs(temp_array - blocking_probabilities_list)

plt.figure()
x_axis = np.linspace(5, 95, 19)
plt.plot(x_axis, gap[0], color='blue', label='K = 10')
plt.plot(x_axis, gap[1], color='red', label='K = 20')
plt.plot(x_axis, gap[2], color='green', label='K = 50')
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
plt.title('The gap between simulation and formula')
plt.xlabel("arrival rate")
plt.ylabel('gap')
plt.show()





