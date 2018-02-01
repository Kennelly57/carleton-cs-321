import networkx as nx
import random
import sys, os
import numpy

import matplotlib.pyplot as plt


class Room:
    def __init__(self, name):
        self.name = name
        self.temperature = 65 + random.randint(0, 10)
        self.humidity = 45 + random.randint(0, 10)
        self.temp_diff = 0
        self.humidity_diff = 0
        self.edges = []
        self.edge_weights = []

    def get_temp(self):
        return self.temperature

    def get_humidity(self):
        return self.humidity

    def set_temp(self,temp):
        self.temperature = temp

    def set_humidity(self,hum):
        self.humidity = hum

    def add_edge(self, room, edge_weight):
        self.edges.append(room)
        self.edge_weights.append(edge_weight)

    def get_edges(self):
        return self.edges

    def get_edge_weights(self):
        return self.edge_weights

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name


def get_temp_mean(room_list):
    temp_mean = 0

    for room in room_list:
        temp_mean += room.get_temp()

    temp_mean /= 12
    return temp_mean


def get_humidity_mean(room_list):
    humidity_mean = 0

    for room in room_list:
        humidity_mean += room.get_humidity()

    humidity_mean /= 12
    return humidity_mean


def get_temp_std(room_list):
    temp_mean = get_temp_mean(room_list)

    running_sum = 0

    for room in room_list:
        running_sum += (room.get_temp() - temp_mean)**2

    running_sum /= 12
    running_sum = running_sum ** (1/2)

    return running_sum


def get_humidity_std(room_list):
    humidity_mean = get_humidity_mean(room_list)

    running_sum = 0

    for room in room_list:
        running_sum += (room.get_humidity() - humidity_mean)**2

    running_sum /= 12
    running_sum = running_sum ** (1/2)

    return running_sum

def initialize_office():
    G = nx.Graph()
    G.add_edge(0, 1, weight=13)
    G.add_edge(0, 2, weight=15)
    G.add_edge(2, 6, weight=23)
    G.add_edge(1, 3, weight=7)
    G.add_edge(5, 6, weight=9)
    G.add_edge(3, 5, weight=10)
    G.add_edge(3, 4, weight=6)
    G.add_edge(5, 8, weight=16)
    G.add_edge(7, 8, weight=5)
    G.add_edge(6, 9, weight=17)
    G.add_edge(8, 9, weight=8)
    G.add_edge(9, 10, weight=2)
    G.add_edge(10, 11, weight=19)

    return G


def initialize_rooms():
    rooms = []
    for i in range(12):
        office = Room(i)
        rooms.append(office)

    rooms[0].add_edge(rooms[1], 13)
    rooms[0].add_edge(rooms[2], 15)
    rooms[2].add_edge(rooms[6], 23)
    rooms[1].add_edge(rooms[3], 7)
    rooms[5].add_edge(rooms[6], 9)
    rooms[3].add_edge(rooms[5], 10)
    rooms[3].add_edge(rooms[4], 6)
    rooms[5].add_edge(rooms[8], 16)
    rooms[7].add_edge(rooms[8], 5)
    rooms[6].add_edge(rooms[9], 17)
    rooms[8].add_edge(rooms[9], 8)
    rooms[9].add_edge(rooms[10], 2)
    rooms[10].add_edge(rooms[11], 19)

    return rooms


def depth_first_search(Graph, start):
    return list(nx.dfs_preorder_nodes(Graph, source=start))


def find_outlier(room_list):
    max_temp_diff = 0
    max_humidity_diff = 0
    max_combined_diff = 0

    temp_ind = 0
    humidity_ind = 0
    combined_ind = 0

    for room_index in room_list:
        current_temp_diff = abs(room_list[room_index].get_temp() - 72)
        current_humidity_diff = abs(room_list[room_index].get_humidity() - 47)

        if current_temp_diff > max_temp_diff:
            max_temp_diff = abs(room_list[room_index].get_temp() - 72)
            temp_ind = room_index

        if current_humidity_diff > max_humidity_diff:
            max_humidity_diff = abs(room_list[room_index].get_humidity() - 47)
            humidity_ind = room_index

        if current_temp_diff + current_humidity_diff > max_combined_diff:
            max_combined_diff = abs(room_list[room_index].get_temp() - 72) + abs(room_list[room_index].get_humidity() - 47)
            combined_ind = room_index

    if 47 < get_humidity_mean(room_list) < 48 and get_humidity_std(room_list) < 1.75:
        return temp_ind

    if 72 < get_temp_mean(room_list) < 73 and get_temp_std(room_list) < 1.5:
        return humidity_ind

    return combined_ind


# def heat_miser_part1(graph, room_list):
#     starting_room = random.randint(0, 11)
#
#     meanTemp = get_temp_mean(room_list)
#     stdTemp = get_temp_std(room_list)
#     meanHum = get_humidity_mean(room_list)
#     stdHum = get_humidity_std(room_list)
#
#     trials = 0
#     while (72 > meanTemp) or (meanTemp > 73) or (stdTemp > 1.5) or (47 > meanHum) or (meanHum > 47.99) or (stdHum > 1.75):
#         search_list = depth_first_search(graph, starting_room)
#         starting_room = find_outlier(search_list, room_list)
#         room_list[starting_room].set_temp(72.5)
#         room_list[starting_room].set_humidity(47.5)
#         trials += starting_room
#
#         meanTemp = get_temp_mean(room_list)
#         stdTemp = get_temp_std(room_list)
#         meanHum = get_humidity_mean(room_list)
#         stdHum = get_humidity_std(room_list)
#
#     return trials

def a_star_search(room_list_graph, starting_room, goal_room):
    edge_list = room_list_graph[starting_room].get_edges()
    edge_weights = room_list_graph[starting_room].get_edge_weights()
    

def heat_miser_part2(graph, room_list):
    starting_room = random.randint(0, 11)

    meanTemp = get_temp_mean(room_list)
    stdTemp = get_temp_std(room_list)
    meanHum = get_humidity_mean(room_list)
    stdHum = get_humidity_std(room_list)

    trials = 0
    while (72 > meanTemp) or (meanTemp > 73) or (stdTemp > 1.5) or (47 > meanHum) or (meanHum > 47.99) or (stdHum > 1.75):
        goal_room = find_outlier(room_list)
        cheapest_path = a_star_search(room_list, starting_room, goal_room)
        room_list[starting_room].set_temp(72.5)
        room_list[starting_room].set_humidity(47.5)

        meanTemp = get_temp_mean(room_list)
        stdTemp = get_temp_std(room_list)
        meanHum = get_humidity_mean(room_list)
        stdHum = get_humidity_std(room_list)

        # Need to add the length of the path, not 1
        trials += 1
        starting_room = goal_room

    return trials

def print_graph(list_of_trials,xlab,ylab,filename):
    list_of_trials.sort()

    f = plt.figure()
    plt.plot(list_of_trials)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    f.savefig(filename, bbox_inches='tight')

def main():
    #G = initialize_office()
    room_list = initialize_rooms()

    # trial_list = []
    # for i in range(100):
    #     room_list = initialize_rooms()
    #     trial_list.append(heat_miser_part1(G, room_list))
    #
    # print_graph(trial_list, "Trial Number", "Number of Rooms Entered", "fig1.pdf")
    #
    # print(numpy.mean(trial_list))

    heuristics_file = open('HeatMiserHeuristic.txt', 'r')
    heuristicList = []
    for line in heuristics_file:
        currentSplit = line.split()
        heuristicList.append(currentSplit)

    # print(room_list[0].get_edges()[0].get_name())
    # print(room_list[0].get_edges()[1].get_name())
    # print(room_list[0].get_edge_weights())
main()