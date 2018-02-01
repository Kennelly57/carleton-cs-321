import networkx as nx
import random
import sys, os

import matplotlib.pyplot as plt


class Room:
    def __init__(self):
       self.temperature = 65 + random.randint(0, 10)
       self.humidity = 45 + random.randint(0, 10)
       self.temp_diff = 0
       self.humidity_diff = 0

    def get_temp(self):
        return self.temperature

    def get_humidity(self):
        return self.humidity

    def set_temp(self,temp):
        self.temperature = temp

    def set_humidity(self,hum):
        self.humidity = hum


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
        office = Room()
        rooms.append(office)

    return rooms

def depth_first(Graph, start):
   return list(nx.dfs_preorder_nodes(Graph, source=start))

def find_outlier(search_list, room_list):
    max_temp_diff = 0
    max_humidity_diff = 0
    max_combined_diff = 0

    temp_ind = 0
    humidity_ind = 0
    combined_ind = 0

    for room_index in search_list:
        if abs(room_list[room_index].get_temp() - 72) > max_temp_diff:
            max_temp_diff = abs(room_list[room_index].get_temp() - 72)
            temp_ind = room_index

        if abs(room_list[room_index].get_humidity() - 47) >  max_humidity_diff:
            max_humidity_diff = abs(room_list[room_index].get_humidity() - 47)
            humidity_ind = room_index

        if abs(room_list[room_index].get_temp() - 72) + abs(room_list[room_index].get_humidity() - 47) > max_combined_diff:
            max_combined_diff = abs(room_list[room_index].get_temp() - 72) + abs(room_list[room_index].get_humidity() - 47)
            combined_ind = room_index

    if 47 < get_humidity_mean(room_list) < 48 and get_humidity_std(room_list) < 1.75:
        return temp_ind

    if 72 < get_temp_mean(room_list) < 73 and get_temp_std(room_list) < 1.5:
        return humidity_ind

    return combined_ind


def heat_miser(Graph,room_list):
    starting_room = random.randint(0, 11)

    meanTemp = get_temp_mean(room_list)
    stdTemp = get_temp_std(room_list)
    meanHum = get_humidity_mean(room_list)
    stdHum = get_humidity_std(room_list)

    trials = 0
    while (72 > meanTemp) or (meanTemp > 73) or (stdTemp > 1.5) or (47 > meanHum) or (meanHum > 47.99) or (stdHum > 1.75):
        search_list = depth_first(Graph, starting_room)
        starting_room = find_outlier(search_list, room_list)
        room_list[starting_room].set_temp(72.5)
        room_list[starting_room].set_humidity(47.5)

        meanTemp = get_temp_mean(room_list)
        stdTemp = get_temp_std(room_list)
        meanHum = get_humidity_mean(room_list)
        stdHum = get_humidity_std(room_list)

        trials += 1



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
    G = initialize_office()
    room_list = initialize_rooms()

    heat_miser(G, room_list)

    trial_list = []
    # for i in range(100):
    #
    #     room_list = initialize_rooms()
    #     trial_list.append(heat_miser(G, room_list))
    #
    # print_graph(trial_list, "Trial Number", "Number of Rooms Entered", "fig1.pdf")
    #
    # print(numpy.mean(trial_list))

    heuristics_file = open('HeatMiserHeuristic.txt', 'r')
    heuristicList = []
    for line in heuristics_file:
        currentSplit = line.split()
        heuristicList.append(currentSplit)

    print(heuristicList[1])
main()