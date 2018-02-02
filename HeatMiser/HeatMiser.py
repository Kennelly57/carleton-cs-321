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

    def getTemp(self):
        return self.temperature

    def getHumidity(self):
        return self.humidity

    def setTemp(self,temp):
        self.temperature = temp

    def setHumidity(self,hum):
        self.humidity = hum

    def addEdge(self, room, edge_weight):
        self.edges.append(room)
        self.edge_weights.append(edge_weight)

    def getEdges(self):
        return self.edges

    def getEdgeWeights(self):
        return self.edge_weights

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name


def getTempMean(room_list):
    temp_mean = 0

    for room in room_list:
        temp_mean += room.get_temp()

    temp_mean /= 12
    return temp_mean


def getHumidityMean(room_list):
    humidity_mean = 0

    for room in room_list:
        humidity_mean += room.get_humidity()

    humidity_mean /= 12
    return humidity_mean


def getTempSTD(room_list):
    temp_mean = getTempMean(room_list)

    running_sum = 0

    for room in room_list:
        running_sum += (room.get_temp() - temp_mean)**2

    running_sum /= 12
    running_sum = running_sum ** (1/2)

    return running_sum


def getHumiditySTD(room_list):
    humidity_mean = getHumidityMean(room_list)

    running_sum = 0

    for room in room_list:
        running_sum += (room.get_humidity() - humidity_mean)**2

    running_sum /= 12
    running_sum = running_sum ** (1/2)

    return running_sum

def initializeOffice():
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
    rooms.append(Room(0))
    for i in range(1, 13):
        office = Room(i)
        rooms.append(office)

    rooms[1].addEdge(rooms[2], 13)
    rooms[1].addEdge(rooms[3], 15)
    rooms[3].addEdge(rooms[7], 23)
    rooms[2].addEdge(rooms[4], 7)
    rooms[6].addEdge(rooms[7], 9)
    rooms[4].addEdge(rooms[6], 10)
    rooms[4].addEdge(rooms[5], 6)
    rooms[6].addEdge(rooms[9], 16)
    rooms[8].addEdge(rooms[9], 5)
    rooms[7].addEdge(rooms[10], 17)
    rooms[9].addEdge(rooms[10], 8)
    rooms[10].addEdge(rooms[11], 2)
    rooms[11].addEdge(rooms[12], 19)

    return rooms


def depthFirstSearch(Graph, start):
    return list(nx.dfs_preorder_nodes(Graph, source=start))


def findOutlier(room_list):
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

    if 47 < getHumidityMean(room_list) < 48 and getHumiditySTD(room_list) < 1.75:
        return temp_ind

    if 72 < getTempMean(room_list) < 73 and getTempSTD(room_list) < 1.5:
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

def a_star_search(room_list_graph, starting_room, goal_room, straightLineDistance):
    edge_list = room_list_graph[starting_room].get_edges()
    edge_weights = room_list_graph[starting_room].get_edge_weights()
    current_room = starting_room

    while current_room is not goal_room:
        cumulativeDistance = 0
        lowestEdgeWeight = 10000
        lowestEdgeWeightIndex = "NAN"
        for i in range(edge_weights.length()):
            if edge_weights[i] < lowestEdgeWeight:
                lowestEdgeWeight = edge_weights[i]
                lowestEdgeWeightIndex = i


def findStraightLineDistance(startingRoom, goalRoom, heuristics):
    baseRange = startingRoom.getName()*11
    for i in range(baseRange - 10, baseRange):
        if int(heuristics[i][1]) == goalRoom.getName():
            return heuristics[i][2]
    return -1

def heat_miser_part2(room_list, heuristics):
    starting_room = random.randint(0, 11)

    meanTemp = getTempMean(room_list)
    stdTemp = getTempSTD(room_list)
    meanHum = getHumidityMean(room_list)
    stdHum = getHumiditySTD(room_list)

    trials = 0
    while (72 > meanTemp) or (meanTemp > 73) or (stdTemp > 1.5) or (47 > meanHum) or (meanHum > 47.99) or (stdHum > 1.75):
        goal_room = findOutlier(room_list)
        straightLineDistance = findStraightLineDistance(starting_room, goal_room, heuristics)
        cheapest_path = a_star_search(room_list, starting_room, goal_room, straightLineDistance)
        room_list[starting_room].set_temp(72.5)
        room_list[starting_room].set_humidity(47.5)

        meanTemp = getTempMean(room_list)
        stdTemp = getTempSTD(room_list)
        meanHum = getHumidityMean(room_list)
        stdHum = getHumiditySTD(room_list)

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
    # print(heat_miser_part2(room_list,heuristicList))
    print(findStraightLineDistance(room_list[2], room_list[4], heuristicList))
    # print(room_list[0].get_edges()[0].getName())
    # print(room_list[0].get_edges()[1].getName())
    # print(room_list[0].get_edge_weights())
main()