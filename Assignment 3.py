import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self):
        self.xCords = []
        self.yCords = []
        self.xCent = 0
        self.yCent = 0

    def getXCords(self):
        return self.xCords

    def getYCords(self):
        return self.yCords

    def addValue(self,xval,yval):
        self.xCords.append(xval)
        self.yCords.append(yval)

    def getXCent(self):
        xMean = np.mean(self.xCords)
        return float(xMean)

    def getYCent(self):
        yMean = np.mean(self.yCords)
        return float(yMean)

    def setXCent(self,xval):
        self.xCent = xval

    def setYCent(self,yval):
        self.yCent = yval

    def update(self):
        self.setXCent(self.getXCent())
        self.setYCent(self.getYCent())

    def remove(self,index):
        del self.xCords[index]
        del self.yCords[index]



def cleanData():
    df = pd.read_csv('HW3_Data.txt', sep=" ")
    data = []
    for key, value in df.iterrows():
        data.append(value[0].split())

    return data


def initializeCluster(data,k):
    clusterList = []
    for i in range(k):
        clusterList.append(Cluster())
        clusterList[i].addValue(float(data[i][1]), float(data[i][2]))
        clusterList[i].update()

    return clusterList

def reshuffle(clusterList):
    xCentList = []
    yCentList = []
    for i in range(len(clusterList)):
        xCentList.append(clusterList[i].getXCent())
        yCentList.append(clusterList[i].getYCent())

    for i in range(len(clusterList)):
        to_remove_inds = []
        xCentroid = xCentList[i]
        yCentroid = yCentList[i]

        xCords = clusterList[i].getXCords()
        yCords = clusterList[i].getYCords()

        for j in range(len(clusterList[i].getXCords())):
            currentX = xCords[j]
            currentY = yCords[j]

            currentDistance = EucliDistance(xCentroid,currentX,yCentroid,currentY)

            tempMinDistance = 10000
            tempMinIndex = -1
            for k in range(len(clusterList)):
                tempDistance = EucliDistance(currentX, clusterList[k].getXCent(),currentY,clusterList[k].getYCent())
                if tempDistance < tempMinDistance:
                    tempMinDistance = tempDistance
                    tempMinIndex = k

            if tempMinDistance < currentDistance:
                clusterList[tempMinIndex].addValue(currentX,currentY)
                to_remove_inds.append(j)
                #TODO delete currentX and currentY from clusterlist[i]

        to_remove_inds.reverse()
        for ind in to_remove_inds:
            clusterList[i].remove(ind)

    # print(xCentList)
    # print(yCentList)


def EucliDistance(x1,x2,y1,y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**1/2


def kMeansCluster(data,k):
    clusterList = initializeCluster(data,k)

    for i in range(k,len(data)):
        current_xCord = float(data[i][1])
        current_yCord = float(data[i][2])

        minIndex = -1
        minDistance = 99999
        for j in range(k):
            currentCluster = clusterList[j]
            #temp_distance = ((currentCluster.getXCent() - current_xCord)**2 + (currentCluster.getYCent() - current_yCord)**2)**1/2
            temp_distance = EucliDistance(currentCluster.getXCent(),current_xCord,currentCluster.getYCent(),current_yCord)

            if temp_distance < minDistance:
                minDistance = temp_distance
                minIndex = j

        clusterList[minIndex].addValue(current_xCord, current_yCord)
        clusterList[minIndex].update


    reshuffle(clusterList)

    return clusterList

def calcError(clusterList):
    sum_of_error = 0
    for cluster in clusterList:
        xCentroid = cluster.getXCent()
        yCentroid = cluster.getYCent()

        for xval,yval in zip(cluster.getXCords(),cluster.getYCords()):
            temp_val = EucliDistance(xval,xCentroid,yval,yCentroid)
            sum_of_error += temp_val

    return sum_of_error


def elbowMethod(data):
    error_list = []
    for i in range(1,10):
        clustered = kMeansCluster(data, i)
        error = calcError(clustered)
        error_list.append(error)

    elbow = plt.figure()
    plt.plot([1,2,3,4,5,6,7,8,9],error_list)
    plt.axis([1, 9, 0, 7000000])
    plt.ylabel("Sum of Squared Error")
    plt.xlabel("Number of Clusters")
    plt.show()

    for i in range(1,len(error_list)):
        param = (error_list[i-1] - error_list[i])/error_list[i-1]
        if param < 0.17:
            return i

    return i






def print_graph(clusterList,xlab,ylab,filename):

    f = plt.figure()
    color_list = ['C0','C1','C2','C4','C5','C6','C7','C8','C9']
    area = np.pi * 40

    for i in range(len(clusterList)):
        plt.scatter(clusterList[i].getXCords(), clusterList[i].getYCords(), c=color_list[i])
        plt.scatter(clusterList[i].getXCent(), clusterList[i].getYCent(), c='C3', s=area)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    f.savefig(filename, bbox_inches='tight')

def main():

    data = cleanData()
    k = elbowMethod(data)
    clustered = kMeansCluster(data,k)
    print_graph(clustered,"a","b","test")

main()





