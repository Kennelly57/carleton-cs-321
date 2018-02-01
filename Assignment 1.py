import random
import numpy
def main():

    def initialize_rooms():
        list_of_rooms = []
        list_of_temp = []
        list_of_humidity = []
        for offices in range(12):
            room_enviro = []
            temp = 65 + random.randint(0,10)
            humidity = 45 + random.randint(0,10)

            room_enviro.append(temp)
            list_of_temp.append(temp)
            room_enviro.append(humidity)
            list_of_humidity.append(humidity)

            list_of_rooms.append(room_enviro)

        room_attributes = [list_of_rooms,list_of_temp,list_of_humidity]
        return room_attributes



    office = initialize_rooms()
    environment = office[0]
    temperature = office[1]
    humidity = office[2]

    print(environment)
    print(temperature)
    print(humidity)

    print(numpy.mean(temperature))
    print(numpy.std(temperature))

    print(numpy.mean(humidity))
    print(numpy.std(humidity))


main()