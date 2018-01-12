import random
def main():


    def initialize_rooms():
        list_of_rooms = []
        for offices in range(12):
            room_enviro = []
            temp = 65 + random.randint(0,10)
            room_enviro.append(temp)
            humidity = 45 + random.randint(0,10)
            room_enviro.append(humidity)

            list_of_rooms.append(room_enviro)

        return list_of_rooms

    office = initialize_rooms()

    print(office)


main()