CC = g++
CC_FLAGS = -std=c++11 -O2 -Wall `pkg-config --cflags --libs opencv4` 

all: stat_1

stat_1: Stat_1.cpp
	$(CC) $(CC_FLAGS) Stat_1.cpp -o stat_1

clean: 
	rm stat_1
