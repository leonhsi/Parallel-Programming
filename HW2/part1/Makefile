CXX := gcc
CXXFLAGS := -I./common -O3 -Wall -pthread

all: pi.out

pi.o: pi.c
	$(CXX) $(CXXFLAGS) -c pi.c

pi.out: pi.o
	gcc -I./common -pthread pi.o -o pi.out

clean:
	rm -f *.o *.s pi.out *~
