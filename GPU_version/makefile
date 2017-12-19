
CC=/usr/local/cuda/bin/nvcc -std=c++11
INCLUDE=-I/usr/local/cuda/include
HEADER=-I/usr/local/cuda/samples/common/inc
CFLAGS = -Wall -O2
SOURCES = dns.cu
LIBS = -lGL -lGLU -lglut
EXECUTABLES = dns

all:
	$(CC) $(HEADER) $(CFLAS) $(SOURCES) $(LIBS) -o $(EXECUTABLES)
	./$(EXECUTABLES)
clean:
        
	rm -rf dns.o
