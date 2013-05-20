#gcc -I /opt/local/include -L /opt/local/lib -lgsl -o myIzhik myIzhik.c

OBJS = myIzhik
SRCS = $(OBJS).c

#INCLUDES += -I/opt/local/include 

#INCLUDE_LIBS += -L/opt/local/lib
#LIBS += -lgsl #-lgslcblas 
LIBS += -lm 
# -ansi complains on linux since drand48 is deprecated (wtf?!?)
#CFLAGS += -ansi -pedantic

CFLAGS += -Wall -W -Wmissing-prototypes -Wstrict-prototypes -Wconversion -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -Wnested-externs -fshort-enums -fno-common 
# debug
#CFLAGS += -g -DDEBUG -Werror
# production
CFLAGS += -O3 

# openMP flags
OMPFLAGS += -ftree-vectorize -fopenmp -Werror 

#DEFINES += -D_GSL -Dinline=

#GCC = gcc
# for mac ports
GCC = /opt/local/bin/gcc-mp-4.3

OPTS = $(CFLAGS) $(INCLUDES) $(INCLUDE_LIBS) $(LIBS) $(DEFINES)

all: omp nonomp

clean: 
	rm $(OBJS)-omp $(OBJS)-nonomp

omp: $(SRCS)
	$(GCC) $(OPTS) $(OMPFLAGS) -o $(OBJS)-omp $(SRCS)

nonomp: $(SRCS)
	$(GCC) $(OPTS) -o $(OBJS)-nonomp $(SRCS) 
	
