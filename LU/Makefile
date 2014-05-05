TARGET = LU
OBJS = $(addsuffix .o, $(basename $(wildcard *.c)))

#CC := gcc
CC := scalasca -instrument gcc
CFLAGS := -O3
#CFLAGS := -g3
CFLAGS := $(CFLAGS) -Wall -W -Wmissing-prototypes -Wmissing-declarations -Wredundant-decls -Wdisabled-optimization
CFLAGS := $(CFLAGS) -Wpadded -Winline -Wpointer-arith -Wsign-compare -Wendif-labels
LDFLAGS := -lm

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(CFLAGS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf $(OBJS) $(TARGET)

.c.o:
	$(CC) -c $(CFLAGS) $*.c

