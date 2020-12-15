CC = gcc
CFLAGS = -O2 -march=native

.PHONY: all clean

all: step

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

