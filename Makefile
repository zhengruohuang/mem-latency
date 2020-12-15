CC = gcc
CFLAGS = -O2 -march=native

TARGETS = step

.PHONY: all clean

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

