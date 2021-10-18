CC=gcc
CFLAGS_DEBUG=-c -fPIC -Wall -Wextra -pedantic -O0 -g
CFLAGS_RELEASE=-c -fPIC -Wall -Wextra -pedantic -O3
LDFLAGS=-shared -Wl,-soname,$@

mciast.so: mciast.o
	$(CC) $(LDFLAGS) -o $@ $<

mciast.o: mciast.c
	$(CC) $(CFLAGS_DEBUG) -o $@ $< -lm

