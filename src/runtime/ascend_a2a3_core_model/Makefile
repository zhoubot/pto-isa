# Ascend A2/A3 Core Simulator Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -fPIC
DEBUG_FLAGS = -g -DDEBUG

# Source files
SRCS = a2a3_core_model.c a2a3_incore_sim.c
OBJS = $(SRCS:.c=.o)
DEPS = a2a3_core_model.h a2a3_incore_sim.h

# Output library
LIB = liba2a3_core.a
SHARED_LIB = liba2a3_core.so

# Test program
TEST_SRC = test_core_sim.c
TEST_BIN = test_core_sim

.PHONY: all clean test debug lib shared

all: lib

lib: $(LIB)

shared: $(SHARED_LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(SHARED_LIB): $(OBJS)
	$(CC) -shared -o $@ $^

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS += $(DEBUG_FLAGS)
debug: clean lib

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -la2a3_core

clean:
	rm -f $(OBJS) $(LIB) $(SHARED_LIB) $(TEST_BIN)

# Installation
PREFIX ?= /usr/local
install: lib
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include/a2a3_core
	install -m 644 $(LIB) $(PREFIX)/lib/
	install -m 644 $(DEPS) $(PREFIX)/include/a2a3_core/

.PHONY: install
