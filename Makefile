CC=g++
CXX=g++
RANLIB=ranlib

LIBSRC=MapReduceFramework.cpp MapReduceFramework.h Barrier.cpp Barrier.h
LIBOBJ=$(LIBSRC:.cpp=.o)

INCS=-I.
CFLAGS = -Wall -std=c++11 -g $(INCS)
CXXFLAGS = -Wall -std=c++11 -g $(INCS)

MAPREDUCEFRAMEWORKLIB = libMapReduceFramework.a
TARGETS = $(MAPREDUCEFRAMEWORKLIB)

TAR=tar
TARFLAGS=-cvf
TARNAME=ex3.tar
TARSRCS=$(LIBSRC) Makefile README

all: $(TARGETS)

$(TARGETS): $(LIBOBJ)
	$(AR) $(ARFLAGS) $@ $^
	$(RANLIB) $@

clean:
	$(RM) $(TARGETS) $(MAPREDUCEFRAMEWORKLIB) $(OBJ) $(LIBOBJ) *~ *core

depend:
	makedepend -- $(CFLAGS) -- $(SRC) $(LIBSRC)

tar:
	$(TAR) $(TARFLAGS) $(TARNAME) $(TARSRCS)
.PHONY