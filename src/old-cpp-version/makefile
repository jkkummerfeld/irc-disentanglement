# Written with help from https://stackoverflow.com/questions/2481269/how-to-make-a-simple-c-makefile

CPPFLAGS=-g -std=c++11 -I$(DYNET) -I$(EIGEN) -I.
LDLIBS=-ldynet
LDFLAGS=-L$(DYNET)/build/dynet

SRCS=data.cc models.cc main.cc eval.cc
OBJS=$(subst .cc,.o,$(SRCS))

all: predictor

predictor: $(OBJS)
	$(CXX) $(LDFLAGS) -o predictor $(OBJS) $(LDLIBS) -Wl,-rpath,$(DYNET)/build/dynet

eval.o: eval.cc eval.h
	$(CXX) $(CPPFLAGS) -c -o $@ $<

data.o: data.cc data.h eval.h
	$(CXX) $(CPPFLAGS) -c -o $@ $<

models.o: models.cc models.h eval.h
	$(CXX) $(CPPFLAGS) -c -o $@ $<

main.o: main.cc models.h data.h eval.h
	$(CXX) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -rf *.o
