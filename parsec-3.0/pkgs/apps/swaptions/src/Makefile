PREFIX := ${PARSECDIR}/pkgs/apps/swaptions/inst/${PARSECPLAT}
RSDGPREFIX = /home/ubuntu/parsec-3.0/rsdglib
INCLUDE = -I$(RSDGPREFIX) -I/usr/include/curl/
CXXFLAGS = -std=c++11 -pthread -O0 
LIBS = -lcurl
EXEC = swaptions 

OBJS= CumNormalInv.o MaxFunction.o RanUnif.o nr_routines.o icdf.o \
        HJM_SimPath_Forward_Blocking.o HJM.o HJM_Swaption_Blocking.o  \
        HJM_Securities.o

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(RSDGPREFIX)/rsdg.a -o $@ $(LIBS) $(INCLUDE)

.cpp.o: 
	$(CXX) $(CXXFLAGS) -c $*.cpp $(INCLUDE) -o $*.o

.c.o:
	$(CXX) $(CXXFLAGS) $(DEF) -c $*.c -o $*.o

clean:
	rm -f $(OBJS) $(EXEC)

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(EXEC) $(PREFIX)/bin/$(EXEC)

