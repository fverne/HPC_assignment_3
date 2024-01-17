TARGET= libmatmult.so
OBJS	= matmult_c.o

CC	= nvc
CXX	= nvc++

# Variables
TEAMS=16384
THREADS=16

OPT	= -g -fast -Msafeptr -Minfo -mp=gpu -gpu=pinned -gpu=lineinfo -gpu=cc90 -cuda -mp=noautopar -cudalib=cublas
PIC   = -fpic -shared
ISA	=
PARA	=
INC   =
LIBS	=

# DEFINEs in source code
DEFS = -D_TEAMS=$(TEAMS) -D_THREADS=$(THREADS)

CXXFLAGS= $(OPT) $(PIC) $(INC) $(ISA) $(PARA) $(XOPT) $(DEFS)

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

matmult_c.o: matmult_c.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^

clean:
	@/bin/rm -f $(TARGET) $(OBJS)
