TARGET= libmatmult.so
OBJS	= matmult_c.o

CC	= nvc
CXX	= nvc++

# Variables
TEAMS=114
THREADS=16
BLK_SIZE=4
SLABS=4

OPT	= -g -fast -Msafeptr -Minfo -mp=gpu -gpu=pinned -gpu=lineinfo -gpu=cc90 -cuda -mp=noautopar -cudalib=cublas 
PIC   = -fpic -shared
ISA	=
PARA	=
INC   =
LIBS	=

# DEFINEs in source code
DEFS = -D_TEAMS=$(TEAMS) -D_THREADS=$(THREADS) -D_BLK_SIZE=$(BLK_SIZE) -D_SLABS=$(SLABS)

CXXFLAGS= $(OPT) $(PIC) $(INC) $(ISA) $(PARA) $(XOPT) $(DEFS)

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

matmult_c.o: matmult_c.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^

clean:
	@/bin/rm -f $(TARGET) $(OBJS)
