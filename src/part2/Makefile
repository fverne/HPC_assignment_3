THREAD_LIMIT ?= 1024
NUM_TEAMS ?= 114

$(info NUM_TEAMS: $(NUM_TEAMS))
$(info THREAD_LIMIT: $(THREAD_LIMIT))

# Targets
TARGET_J = poisson_j
TARGET_JA = poisson_ja
TARGET_JOMP = poisson_jomp
TARGET_JDIST = poisson_jdist
TARGET_JEXPAND = poisson_jexpand
TARGET_JDUP = poisson_jdup

# Main object files
MAIN_J	= main_j.o
MAIN_JA = main_ja.o
MAIN_JOMP = main_jomp.o
MAIN_JDIST = main_jdist.o
MAIN_JEXPAND = main_jexpand.o
# Add multiple gpu version
MAIN_JDUP = main_jdup.o

# Object files for each target
OBJS_J	= $(MAIN_J) jacobi.o
OBJS_JA = $(MAIN_JA) jacobi_alloc.o
OBJS_JOMP = $(MAIN_JOMP) jacobi_omp.o
OBJS_JDIST = $(MAIN_JDIST) jacobi_dist.o
OBJS_JEXPAND = $(MAIN_JEXPAND) jacobi_expand.o
OBJS_JDUP = $(MAIN_JDUP) jacobi_dup.o

# Common object files
OBJS_COMMON = print.o alloc3d.o util.o

# Compiler settings
CXX	= nvc++

# Compiler flags
OPT	= -g -fast -Msafeptr -Minfo -acc -mp=gpu -gpu=pinned -gpu=cc90 -gpu=lineinfo -cuda -mp=noautopar
ISA	=
PARA	=
INC   =
LIBS	=
EXTRA =-v -D_THREAD_LIMIT=$(THREAD_LIMIT) -D_NUM_TEAMS=$(NUM_TEAMS)

CXXFLAGS = $(OPT) $(EXTRA)

all: $(TARGET_J) $(TARGET_JA) $(TARGET_JOMP) $(TARGET_JDIST) $(TARGET_JEXPAND) $(TARGET_JDUP)

$(TARGET_J): $(OBJS_COMMON) $(OBJS_J)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJS_J) $(OBJS_COMMON) $(LIBS)

$(TARGET_JA): $(OBJS_COMMON) $(OBJS_JA)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJS_JA) $(OBJS_COMMON) $(LIBS)

$(TARGET_JOMP): $(OBJS_COMMON) $(OBJS_JOMP)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJS_JOMP) $(OBJS_COMMON) $(LIBS)

$(TARGET_JDIST): $(OBJS_COMMON) $(OBJS_JDIST)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJS_JDIST) $(OBJS_COMMON) $(LIBS)

$(TARGET_JEXPAND): $(OBJS_COMMON) $(OBJS_JEXPAND)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJS_JEXPAND) $(OBJS_COMMON) $(LIBS)

$(TARGET_JDUP): $(OBJS_COMMON) $(OBJS_JDUP)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJS_JDUP) $(OBJS_COMMON) $(LIBS)

$(MAIN_J): main.cpp print.h jacobi.h
	$(CXX) -o $@ -D_JACOBI $(CXXFLAGS) -c main.cpp 

$(MAIN_JA): main.cpp print.h jacobi_alloc.h
	$(CXX) -o $@ -D_JACOBI_ALLOC $(CXXFLAGS) -c main.cpp 

$(MAIN_JOMP): main.cpp print.h jacobi_omp.h
	$(CXX) -o $@ -D_JACOBI_OMP $(CXXFLAGS) -c main.cpp

$(MAIN_JDIST): main.cpp print.h jacobi_dist.h
	$(CXX) -o $@ -D_JACOBI_DIST $(CXXFLAGS) -c main.cpp

$(MAIN_JEXPAND): main.cpp print.h jacobi_expand.h
	$(CXX) -o $@ -D_JACOBI_EXPAND $(CXXFLAGS) -c main.cpp

$(MAIN_JDUP): main.cpp print.h jacobi_dup.h
	$(CXX) -o $@ -D_JACOBI_DUP $(CXXFLAGS) -c main.cpp

print.o: print.cpp print.h
	$(CXX) -o $@ $(CXXFLAGS) -c print.cpp

alloc3d.o: alloc3d.cpp
	$(CXX) -o $@ $(CXXFLAGS) -c alloc3d.cpp

util.o: util.cpp
	$(CXX) -o $@ $(CXXFLAGS) -c util.cpp

clean:
	/bin/rm -f *.o $(TARGET_J) $(TARGET_JA) $(TARGET_JOMP) $(TARGET_JDIST)  $(TARGET_JEXPAND) $(TARGET_JDUP)
