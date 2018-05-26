MODNAME=utils
SRC=utils.f90
FC=gfortran
F90FLAGS=-O3 -fPIC -march=native

CFLAGS=$(shell pkg-config --cflags fgsl)
LFLAGS= $(shell pkg-config --libs fgsl)

F2PY_FUNCTIONS=init integrate sigma
OBJS=constants.o utils.o

TARGET=utils.so

all: $(OBJS)
	$(FC) $(F90FLAGS) $(LFLAGS) $^ -shared -o $(TARGET)

# %.pyf: %.f90
# 	f2py $^ -h $@ -m $(MODNAME) only: $(F2PY_FUNCTIONS)

%.o: %.f90
	$(FC) -c $^ -o $@ $(F90FLAGS) $(CFLAGS)

clean:
	rm *.o *.mod -f

test: $(OBJS) test.f90
	$(FC) $(F90FLAGS) $(LFLAGS) $^ -o test
