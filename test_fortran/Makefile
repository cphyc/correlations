MODNAME=utils
SRC=utils.f90
FC=gfortran
F90FLAGS=-O3 -fPIC -march=native -Wall -fopenmp

CFLAGS=-I/home/ccc/.local/include/fgsl # $(shell pkg-config --cflags fgsl)
LFLAGS= -L/home/ccc/.local/lib -lfgsl -lgsl -lgslcblas -lm #$(shell pkg-config --libs fgsl)

F2PY_FUNCTIONS=init integrate sigma
OBJS=hashtable.o constants.o utils.o

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
	$(FC) $(F90FLAGS) $(LFLAGS) $(OBJS) test.f90 -o test
	./test | tee test.output_f
	# python ./test.py | tee test.output_py
