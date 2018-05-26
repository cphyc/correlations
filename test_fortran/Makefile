MODNAME=utils
SRC=utils.f90
CFLAGS=$(shell pkg-config --cflags fgsl)
LFLAGS= $(shell pkg-config --libs fgsl)

all: utils.pyf 
	f2py -c $^ $(SRC) -m $(MODNAME) $(CFLAGS) $(LFLAGS) --verbose


%.pyf: %.f90
	f2py $^ -h $@ -m $(MODNAME)

clean:
	rm utils.pyf
