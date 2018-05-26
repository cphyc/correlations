MODNAME=utils
SRC=utils.f90 quadpack.f90

all: utils.pyf 
	f2py -c $^ $(SRC) -m $(MODNAME)


%.pyf: %.f90
	f2py $^ -h $@ -m $(MODNAME)

clean:
	rm utils.pyf
