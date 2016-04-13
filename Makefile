
all:


build: force_look
	python setup.py build

install: force_look
	python setup.py install --prefix=$(HOME)/local/

force_look:
	True
