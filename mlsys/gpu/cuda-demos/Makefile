default:
	mkdir -p bin/
	cmake -B build -DCMAKE_INSTALL_PREFIX=$(shell pwd) -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON -DMULTINODE=ON .
	make -j${nproc} -C build install

single:
	mkdir -p bin/
	cmake -B build -DCMAKE_INSTALL_PREFIX=$(shell pwd) -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON .
	make -j${nproc} -C build install

clean:
	if [ -d "build" ]; then make -C build clean; fi
	rm -f bin/*

clobber:
	rm -rf build/
	rm -f bin/*

