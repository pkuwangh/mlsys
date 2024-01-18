default:
	cmake -B nvbandwidth/build  -DCMAKE_INSTALL_PREFIX=$(shell pwd) -DCMAKE_BUILD_TYPE=Release nvbandwidth/
	make -j${nproc} -C nvbandwidth/build
	mkdir -p bin/
	cp nvbandwidth/build/nvbandwidth bin/

clean:
	if [ -d "nvbandwidth/build" ]; then make -C nvbandwidth/build clean; fi
	rm -f bin/nvbandwidth

clobber:
	rm -rf nvbandwidth/build
	rm -rf bin/

