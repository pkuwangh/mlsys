default:
	mkdir -p bin/
	cmake -B nvbandwidth/build -DCMAKE_BUILD_TYPE=Release nvbandwidth/
	make -j${nproc} -C nvbandwidth/build
	cp nvbandwidth/build/nvbandwidth bin/
	cmake -B cuda-demos/build -DCMAKE_INSTALL_PREFIX=$(shell pwd) -DCMAKE_BUILD_TYPE=Release cuda-demos/
	make -j${nproc} -C cuda-demos/build install

clean:
	if [ -d "nvbandwidth/build" ]; then make -C nvbandwidth/build clean; fi
	if [ -d "cuda-demos/build" ]; then make -C cuda-demos/build clean; fi
	rm -f bin/*

clobber:
	rm -rf nvbandwidth/build
	rm -rf cuda-demos/build
	rm -f bin/*

