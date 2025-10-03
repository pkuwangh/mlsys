default:
	mkdir -p bin/
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON .
	make -j${nproc} -C build
	cp build/nvbandwidth bin/

single_node:
	mkdir -p bin/
	cmake -B build -DCMAKE_BUILD_TYPE=Release .
	make -j${nproc} -C build
	cp build/nvbandwidth bin/

clean:
	if [ -d "build" ]; then make -C build clean; fi
	rm -f bin/*

clobber:
	rm -rf build/
	rm -f bin/*

