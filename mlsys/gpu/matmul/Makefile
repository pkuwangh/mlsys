default:
	mkdir -p bin/
	cmake -B build -DCMAKE_INSTALL_PREFIX=$(shell pwd) -DCMAKE_BUILD_TYPE=Release .
	make -j${nproc} -C build install

amphere:
	mkdir -p bin/
	cmake -B build -DCMAKE_INSTALL_PREFIX=$(shell pwd) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86-real;86-virtual" .
	make -j${nproc} -C build install

clean:
	if [ -d "build" ]; then make -C build clean; fi
	rm -f bin/*

clobber:
	rm -rf build/
	rm -f bin/*

