default: hopper

__arch_common:
	mkdir -p bin/

blackwell: __arch_common
	cmake -B build -DCMAKE_INSTALL_PREFIX=$(shell pwd) -DCMAKE_BUILD_TYPE=Release -DMLSYS_CUDA_ARCHITECTURES="100a-real;100a-virtual" .
	make -j${nproc} -C build install

hopper: __arch_common
	cmake -B build -DCMAKE_INSTALL_PREFIX=$(shell pwd) -DCMAKE_BUILD_TYPE=Release -DMLSYS_CUDA_ARCHITECTURES="90a-real;90a-virtual" .
	make -j${nproc} -C build install

amphere: __arch_common
	cmake -B build -DCMAKE_INSTALL_PREFIX=$(shell pwd) -DCMAKE_BUILD_TYPE=Release -DMLSYS_CUDA_ARCHITECTURES="86-real;86-virtual" .
	make -j${nproc} -C build install

clean:
	if [ -d "build" ]; then make -C build clean; fi
	rm -f bin/*

clobber:
	rm -rf build/
	rm -f bin/*

