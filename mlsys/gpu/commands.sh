#!/bin/bash

./nvbandwidth/bin/nvbandwidth -t host_to_device_memcpy_ce -t device_to_host_memcpy_ce -t device_to_device_memcpy_read_ce -t device_to_device_memcpy_write_ce
./nvbandwidth/bin/nvbandwidth -t host_to_device_memcpy_sm -t device_to_host_memcpy_sm -t device_to_device_memcpy_read_sm -t device_to_device_memcpy_write_sm

# for GH200/GB200
./cuda-demos/bin/uvm_vec_add -a all

mpirun --tag-output -np 2 --host <host1>,<host2> -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./cuda-demos/bin/mpi_egm

