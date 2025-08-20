#!/bin/bash

./bin/nvbandwidth -t host_to_device_memcpy_ce -t device_to_host_memcpy_ce -t device_to_device_memcpy_read_ce -t device_to_device_memcpy_write_ce
./bin/nvbandwidth -t host_to_device_memcpy_sm -t device_to_host_memcpy_sm -t device_to_device_memcpy_read_sm -t device_to_device_memcpy_write_sm

# for GH200/GB200
./bin/uvm_vec_add -a all

mpirun --tag-output -np 2 --host <host1>,<host2> -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./bin/mpi_egm

