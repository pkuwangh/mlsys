#!/bin/bash

# for GH200/GB200

./cuda-demos/bin/uvm_vec_add -a all

mpirun --tag-output -np 2 --host <host1>,<host2> -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./cuda-demos/bin/mpi_egm

