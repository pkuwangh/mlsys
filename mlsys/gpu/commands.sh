#!/bin/bash

mpirun --tag-output -np 2 --host <host1>,<host2> -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./bin/mpi_egm

