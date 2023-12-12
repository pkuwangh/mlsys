#!/bin/bash

# options
#SBATCH --time=1

# execute on the first node in the allocation
/bin/hostname

# 2 job steps
srun -l /bin/hostname
srun -l /bin/pwd
