# Slurm

- [Slurm](#slurm)
  - [Architecture](#architecture)
  - [Commands](#commands)
    - [Admin](#admin)
    - [Run](#run)
    - [Interact](#interact)
    - [Example commands](#example-commands)

Slurm is a cluster management and job scheduling system for Linux clusters.

## Architecture

Components

- a `slurmd` daemon on each compute node
- a central `slurmctld` daemon on management node

Concepts

- partition: which group nodes into logical sets, i.e. a job queue
- job steps: which are sets of tasks within a job

## Commands

### Admin

- `sinfo`: report state of paritions and nodes managed by Slurm.
- `sacct`: report job or job step accounting info about active / completed jobs.
- `scontrol`: view and/or modify Slurm state.
- `sstat`: get information about the resources utilized by a running job or job step.

### Run

- `sbatch`: submit a job script for later execution.
  - The script typically contains one or more `srun` commands to launch parallel tasks.
- `srun`: submit a job for execution or initiate job steps in real time.
- `salloc`: allocate resources for a job in real time.
  - This is typically used to allocate resources and spawn a shell; and then execute `srun` commands in the shell.

### Interact

- `sattach`: attach stdin, stdout, stderr plus signal capabilities.
- `sbcast`: transfer a file from local disk to compute nodes.
- `scancel`: cancel a pending or running job / job step.

### Example commands

```bash
# cluster/machines
sinfo -p <partition>
sinfo -T  # reservation info
sinfo -R  # reason for down / drained nodes
sinfo -n <nodes> -v
scontrol show <entity> [ID]
# entity can be partition, reservation, node, topology, or job
scontrol show partition [partition]
scontrol show node [nodes]

# jobs
# historic job ID & name, AllocCPUs, state
sacct [-u <USER>]
sacct -a -r <partition> -s <state>  # running: r
sacct -j <job id> -l --json | jq

# running job ID & name, user, nodes, node list
squeue -p <partition>
squeue -R <reservation>
squeue -u <USER> -t all   # --states/-t
squeue -j <job id> -l     # --long/-l
squeue -j <job id> -l -v
squeue -n <job name> -l -v

# details of jobs
sstat -j <job id>
scontrol show job [job id]

# run!
# create a resource alloc and launch tasks for a job step in one command
# number of nodes to run on
srun -N3 -l /bin/hostname
# number of tasks to run
srun -n4 -l /bin/hostname

# submit a script for later execution
# on specific nodes, subsequent job steps spawn 4 tasks each;
# in the script, options can be added with a prefix of "#SBATCH"
sbatch -n4 -w "gpu-n31-[01-03]" -o demo.stdout ./examples/slurm-sbatch-demo-script.sh

# create a resource alloc and spawn job steps
# create resource allocation
salloc -N16 -p <partition> [--reservation name -w hosts -x hosts --nodefile filename] bash
# this start a shell within the allocation
# transfer file to each node
sbcast ./examples/slurm-runner-sbcast-demo.sh /tmp/runner.sh
# launch the run
srun /tmp/runner.sh
srun rm /tmp/runner.sh
# exit the created shell and release the resource
exit
```
