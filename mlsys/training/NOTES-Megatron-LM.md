# Training Megatron-LM in Slurm Cluster

## Preparation

```bash
# find out my resource
scontrol show reservation <reservation name>

# find out current running jobs
squeue -R <reservation name>

# find out past jobs
sacct -a --format="JobID,JobName,User,Reservation,Partition,Account,State,ExitCode,Elapsed,NNodes" | grep <reservation name prefix>
```

## Launch

```bash
# launch
sbatch gpt3_training-n4.sh

# check my running job
squeue -u $USER
# check my past jobs
sacct --format="JobID,JobName,User,Reservation,Partition,Account,State,ExitCode,Elapsed,NNodes"
```
