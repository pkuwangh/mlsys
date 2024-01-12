#!/bin/bash

./run-nccl-tests-baremetal-mpi.py -r s-s126-bot-dvt1c-ks-1183,s-s126-bot-dvt1c-ks-1184,s-s126-bot-dvt1c-ks-1185,s-s126-bot-dvt1c-ks-1186 -i enP4s1f1 -p /home/nvidia/haowan/nccl_test/ --test all_reduce --real > ~/haowan/results/log_nccl_allreduce_l1_02_n4
./run-nccl-tests-baremetal-mpi.py -f ./hosts/superpowers.yml -d l1_02 -i enP4s1f1 -p /home/nvidia/haowan/nccl_test/ --test all_reduce --real > ~/haowan/results/log_nccl_allreduce_l1_02_n8
./run-nccl-tests-baremetal-mpi.py -f ./hosts/superpowers.yml -d l2 -i enP4s1f1 -p /home/nvidia/haowan/nccl_test/ --test all_reduce --real > ~/haowan/results/log_nccl_allreduce_l2_n16

./run-nccl-tests-baremetal-mpi.py -r s-s126-bot-dvt1c-ks-1183,s-s126-bot-dvt1c-ks-1184,s-s126-bot-dvt1c-ks-1185,s-s126-bot-dvt1c-ks-1186 -i enP4s1f1 -p /home/nvidia/haowan/nccl_test/ --test all_gather --real > ~/haowan/results/log_nccl_allgather_l1_02_n4
./run-nccl-tests-baremetal-mpi.py -f ./hosts/superpowers.yml -d l1_02 -i enP4s1f1 -p /home/nvidia/haowan/nccl_test/ --test all_gather --real > ~/haowan/results/log_nccl_allgather_l1_02_n8
./run-nccl-tests-baremetal-mpi.py -f ./hosts/superpowers.yml -d l2 -i enP4s1f1 -p /home/nvidia/haowan/nccl_test/ --test all_gather --real > ~/haowan/results/log_nccl_allgather_l2_n16

