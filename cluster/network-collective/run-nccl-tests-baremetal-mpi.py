#!/usr/bin/env python3

import argparse
import os
import sys
import yaml


def get_bin(args, test_name):
    if args.path:
        return os.path.join(args.path, f"{test_name}_perf")
    else:
        return f"{test_name}_perf"


def get_hostlist(args):
    hostlist = []
    if args.hostfile:
        if os.path.exists(args.hostfile):
            with open(args.hostfile, "rt") as fp:
                hosts_all = yaml.safe_load(fp)
                hosts_sel = None
                if args.domain_group is None:
                    hosts_sel = hosts_all
                else:
                    for k1, v1 in hosts_all.items():
                        if args.domain_group == k1:
                            hosts_sel = {k1: v1}
                            break
                        if type(v1) is dict:
                            for k2, v2 in v1.items():
                                if args.domain_group == k2:
                                    hosts_sel = {k2: v2}
                # print(hosts_sel)
                if hosts_sel is None:
                    print("Fail to find {args.domain_group} in hostfile")
                    sys.exit(1)
                for x in hosts_sel:
                    if type(hosts_sel[x]) is list:
                        hostlist += hosts_sel[x]
                    elif type(hosts_sel[x]) is dict:
                        for _, v in hosts_sel[x].items():
                            if type(v) is list:
                                hostlist += v
                            else:
                                print("Unexpected hostfile format; more than 2 levels?")
                                sys.exit(0)
                    else:
                        print("Unexpected hostfile format")
                        sys.exit(0)
        else:
            print(f"Fail to find {args.hostfile}")
            sys.exit(1)
    elif args.hosts:
        hostlist = args.hosts.split(",")
    else:
        print("Either --hostfile/-f or --hosts/-r has to be specified")
        sys.exit(1)
    return sorted(list(set(hostlist)))


def get_cmd(args, hostlist, test_name):
    cmd = ["mpirun"]
    cmd += ["--mca", "btl", "tcp,self"]
    if args.tcp_if:
        cmd += ["--mca", "btl_tcp_if_include", args.tcp_if]
    cmd += ["-np", str(len(hostlist))]
    cmd += ["--host", ",".join(hostlist)]
    cmd += ["-x", "NCCL_DEBUG=INFO"]
    cmd += ["-x", "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"]
    cmd += [get_bin(args, test_name)]
    cmd += [f"-t{args.num_gpus}"]
    cmd += [f"-g1"]
    cmd += [f"-b{args.begin_size}"]
    cmd += [f"-e{args.end_size}"]
    cmd += [f"-f2"]
    cmd += [f"-c0"]
    return cmd


def main(args):
    hostlist = get_hostlist(args)
    print(f"Running on the following {len(hostlist)} hosts:")
    for hostname in hostlist:
        print(f"-- {hostname}")
    testlist = args.test if len(args.test) == 1 else args.test[1:]
    for test_name in testlist:
        print(f"\nRunning {test_name} ...")
        cmd = get_cmd(args, hostlist, test_name)
        print(" ".join(cmd))
        if args.real:
            os.system(" ".join(cmd))


def get_nccl_tests():
    return [
        "all_gather",
        "all_reduce",
        "alltoall",
        "broadcast",
        "gather",
        "reduce",
        "reduce_scatter",
        "scatter",
        "sendrecv",
    ]


def setup_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hostfile", "-f", type=str, default=None, help="host yaml file"
    )
    parser.add_argument(
        "--domain-group",
        "-d",
        type=str,
        default=None,
        help="group name in host yaml file, e.g. l2 or l1_01",
    )
    parser.add_argument(
        "--hosts", "-r", type=str, default=None, help="comma-separated hostnames"
    )
    parser.add_argument(
        "--path", "-p", type=str, default=None, help="path to NCCL binaries"
    )
    parser.add_argument(
        "--tcp-if",
        "-i",
        type=str,
        default=None,
        help="specific TCP interface for mpirun mca btl_tcp_if_include, e.g. enP4s1f1",
    )
    parser.add_argument(
        "--test",
        action="append",
        default=["sendrecv"],
        choices=get_nccl_tests(),
        help="specific NCCL tests; accept multiple -t",
    )
    parser.add_argument(
        "--num-gpus",
        "-g",
        type=int,
        default=1,
        help="num of gpus per node; will run 1 thread per gpu",
    )
    parser.add_argument(
        "--begin-size",
        "-b",
        type=str,
        default="16",
        help="message size start",
    )
    parser.add_argument(
        "--end-size",
        "-e",
        type=str,
        default="2g",
        help="message size end",
    )
    parser.add_argument("--real", action="store_true", help="for real")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
