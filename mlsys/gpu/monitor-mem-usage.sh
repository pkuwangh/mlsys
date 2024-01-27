#!/bin/bash

numastat -mz | grep -e "Mem\|HugePages_"

