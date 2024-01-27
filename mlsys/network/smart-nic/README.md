# Smart NICs

- [Smart NICs](#smart-nics)
  - [BlueField-3 DPU](#bluefield-3-dpu)
    - [Documentation](#documentation)
    - [Setup](#setup)
      - [Useful checks - Host](#useful-checks---host)
      - [Install packages - Host](#install-packages---host)
      - [Useful checks - DPU](#useful-checks---dpu)
      - [Install packages - DPU](#install-packages---dpu)
      - [Configuration from host](#configuration-from-host)
  - [L2 Reflection](#l2-reflection)
    - [Test connection with DPDK](#test-connection-with-dpdk)
      - [Install DPDK and Pktgen](#install-dpdk-and-pktgen)
      - [Running testpmd](#running-testpmd)

## BlueField-3 DPU

### Documentation

- [BF DPU adminstrator quick start guide](https://docs.nvidia.com/networking/display/bf3dpuvpi/bluefield+dpu+administrator+quick+start+guide)
- [NVIDIA DOCA developer quick start guide](https://docs.nvidia.com/doca/sdk/developer-qsg/index.html)
  - [Linux installation guide](https://docs.nvidia.com/doca/sdk/installation-guide-for-linux/index.html#manual-bluefield-image-installation)

### Setup

#### Useful checks - Host

```bash
# list PCIe devices
sudo lspci | grep Mellanox
sudo ibdev2netdev -v

# get device ID
sudo mst status -v

# check DPU mode config, link type, etc.
sudo mlxconfig -d /dev/mst/mt41692_pciconf0 -e q | grep -e "INTERNAL\|LINK_TYPE"
```

#### Install packages - Host

```bash
wget http://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox-SHA256

# MOFED
wget https://content.mellanox.com/ofed/MLNX_OFED-23.07-0.5.1.2/MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu22.04-aarch64.tgz
tar xvf MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu22.04-aarch64.tgz
cd MLNX_OFED_LINUX-23.07-0.5.1.2-ubuntu22.04-aarch64/
sudo ./mlnxofedinstall
sudo /etc/init.d/openibd restart

# rshim - user-space driver for BlueField SoC
sudo apt install rshim
sudo systemctl status rshim

# assign a dynamic IP to tmfifo_net0 interface (RShim host interface)
sudo ifconfig tmfifo_net0 192.168.100.1 netmask 255.255.255.252 up
```

#### Useful checks - DPU

```bash
# check current running BFB
sudo cat /etc/mlnx-release

# BF version check
sudo bfvcheck
```

#### Install packages - DPU

```bash
# from host, download image
wget https://content.mellanox.com/BlueField/BFBs/Ubuntu22.04/DOCA_2.2.1_BSP_4.2.2_Ubuntu_22.04-13.23-09.prod.bfb
# install a new BFB
sudo bfb-install --bfb DOCA_2.2.1_BSP_4.2.2_Ubuntu_22.04-13.23-09.prod.bfb --rshim rshim0

# get onto DPU, ubuntu:xTRA123BF3
sudo screen -S hao /dev/rshim0/console 115200
sudo screen -ls
sudo screen -r hao
# or directly ssh
ssh ubuntu@192.168.100.2
```

#### Configuration from host

```bash
# configure DPU mode
sudo mst start
sudo mst status -v
# reset to default DPU mode (from e.g. NIC mode)
# revert back from NIC mode
sudo mlxconfig -d /dev/mst/mt41692_pciconf0 s INTERNAL_CPU_MODEL=1 \
    INTERNAL_CPU_PAGE_SUPPLIER=0 \
    INTERNAL_CPU_ESWITCH_MANAGER=0 \
    INTERNAL_CPU_IB_VPORT0=0 \
    INTERNAL_CPU_OFFLOAD_ENGINE=0
# or reset to default values
sudo mlxconfig -d /dev/mst/mt41692_pciconf0 r
# reboot required
sudo reboot
sudo mlxconfig -d /dev/mst/mt41692_pciconf0 q | grep INTERNAL
```

## L2 Reflection

### Test connection with DPDK

#### Install DPDK and Pktgen

```bash
# install DPDK
grep HUGETLB /boot/config-$(uname -r)
grep -i Hugepagesize /proc/meminfo
echo 1024 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
# echo 64 | sudo tee /sys/kernel/mm/hugepages/hugepages-524288kB/nr_hugepages
grep -i huge /proc/meminfo
sudo apt-get install build-essential meson ninja-build python3-pyelftools libnuma-dev
wget https://fast.dpdk.org/rel/dpdk-23.07.tar.xz
tar xvf dpdk-23.07.tar.xz && cd dpdk-23.07/
meson setup -Dexamples=l2fwd,l3fwd build
cd build/
ninja
sudo meson install
sudo ldconfig
# install pktgen-dpdk
export PKG_CONFIG_PATH=/usr/local/lib/aarch64-linux-gnu/pkgconfig
sudo apt-get install libpcap-dev
git clone git://dpdk.org/apps/pktgen-dpdk
make rebuild
```

#### Running testpmd

```bash
# On host side; the exact device ID below is found by `sudo mst status -v | grep mlx5_2`
sudo ./pktgen -l 4-7 -a 06:01:00.1

# On DPU side
sudo ./dpdk-testpmd -l 4-15 -a 03:00.0 -- --burst=4 --forward-mode=macswap
```
