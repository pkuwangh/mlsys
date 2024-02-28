# Redis

## Build Redis

```bash
cd redis/
make

# to build with TLS support
make BUILD_TLS=yes
# to build with systemd support
make USE_SYSTEMD=yes

# to fix problem with dependencies
make distclean
```

## Run Redis

```bash
# server
./redis/src/redis-server redis.conf

# cli
# stats
./redis/src/redis-cli -i 5 --stat
# reset stats
./redis/src/redis-cli flushdb
# list all keys
./redis/src/redis-cli keys \*
```

## benchmarking

### redis-benchmark

```bash
# simple
./redis/src/redis-benchmark -h 127.0.0.1
# addtional args
# -r <key space range>, default to rand_int
# -q <quiet>, only QPS
# -c <# clients>
# -n <# reqs>
# -d <size>
# -P <pipelined # reqs per client req>
# -t <commands>, e.g. `SET,GET`
# -l <loop forever>

# sweep connections
./redis/src/redis-benchmark -h 127.0.0.1 -c <1/5/10/50/100> -n 2000000 -t SET -d 100 -q
```
