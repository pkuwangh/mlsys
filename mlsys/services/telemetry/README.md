# Telemetry

## Fluent-bit

### Build fluent-bit

```bash
sudo apt-get install cmake flex bison libyaml-dev libssl-dev
cd fluent-bit/build
cmake ..
make -j24
```

### Collect logs

```bash
# launch fluent-bit with forward input and stdout output
sudo mkdir -p /var/run/fluent
sudo chown -R $USER:$USER /var/run/fluent
ls -ld /var/run/fluent
./fluent-bit/build/bin/fluent-bit -c conf-fluent-bit-forward-stdout.conf [-vv]
```

### Emit logs using fluent-cat

```bash
sudo apt-get install ruby-dev
sudo gem install fluentd
# -u/--unix make it use unix socket instead of tcp
echo '{"key 1": 123456789, "key 2": "abcdefg"}' | fluent-cat -u my_tag
```

### Emit logs from application

Simply write `msgpack.packb([tag: str, timestamp: int, msg: Dict[str, Any]])` to socket at the path specified by `Unix_Path` specified in `fluent-bit` configuration file.

An example can be found in [`app-log-unix-socket.py`](./app-log-unix-socket.py)

```bash
./app-log-unix-socket.py
```
