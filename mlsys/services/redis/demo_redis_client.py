#!/usr/bin/env python3

import redis
import time

r = redis.Redis(host="localhost", port=6379, db=0)

commands = [
    ["setting key=foo value=bar", r.set, ("foo", "bar")],
    ["setting key=foo value=barx", r.set, ("foo", "barx")],
    ["getting key=foo", r.get, ("foo",)],
    ["getting key=fooxxx", r.get, ("fooxxx",)],
    ["adding key=cnt value=1", r.incrby, ("cnt", 1)],
    ["adding key=cnt value=1", r.incrby, ("cnt", 1)],
    ["adding key=foo value=1", r.incrby, ("foo", 1)],
    ["deleting key=foo", r.delete, ("foo",)],
    ["deleting key=fooxxx", r.delete, ("fooxxx",)],
]

for cmd in commands:
    print(cmd[0])
    try:
        x = cmd[1](*cmd[2])
    except Exception as e:
        print(f"caught exception {e}")
    else:
        print(f"returns {x} of type {type(x)}")

print(f"\nall keys: {r.keys()}")

print("\npipeline")
pipe = r.pipeline()
pipe.set("foo1", "bar1")
pipe.set("foo2", "bar2")
x = pipe.execute()
print(x)
pipe.get("foo1")
pipe.get("foo2")
pipe.get("foo3")
x = pipe.execute()
print(x)

time.sleep(10)
