#!/usr/bin/env python3

import msgpack
import socket
from datetime import datetime


def send_log_to_fluent_bit(message):
    # Define the Unix socket path
    socket_path = "/var/run/fluent/fluent.sock"

    # Create a socket object using AF_UNIX as the address family
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        # Connect to the Fluent Bit socket
        sock.connect(socket_path)

        # Convert the message to JSON and encode it to bytes
        fluent_msg = ["app-python", int(datetime.now().timestamp()), message]
        message_bytes = msgpack.packb(fluent_msg)

        # Send the encoded message
        sock.send(message_bytes)


# Example log message as a dictionary
log_message = {
    "job": "test-python",
    "shape_m": 32,
    "shape_n": 64,
    "__value__": 8.5,
}

# Send the log message to Fluent Bit
send_log_to_fluent_bit(log_message)
