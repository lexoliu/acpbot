"""`acpbot mcp-bridge` equivalent for test images that lack an acpbot binary.

Pumps stdin to a `tcp:<host>:<port>` socket and the socket back to stdout —
byte-level, exactly like the real bridge's tcp path.
"""

import socket
import sys
import threading


def main():
    target = sys.argv[1]
    assert target.startswith("tcp:"), f"unsupported bridge target {target!r}"
    host, port = target[4:].rsplit(":", 1)
    sock = socket.create_connection((host, int(port)))

    def upstream():
        for line in sys.stdin:
            sock.sendall(line.encode())

    threading.Thread(target=upstream, daemon=True).start()
    reader = sock.makefile("r")
    for line in reader:
        sys.stdout.write(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
