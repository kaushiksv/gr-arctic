#!/usr/bin/env python

import socket


TCP_IP = '127.0.0.1'
TCP_PORT = 5006
BUFFER_SIZE = 20  # Normally 1024, but we want fast response

while True:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    conn, addr = s.accept()
    print('Connection address:', addr)
    while 1:
        data = conn.recv(BUFFER_SIZE)
        print("type(data) = " + str(type(data)) + "\n")
        if not data: break
        # self.CLTUs.push(self.ack_seq + data)
        print("received data:", data)
        conn.send(data) # echo
        conn.send(bytes("200 OK (by k)\n", "utf-8"))
    conn.close()
