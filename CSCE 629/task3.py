#CSCE 629 Task 3
#Marvin Newlin & James Marvin
from scapy.all import *
from socket import *
import time

miServer = 'mi5.m4i.local'
serverPort = 2600
sock = conf.L3socket(iface='eth0')


payload = 'Ethan Hunt '
srcPort = [3, 33, 333, 3333, 33333]
payloadNum = []
for x in range 0 to 9999:
	payloadNum.append(x)


for x in range 0 to 9999:
	payload_total = payload + str(payloadNum[x])
	sourceP = srcPort[x//5]
	sock.send(IP(dst=miServer)/UDP(dport=serverPort, sport=sourceP)/payload_total)
	sleep(0.1)