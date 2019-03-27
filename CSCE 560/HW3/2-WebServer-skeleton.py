# Import socket module
from socket import *    
from pathlib import Path, PureWindowsPath
# Create a TCP server socket
#(AF_INET is used for IPv4 protocols)
#(SOCK_STREAM is used for TCP)

#Edits made for CSCE 560 Assignment by Marvin Newlin

serverSocket = socket(AF_INET, SOCK_STREAM)
BUF_SIZE = 1024
#Prepare a server socket
#Fill in start
#Fill in end
serverPort = 12000
serverSocket.bind((' ',serverPort))
serverSocket.listen(1)
# Server should be up and running and listening to the incoming connections
while True:
    print ('Ready to serve...')
	
    # Set up a new connection from the client
    #Fill in
    connectionSocket, addr = serverSocket.accept()     
    #Fill in end
	
    # If an exception occurs during the execution of try clause
    # the rest of the clause is skipped
    # If the exception type matches the word after except
    # the except clause is executed
    try:
        # Receives the request message from the client
        #Fill in start
        message =  connectionSocket.recv(BUF_SIZE).decode()
        #Fill in end
        # Extract the path of the requested object from the message
        # The path is the second part of HTTP header, identified by [1]
        filename = message.split()[1]
        # Because the extracted path of the HTTP request includes 
        # a character '/', we read the path from the second character 
        f = open(filename[1:])
        # Store the entire content of the requested file in a temporary buffer
        #Fill in start
        outputdata = []
        with f as file:
            outputdata.append(file.read())
        #Fill in end
        # Send the HTTP response header line to the connection socket
        #Fill in start
        http_ok = "HTTP/1.1 200 OK\n\n"
        connectionSocket.send(http_ok.encode()) 
        #Fill in end       
        # Send the content of the requested file to the connection socket
        for i in range(0, len(outputdata)):  
            connectionSocket.send(outputdata[i].encode())
            connectionSocket.send("\r\n".encode())
		
        # Close the client connection socket
        connectionSocket.close()

    except IOError:
        # Send HTTP response message for file not found
        #Fill in start
        http_fnf = "HTTP/1.1 404 NOT FOUND\n\n"
        connectionSocket.send(http_fnf.encode()) 
        File404 = "filenotfound.html"
        f = open(File404)
        outputdata = []
        with f as file:
            outputdata.append(file.read())
        for i in range(0, len(outputdata)):  
            connectionSocket.send(outputdata[i].encode())
        connectionSocket.send("\r\n".encode())
        #Fill in end
    		# Close the client connection socket
        #Fill in start
        connectionSocket.close()
        #Fill in end

serverSocket.close()  
