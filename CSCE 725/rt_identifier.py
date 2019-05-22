
import pandas as pd

# Filename for mil bus traffic
dumpfile = 'dumpfile.txt'
mil_bus_dump = pd.read_csv(dumpfile, sep=',', header=None, skipinitialspace=True)

# Select Status Word Column
status_words = mil_bus_dump.iloc[:,5]


# Converts hex string representing full status word
# to the integer RT Address
def convert_hex_rt_addr(hex_string):
	dec = int(hex_string,16)
	bin_rep = bin(dec)
	# Discard first two elements of binary array (garbage)
	bin_rep = bin_rep[2:]
	# Remote Terminal Address
	rt_addr =  bin_rep[3:8]
	rt_addr_int = int(rt_addr,2)
	return rt_addr_int

# Counts the number of Unique RTs that send Status Words
# and also the RT address for every message sent
def locate_rt_address(address_list, total_rt_list):
	identified_rt_list = []
	for hex_string in address_list:
		rt_addr_int = convert_hex_rt_addr(hex_string)
		total_rt_list.append(rt_addr_int)
		if rt_addr_int not in identified_rt_list:
			identified_rt_list.append(rt_addr_int)
	return identified_rt_list, total_rt_list

total_rt_list = []
address_list = status_words
identified_rt_list, total_rt_list = locate_rt_address(address_list, total_rt_list)
# Repeat Process with RT 2 RT Status Words
rt2rt_address_list = mil_bus_dump.iloc[:,4]
total_rt2rt_list = []
identified_rt_rt_status_list,total_rt2rt_list = locate_rt_address(rt2rt_address_list, total_rt2rt_list)
	
freq_list = []
freq_rt2rt_list = []

print("Number of Identified Remote Terminals: ",len(identified_rt_list))
print("List of Identified Remote Terminals")
print(identified_rt_list)	
	
	
for unique_rt in identified_rt_list:
	freq = total_rt_list.count(unique_rt)
	freq_list.append((unique_rt, freq))

# Print Table for RTs and message Frequency
print("RT Addresses and Message Frequency")	
for item in freq_list:
	print(item)
    
print("Number of Terminals that sent RT 2 RT Status Words: ",len(identified_rt_rt_status_list))
print(identified_rt_rt_status_list)

for unique_rt in identified_rt_rt_status_list:
	freq = total_rt2rt_list.count(unique_rt)
	freq_rt2rt_list.append((unique_rt, freq))
	
print("RT Addresses and Message Frequency for RT2RT Status Words")	
for item in freq_rt2rt_list:
	print(item)	


# Grab Data words following status messages  where RT Address is one we want
RT_add = 15 #Address of most frequent status message sender

status_and_data = mil_bus_dump.iloc[:,5:]
data = []

#pert_data = status_and_data[convert_hex_rt_addr(status_and_data.iloc[:,0]) == RT_add]

"""
for row in status_and_data.index:
    curr_row = status_and_data.iloc[row]
    addr = curr_row[0]
    conv_addr = convert_hex_rt_addr(addr)
    if conv_addr == RT_add:
        data.append(curr_row)
"""	
# Locate Remote Terminal Addresses in Command Words
"""
command_words = mil_bus_dump.iloc[:,3]

command_words_valid = command_words[command_words <= 65536]
print("First Command Word: ",command_words_valid.iloc[0])

command_words_valid = pd.Series(command_words_valid, dtype=str)
def get_num_data_words(command_word):
    bin_word = bin(int(command_word, 16))
    rt_addr_bin = bin_word[2:5]
    num_data_words = int(bin_word[11:16],2)
    print(bin_word)
    return int(rt_addr_bin,2), num_data_words 
	

rt,num_words = get_num_data_words(command_words_valid.iloc[0])
print("Remote Terminal: ",rt)
print("Number of Data Words: ",num_words)	

	
"""