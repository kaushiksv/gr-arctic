import random
import arctic
import numpy as np
from arctic import plop_utils as pu
from textwrap import wrap

f = open("ldpc1_test_file.txt", "r")
fdata = f.read()
f.close()

# CLTU_START_SEQUENCE_LDPC = [0x0347, 0x76C7, 0x2728, 0x95B0]
# ''.join([np.binary_repr(x, width=16) for x in CLTU_START_SEQUENCE_LDPC])
start_seq_str = '0000001101000111011101101100011100100111001010001001010110110000'

k=256
n=512
start_index = fdata.find(start_seq_str)
while start_index!=-1:
	ystr = fdata[(start_index+64):(start_index+64+n)]
	ystr = np.array(list(ystr), dtype=np.uint64)
	
	# Introduce error
	# e1 = random.randrange(64)
	# e2 = random.randrange(64)
	# ystr[e1] ^= 1
	# ystr[e2] ^= 1
	# ystr = np.array(ystr, dtype=np.str)
	# print(ystr)
	xc, xi = pu.extract_cltu_data_ldpc2(ystr)
	pnr = pu.bit_transition_generator(k)
	pnr = np.packbits(pnr, 0)
	xc = [chr(a^b) for a,b in zip(xi, pnr)]
	print(xc)
	start_index = fdata.find(start_seq_str, start_index+1)


