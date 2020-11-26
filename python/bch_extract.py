import random
import arctic
import numpy as np
from arctic import plop_utils as pu
from textwrap import wrap

f = open("/home/kaushik/plop2_tests/bch_randomized.txt.0", "r")
fdata = f.read()
f.close()

start_seq_str = np.binary_repr(0xEB90, width=16)
tail_seq_str = ''.join([np.binary_repr(x, width=16) for x in [0xC5C5, 0xC5C5, 0xC5C5, 0xC579]])

k=56
n=63
P = pu.get_bch_P()
start_index = fdata.find(start_seq_str)
while start_index!=-1:
	c_m = fdata[(start_index+16):(start_index+16+k)]
	c_p = fdata[(start_index+16+k):(start_index+16+k+7)]
	xi = np.array(list(c_m), dtype=np.uint64)
	pi = np.array(list(c_p), dtype=np.uint64)

	
	prn = pu.bit_transition_generator(k)
	prn = np.packbits(prn, 0)
	xc = [chr(a^b) for a,b in zip(np.packbits(xi,0), prn)] # 

	# xc = [chr(a) for a in np.packbits(xi,0)] # No prn
	print(xc)
	p_calc = np.mod(np.matmul(xi, P), 2)
	if any(p_calc!=pi):
		print("PRTY NOT OK")
		print("Calculated: ", p_calc)
		print("Received  : ", pi)

	t = fdata[(start_index+16+k+8):(start_index+16+k+8+64)]
	if not np.array_equal(tail_seq_str, t):
		print("TAIL NOT OK")

	start_index = fdata.find(start_seq_str, start_index+1)


