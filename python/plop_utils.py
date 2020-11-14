import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message

def rotate_row(row):
	result = np.zeros_like(row, dtype=np.uint64)
	if(len(row.shape)==1):
		t = row.shape
		cols = t[0]
		lsb = int(row[cols-1]) & 0x01
		for j in range(cols-1, 0, -1):
			result[j] = ( ((int(row[j-1]) & 1)<<15) | (int(row[j])>>1))
		result[0] = (int(row[0])>>1) | lsb<<15
	else:
		rows, cols = row.shape
		for i in range(rows):
			lsb = int(row[i, cols-1]) & 0x01
			for j in range(cols-1, 0, -1):
				result[i, j] = ( ((int(row[i, j-1]) & 1)<<15) | (int(row[i,j])>>1))
			result[i, 0] = (int(row[i,0])>>1) | lsb<<15
	return result

def rotate_row_bounded(a):
	result = np.zeros_like(a, dtype=np.uint64)
	for i in range(int(len(a))):
		lsb = int(a[i]) & 0x01
		result[i] = (int(a[i])>>1) | lsb<<15
	return result

def bounded_rotate_4x(x):
	return rotate_row_bounded(rotate_row_bounded(rotate_row_bounded(rotate_row_bounded(x))))

def unbounded_rotate_4x(x):
	return rotate_row(rotate_row(rotate_row(rotate_row(x))))

def bittify_uint64(A, select_rows=None):
	if(np.ndim(A)==1):
		result = bittify_uint64([A])[0]
	else:
		rows, cols = np.shape(A)
		result = np.zeros([rows, cols*64], dtype=np.uint64)
		if select_rows is None:
			select_rows = range(rows)
		for i in select_rows:
			temp = []
			for j in range(cols):
				temp.extend(list(np.binary_repr(A[i, j], width=64)))
				#	   Alt: list(bin(A[i,j]).replace('0b', '').zfill(64))
			result[i] = np.uint64(temp)
	return result

def get_ldpc_code2_G_uint64():
	W = np.zeros([256, 4], dtype=np.uint64)
	W[  0, :]  = [0x1D21794A22761FAE, 0x59945014257E130D, 0x74D6054003794014, 0x2DADEB9CA25EF12E]
	W[ 64, :]  = [0x60E0B6623C5CE512, 0x4D2C81ECC7F469AB, 0x20678DBFB7523ECE, 0x2B54B906A9DBE98C]
	W[128, :]  = [0xF6739BCF54273E77, 0x167BDA120C6C4774, 0x4C071EFF5E32A759, 0x3138670C095C39B5]
	W[192, :]  = [0x28706BD045300258, 0x2DAB85F05B9201D0, 0x8DFDEE2D9D84CA88, 0xB371FAE63A4EB07E]
	W = bittify_uint64(W, select_rows=[0, 64, 128, 192])
	for j in range(256):
		for i in range(1, 64):
			W[ i,     (j+i)%256 ] = W[  0][j]
			W[ i+64,  (j+i)%256 ] = W[ 64][j]
			W[ i+128, (j+i)%256 ] = W[128][j]
			W[ i+192, (j+i)%256 ] = W[192][j]
	
	I_256 = np.eye(256, dtype=np.uint64)
	G = np.concatenate((I_256, W), axis=1)
	return G

	# for i in range(1,64):
	# 	W[ i     ] = unbounded_rotate_4x( W[ i-1     ] )
	# 	W[ i+64  ] = unbounded_rotate_4x( W[ i-1+64  ] )
	# 	W[ i+128 ] = unbounded_rotate_4x( W[ i-1+128 ] ) #New rotation. Same for H. G*H verify. pyldpc, custom impl, plop2 producer/consumer.
	# 	W[ i+192 ] = unbounded_rotate_4x( W[ i-1+192 ] )
	
	

def get_ldpc_code1_G_uint64():
	W = np.zeros([64, 4], dtype=np.uint64)
	W[0,:]  = [0x0e69, 0x166b, 0xef4c, 0x0bc2]
	W[16,:] = [0x7766, 0x137e, 0xbb24, 0x8418]
	W[32,:] = [0xc480, 0xfeb9, 0xcd53, 0xa713]
	W[48,:] = [0x4eaa, 0x22fa, 0x465e, 0xea11]
	# print("line40: " + "{}".format(type(W[0])))
	# print("line41: " + "{}".format(type(W[i-1])))
	for i in range(1,16):
		W[i] = unbounded_rotate_4x(W[i-1])
		W[i+16] = unbounded_rotate_4x(W[i-1+16])
		W[i+32] = unbounded_rotate_4x(W[i-1+32]) #New rotation. Same for H. G*H verify. pyldpc, custom impl, plop2 producer/consumer.
		W[i+48] = unbounded_rotate_4x(W[i-1+48])
	# If M=16, 4M=64 bits => 8 Bytes. => 4 u int16s
	# I_4 = np.zeros([64, 4], dtype=np.uint64)
	I_64bits = np.eye(64, dtype=np.uint64)
	G = np.concatenate((I_64bits, bittify16(W)), axis=1)
	return G

Im = np.array([	[0x8000], 
		[0x4000], 
		[0x2000], 
		[0x1000], 
		[0x0800], 
		[0x0400], 
		[0x0200], 
		[0x0100], 
		[0x0080], 
		[0x0040], 
		[0x0020], 
		[0x0010], 
		[0x0008], 
		[0x0004], 
		[0x0002], 
		[0x0001]	], dtype=np.uint64)

def phi(n):
	global Im
	result = np.zeros_like(Im, dtype=np.uint64)
	for i in range(16):
		rotated_row = Im[i]
		for m in range(n):
			rotated_row = rotate_row(rotated_row)
		result[i] = rotated_row
	return result

def get_ldpc_code1_H_uint64():
	h1 = np.concatenate(( np.bitwise_xor(Im, phi(7)), phi(2), phi(14), phi(6), np.zeros([16, 1], np.uint16), phi(0), phi(13), Im ), axis=1)
	h2 = np.concatenate(( phi(6), np.bitwise_xor(Im, phi(15)), phi(0), phi(1), Im, np.zeros([16, 1], np.uint16), phi(0), phi(7) ), axis=1)
	h3 = np.concatenate(( phi(4), phi(1), np.bitwise_xor(Im, phi(15)), phi(14), phi(11), Im, np.zeros([16, 1], np.uint16), phi(3) ), axis=1)
	h4 = np.concatenate(( phi(0), phi(1), phi(9), np.bitwise_xor(Im, phi(13)), phi(14), phi(1), Im, np.zeros([16, 1], np.uint16) ), axis=1)
	H = np.concatenate((h1, h2, h3, h4), axis=0)
	return bittify16(H)

def get_ldpc_matrices():
	return (get_ldpc_code1_G(), get_ldpc_code1_H())

def bittify16(I):
	sz = np.shape(I)
	O = np.zeros([sz[0], 16*sz[1]], dtype=np.uint64)
	# print(sz)
	# print(O.shape)
	for i in range(sz[0]):
		# print(i)
		for j in range(16*sz[1]):
			bit_number = 15-(int(j)%16)
			O[i][j] = int(int(I[i][int(j/16)]) & (1<<(bit_number))) >> bit_number
			# print(j)
	return O

def bittify8(I, dtype=np.uint8):
	if(type(I)==bytes):
		I = list(I)
	sz = np.shape(I)
	# print(sz)
	# print(O.shape)
	if(np.ndim(I)==1):
		O = np.zeros(8*sz[0], dtype=dtype)
		for j in range(8*sz[0]):
			bit_number = 7 - int(j)%8
			O[j] = int(int(I[int(j/8)]) & (1<<(bit_number))) >> bit_number
	else:
		O = np.zeros([sz[0], 8*sz[1]], dtype=dtype)
		for i in range(sz[0]):
			# print(i)
			for j in range(8*sz[1]):
				bit_number = 7 - int(j)%8
				O[i][j] = int(int(I[i][int(j/8)]) & (1<<(bit_number))) >> bit_number
				# print(j)
	return O

BTG_BITS	= [	1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
				1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
			    0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1,
			    0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1,
			    0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
			    0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
			    0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
			    0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
			    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,
			    1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0,
			    0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
			    1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
			    0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0,
			    0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
			    1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
			    0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
			    # Repeats; double-buffering save time of joining arrays
			    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
				1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
			    0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1,
			    0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1,
			    0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
			    0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
			    0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
			    0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
			    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,
			    1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0,
			    0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
			    1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
			    0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0,
			    0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
			    1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
			    0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0		]

btg_counter = np.uint8(0)
def bit_transition_generator(n=1):
	global btg_counter
	if(n>256):
		print("Too long random sequence requested. n=", n)
		sys.exit(1002)
	result = BTG_BITS[btg_counter:(int(btg_counter)+n)]
	btg_counter += n
	# 0..254 valid, i.e. 255 unique values
	if btg_counter>=255: btg_counter -= 255
	return result

def extract_cltu_data_ldpc1(s):
	G = get_ldpc_code1_G_uint64()
	# H = get_ldpc_code1_H_uint64()
	# sa = list(s)
	# si = np.array([int(x) for x in sa])
	# d = decode(np.transpose(H), si, 150)
	msg = get_message(np.transpose(G), s)
	msg = msg.reshape([8,8])
	p = np.transpose([128, 64, 32, 16, 8, 4, 2, 1])
	xi = np.matmul(msg, p)
	xc = [chr(a) for a in xi]
	return (xc, xi)

def extract_cltu_data_ldpc2(s):
	G = get_ldpc_code2_G_uint64()
	# H = get_ldpc_code1_H_uint64()
	# sa = list(s)
	# si = np.array([int(x) for x in sa])
	# d = decode(np.transpose(H), si, 150)
	msg = get_message(np.transpose(G), s)
	msg = msg.reshape([32,8])
	p = np.transpose([128, 64, 32, 16, 8, 4, 2, 1])
	xi = np.matmul(msg, p)
	xc = [chr(a) for a in xi]
	return (xc, xi)
