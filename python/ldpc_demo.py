import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message
# n = 15
# d_v = 4
# d_c = 5
# snr = 20
# H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
# k = G.shape[1]
# v = np.random.randint(2, size=k)
# y = encode(G, v, snr)
# d = decode(H, y, snr)
# x = get_message(G, d)
# assert abs(x - v).sum() == 0



# def rotate_row(row):
# 	result = np.zeros_like(row, dtype=np.uint64)
# 	# print(row)
# 	# print(row[3])
# 	# print(type(row[3]))
# 	lsb = int(row[3]) & 0x01
# 	result[3] = ((uint16(row[3])>>1) | ((uint16(row[2]) & 1)<<15))
# 	result[2] = ((uint16(row[2])>>1) | ((uint16(row[1]) & 1)<<15))
# 	result[1] = ((uint16(row[1])>>1) | ((uint16(row[0]) & 1)<<15))
# 	result[0] = ((uint16(row[0])>>1) | (         lsb<<15))
# 	return result

def rotate_row(row):
	result = np.zeros_like(row, dtype=np.uint64)
	if(len(row.shape)==1):
		t = row.shape
		cols = t[0]
		lsb = uint16(row[cols-1]) & 0x01
		for j in range(cols-1, 0, -1):
			result[j] = ( ((uint16(row[j-1]) & 1)<<15) | (uint16(row[j])>>1))
		result[0] = (uint16(row[0])>>1) | lsb<<15
	else:
		rows, cols = row.shape
		for i in range(rows):
			lsb = uint16(row[i, cols-1]) & 0x01
			for j in range(cols-1, 0, -1):
				result[i, j] = ( ((uint16(row[i, j-1]) & 1)<<15) | (uint16(row[i,j])>>1))
			result[i, 0] = (uint16(row[i,0])>>1) | lsb<<15
	return result

def rotate_row_bounded(a):
	result = np.zeros_like(a, dtype=np.uint64)
	for i in range(int(len(a))):
		lsb = uint16(a[i]) & 0x01
		result[i] = (uint16(a[i])>>1) | lsb<<15
	return result

# def rotate_16bits_in_row(a):
# 	result = np.zeros_like(a, dtype=np.uint8)
# 	for i in range(int(len(a)/2)):
# 		lsb = a[2*i+1] & 0x01
# 		result[2*i+1] = a[2*i+1]>>1 | (a[2*i] & 0x01)<<7
# 		result[2*i] = a[2*i]>>1 | lsb<<7
# 	return result

def bounded_rotate_4x(x):
	return rotate_row_bounded(rotate_row_bounded(rotate_row_bounded(rotate_row_bounded(x))))

def unbounded_rotate_4x(x):
	return rotate_row(rotate_row(rotate_row(rotate_row(x))))

def get_ldpc_code1_G_uint64():
	W = np.zeros([64, 4], dtype=np.uint64)
	W[0,:]  = [0x0e69, 0x166b, 0xef4c, 0x0bc2]
	W[16,:] = [0x7766, 0x137e, 0xbb24, 0x8418]
	W[32,:] = [0xc480, 0xfeb9, 0xcd53, 0xa713]
	W[48,:] = [0x4eaa, 0x22fa, 0x465e, 0xea11]
	print("line40: " + "{}".format(type(W[0])))
	# print("line41: " + "{}".format(type(W[i-1])))
	for i in range(1,16):
		W[i] = unbounded_rotate_4x(W[i-1])
		W[i+16] = unbounded_rotate_4x(W[i-1+16])
		W[i+32] = unbounded_rotate_4x(W[i-1+32]) #New rotation. Same for H. G*H verify. pyldpc, custom impl, plop2 producer/consumer.
		W[i+48] = unbounded_rotate_4x(W[i-1+48])
	# If M=16, 4M=64 bits => 8 Bytes. => 4 uint16s
	# I_4 = np.zeros([64, 4], dtype=np.uint64)
	I_64bits = np.eye(64, dtype=np.uint64)
	G = np.concatenate((I_64bits, bittify16(W)), axis=1)
	return G


# def get_ldpc_code1_G():
# 	G = np.zeros([64, 8], dtype=np.uint8)
# 	G[0,:]  = [0x0e, 0x69, 0x16, 0x6b, 0xef, 0x4c, 0x0b, 0xc2]
# 	G[16,:] = [0x77, 0x66, 0x13, 0x7e, 0xbb, 0x24, 0x84, 0x18]
# 	G[32,:] = [0xc4, 0x80, 0xfe, 0xb9, 0xcd, 0x53, 0xa7, 0x13]
# 	G[48,:] = [0x4e, 0xaa, 0x22, 0xfa, 0x46, 0x5e, 0xea, 0x11]
# 	for i in range(1,16):
# 		G[i] = rotate_16bits_in_row(G[i-1])
# 		G[i+16] = rotate_16bits_in_row(G[i-1+16])
# 		G[i+32] = rotate_16bits_in_row(G[i-1+32])
# 		G[i+48] = rotate_16bits_in_row(G[i-1+48])
# 	return G


# M = 16
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

# def phi_unused(n):
# 	global Im
# 	result = np.zeros_like(Im, dtype=np.uint8)
# 	for i in range(16):
# 		rotated_row = Im[i]
# 		for m in range(n):
# 			rotated_row = rotate_16bits_in_row(rotated_row)
# 		result[i] = rotated_row
# 	return result

def get_ldpc_code1_H_uint64():
	h1 = np.concatenate(( np.bitwise_xor(Im, phi(7)), phi(2), phi(14), phi(6), np.zeros([16, 1], np.uint16), phi(0), phi(13), Im ), axis=1)
	h2 = np.concatenate(( phi(6), np.bitwise_xor(Im, phi(15)), phi(0), phi(1), Im, np.zeros([16, 1], np.uint16), phi(0), phi(7) ), axis=1)
	h3 = np.concatenate(( phi(4), phi(1), np.bitwise_xor(Im, phi(15)), phi(14), phi(11), Im, np.zeros([16, 1], np.uint16), phi(3) ), axis=1)
	h4 = np.concatenate(( phi(0), phi(1), phi(9), np.bitwise_xor(Im, phi(13)), phi(14), phi(1), Im, np.zeros([16, 1], np.uint16) ), axis=1)
	H = np.concatenate((h1, h2, h3, h4), axis=0)
	return bittify16(H)

# def get_ldpc_code1_H():
# 	h1 = np.concatenate(( np.bitwise_xor(Im, phi(7)), phi(2), phi(14), phi(6), np.zeros([16, 2], np.uint8), phi(0), phi(13), Im ), axis=1)
# 	h2 = np.concatenate(( phi(6), np.bitwise_xor(Im, phi(15)), phi(0), phi(1), Im, np.zeros([16, 2], np.uint8), phi(0), phi(7) ), axis=1)
# 	h3 = np.concatenate(( phi(4), phi(1), np.bitwise_xor(Im, phi(15)), phi(14), phi(11), Im, np.zeros([16, 2], np.uint8), phi(3) ), axis=1)
# 	h4 = np.concatenate(( phi(0), phi(1), phi(9), np.bitwise_xor(Im, phi(13)), phi(14), phi(1), Im, np.zeros([16, 2], np.uint8) ), axis=1)
# 	H = np.concatenate((h1, h2, h3, h4), axis=0)
# 	return H

def get_ldpc_matrices():
	return (get_ldpc_code1_G(), get_ldpc_code1_H())

def bittify16(I):
	sz = I.shape
	O = np.zeros([sz[0], 16*sz[1]], dtype=np.uint64)
	# print(sz)
	# print(O.shape)
	for i in range(sz[0]):
		# print(i)
		for j in range(16*sz[1]):
			bit_number = uint16(j)%16
			O[i][j] = uint16(uint16(I[i][uint16(j/16)]) & (1<<(bit_number))) >> bit_number
			# print(j)
	return O

# def bittify(I):
# 	sz = I.shape
# 	O = np.zeros([sz[0], 8*sz[1]], dtype=np.uint8)
# 	# print(sz)
# 	# print(O.shape)
# 	for i in range(sz[0]):
# 		# print(i)
# 		for j in range(8*sz[1]):
# 			bit_number = int(j)%8
# 			O[i][j] = (I[i][int(j/8)] & 1<<(bit_number)) >> bit_number
# 			# print(j)
# 	return O


## Default code
def newmain():
	print("Running example code...")
	n = 15
	d_v = 4
	d_c = 5
	snr = 20
	H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
	print(G.shape)
	print(H.shape)
	print("Type(G): {0}".format(type(G)))
	print("Type(H): {0}".format(type(H)))
	print("Type(G[0][0]): {0}".format(type(G[0][0])))
	print("Type(H[0][0]): {0}".format(type(H[0][0])))

	k = G.shape[1]
	v = np.random.randint(2, size=k)
	y = encode(G, v, snr)
	d = decode(H, y, snr)
	x = get_message(G, d)
	assert abs(x - v).sum() == 0


	## Custom code
	print("\n\nRunning custom code...")
	Gt = get_ldpc_code1_G_uint64()
	Ht = get_ldpc_code1_H_uint64()
	G = np.transpose(np.concatenate((np.identity(64, dtype=np.uint8), bittify(G_unit8)), axis=1))
	# H = bittify(H_uint8)
	H = np.transpose(H)
	print(G.shape)
	print(H.shape)
	# map(np.uint64, G)
	# map(np.uint64, H)
	print("Type(G): {0}".format(type(G)))
	print("Type(H): {0}".format(type(H)))
	print("Type(G): {0}".format(type(G[0][0])))
	print("Type(H): {0}".format(type(H[0][0])))
	# print("Type(G): {0}".format(type(G_unit8[0][0])))
	# print("Type(H): {0}".format(type(H_uint8[0][0])))
	#print(G[48:79,16:47])
	#sau=np.unique(H.sum(0))
	#sbu=np.unique(H.sum(1))
	#print(H.sum(0))
	#print(H.sum(1))
	#sr = sau * sbu
	#print("type(sr): ", type(sr))
	#print("sr.shape: ", sr.shape)
	#print(sr)
	# G_shape = G.shape

	v = np.random.randint(2, size=64)

	snr=20
	y = encode(G, v, snr)
	d = decode(H, y, snr)
	x = get_message(G, d)
	assert abs(x - v).sum() == 0

if __name__ == '__main__':
    newmain()
