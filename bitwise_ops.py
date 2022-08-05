#!/usr/bin/env python3
from numba import jit


@jit
def offset(site : str, spin : str, N : int) -> int:
	"""
	Returns the index of the given site from the back. 
	Used for bitwise operations.
	If spin is an empty string (eg. for spin operators) returns the offset of the down site.
	"""
	return 2 * ( N - 1 - site ) + (1 if spin == "UP" else 0)


@jit
def fermionic_prefactor(m : int, offset : int, N : int) -> int: 
	"""
	Calculates the fermionic prefactor for an operator acting on site given with offset.
	"""
	count=0
	for i in range(offset+1, 2*N):	#count bits from offset to the end
		count+=bit(m, i)
	return -(2 * (count%2)-1)

@jit
def flipBit(m : int, offset : int) -> int:
	"""
	Flips the bit at position offset in the integer m.
	"""
	mask = 1 << offset
	return(m ^ mask)

@jit
def countSetBits(m : int) -> int: 
	"""
	Counts the number of bits that are set to 1 in a given integer.
	"""
	count = 0
	while (m): 
		count += m & 1
		m >>= 1
	return count 

@jit
def bit(m : int , offset : int) -> int:
	"""
	Returns the value of a bit at offset off in integer m.
	"""
	if m & (1 << offset):
		return 1
	else:
		return 0

@jit
def spinUpBits(m : int, N : int) -> int:
	"""
	Counts the number of spin up electrons in the state.
	"""
	count=0
	for i in range(1, 2*N, 2):
		if bit(m, i)==1:
			count+=1
	return count		

@jit
def spinDownBits(m : int, N : int) -> int:
	"""
	Counts the number of spin down electrons in the state.
	"""
	count=0
	for i in range(0, 2*N, 2):
		if bit(m, i)==1:
			count+=1	
	return count