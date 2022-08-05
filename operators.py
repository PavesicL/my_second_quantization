#!/usr/bin/env python3

import numpy as np
import bitwise_ops as bo

"""
README:
The OPERATOR class handles the initialization of operators. Its method apply_to_bitstring defines how a given operator acts on
an integer representing an basis state in the occupation basis. The ordering of sites is (0, UP), (0, DOWN), (1, UP), (1, DOWN), ...

The OPERATOR_STRING class acts as a container for multiple operators. It knows how to apply the whole string of operators to a 
given integer. 

The STATE class is a container for a state. Vector is a np.array of amplitudes and basis is a np.array of integers representing 
the basis. The add_amplitude_to_vector method is used when applying an operator to the state. 

The apply function gets and operator string type and applies it to a given state. 
"""
###################################################################################################

NONE = (None, None) #used when the application of an operator to a state returns 0.

class OPERATOR:

	def __init__(self, name : str, site : int, spin : str = "", **quantum_numbers) -> None:

		self.name = name
		self.site = site
		self.spin = spin

		self.quantum_numbers = quantum_numbers #a dictionary of all other quantum numbers 

	def apply_to_bitstring(self, m : int, N : int) -> (int, int):
		
		offset = bo.offset(self.site, self.spin, N)

		if self.name == "c":
			if not bo.bit(m, offset):
				return NONE
			else:
				return ( bo.fermionic_prefactor(m, offset, N), bo.flipBit(m, offset) )

		if self.name == "cdag":
			if bo.bit(m, offset):
				return NONE
			else:
				return ( bo.fermionic_prefactor(m, offset, N), bo.flipBit(m, offset) )

		if self.name == "n":
			return ( bo.bit(m, offset), m)

		if self.name == "Sz":
			"""
			Here spin = "", so offset is given for the down spin site.
			"""
			return ( 0.5 *( bit(m, offset+1) - bit(m, offset) ), m )

		if self.name == "Sp" or self.name == "S+":
			pass

		if self.name == "Sm" or self.name == "S-":
			pass

	def __str__(self):
		return f"{self.name}({self.site}, {self.spin})"		

class OPERATOR_STRING:

	def __init__(self, *operators : list) -> None:
		self.operators = operators

	def apply_string_to_bitstring(self, m : int, N : int) -> (int, int):
		"""
		Applies the whole string of operators to the bitstring.
		"""
		pref = 1
		for operator in reversed(self.operators):
			prefactor, m = operator.apply_to_bitstring(m, N)
			if (prefactor, m) == NONE:
				return NONE			
			pref *= prefactor	
		return (pref, m)

def apply_string_to_bitstring(*operators, m : int, N : int) -> (int, int):
	"""
	Applies a string of operators to the bitstring.
	"""
	pref = 1
	for operator in reversed(operators):
		prefactor, m = operator.apply_to_bitstring(m, N)
		if (prefactor, m) == NONE:
			return NONE			
		pref *= prefactor	
	return (pref, m)

###################################################################################################

class STATE:

	def __init__(self, vector, basis, N):
		self.vector = np.array(vector)
		self.basis = np.array(basis)
		self.N = N # number of sites on which the state is defined.

	def add_amplitude_to_vector(self, amplitude : float, basis_state : int) -> None:
		"""
		Adds amplitude to the basis_state in state. 
		Finds the index of the basis_state in the basis and adds the amplitude to that site.	
		"""
		ndx = find_index(basis_state, self.basis)
		self.vector[ndx] += amplitude
		
	def __add__(self, other):
		if np.array_equal(self.basis, other.basis) and self.N == other.N:
			return STATE(self.vector + other.vector, self.basis, self.N)
		else:
			raise Exception("Error when adding two states - the basis do not match!")

###################################################################################################

def find_index(basis_state : int, basis : np.array) -> int:
	"""
	Finds the index of a basis_state in the basis. The basis has to be sorted!
	"""
	return np.searchsorted(basis, basis_state)

def apply(operator_string : OPERATOR_STRING, state : STATE) -> STATE:
	"""
	Applies the string of operators to the state.		
	"""
	#Initialize a new empty state in the same basis
	new_state = STATE( vector = np.zeros( len(state.basis) ), basis = state.basis, N = state.N)

	#For every element in the initial state, act on it with the operator string and add the result to the new state. 
	for i in range(len(state.basis)):
		amplitude = state.vector[i]
		basis_state = state.basis[i]

		prefactor, new_basis_state = operator_string.apply_string_to_bitstring(basis_state, state.N)
		if (prefactor, new_basis_state) != NONE:
			new_state.add_amplitude_to_vector(amplitude * prefactor, new_basis_state) 

	return new_state	

###################################################################################################
#TEST
if 0:
	# single site test

	op1 = OPERATOR(name = "cdag", site = 0, spin = "UP")
	op2 = OPERATOR(name = "c", site = 0, spin = "UP")
	op3 = OPERATOR(name = "n", site = 0, spin = "UP")

	basis = [0, 1, 2, 3,]
	vector =[1, 10, 100, 1000]

	bb = STATE(vector, basis, 1)
	print(bb.vector)
	print([bin(i) for i in bb.basis])

	op = OPERATOR_STRING( op1, op2 )
	res = apply(op, bb)

	print(res.vector)
	print(res.basis)

	print("AAAA")

	op = OPERATOR_STRING( op3 )
	res = apply(op, bb)

	print(res.vector)
	print(res.basis)

if 0:
	# two site test
	N = 2 

	op1 = OPERATOR(name = "cdag", site = 0, spin = "UP")
	op2 = OPERATOR(name = "c", site = 1, spin = "UP")

	basis = [i for i in range(16)]
	vector =[i+1 for i in range(16)]

	bb = STATE(vector, basis, N)
	print(bb.vector)
	print([format(m, "0{}b".format(2*N)) for m in bb.basis])

	op = OPERATOR_STRING( op1, op2 )
	res = apply(op, bb)

	print(res.vector)
	for i in range(len(res.basis)):
		print( format(res.basis[i], "0{}b".format(2*N)), res.vector[i] )

