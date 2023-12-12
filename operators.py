#!/usr/bin/env python3

import numpy as np
import my_second_quantization.bitwise_ops as bo
import cmath

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

	def __init__(self, name : str, site : int, spin : str = "") -> None:

		self.name = name
		self.site = site
		self.spin = spin

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
			For the up site add +1.
			"""
			return ( 0.5 *( bo.bit(m, offset+1) - bo.bit(m, offset) ), m )

		if self.name == "Sp" or self.name == "S+":
			raise Exception(f"{self.name} not defined.")

		if self.name == "Sm" or self.name == "S-":
			raise Exception(f"{self.name} not defined.")

		else:
			raise Exception(f"{self.name} not defined.")

	def __str__(self):
		return f"{self.name}({self.site}, {self.spin})"

class OPERATOR_STRING:

	def __init__(self, *operators : list) -> None:

		_ops = []
		for operator in operators:
			if type(operator) == tuple or type(operator) == list:
				op = operator_form_list(list(operator))
				_ops.append(op)
			elif type(operator) == OPERATOR or type(operator) == SPIN_OPERATOR:
				_ops.append(operator)
			else:
				raise Exception("Operator not recognized. Has to be type OPERATOR or SPIN_OPERATOR or list/tuple.")

		self.operators = _ops

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

###################################################################################################
# This is a class that defines spin operators, as acting on a string in the Sz basis.
class SPIN_OPERATOR:

	def __init__(self, name : str, site : int) -> None:

		self.name = name
		self.site = site

		if not name in ("Sx", "Sy", "Sz"):
			raise Exception(f"Spin operator with unrecognized name {name}! Has to be Sx, Sy or Sz.")


	def apply_to_bitstring(self, m : int, N : int) -> (complex, int):

		offset = N - self.site #for a string of spins (Sz basis - up or down), the offset from the back is just this.

		if self.name == "Sx":
			prefactor = +1
			return ( prefactor, bo.flipBit(m, offset) )

		elif self.name == "Sy":
			state = bo.bit(m, offset) #measure the state of the bit
			prefactor = -1j if state==1 else +1j
			return ( prefactor, bo.flipBit(m, offset) )


		elif self.name == "Sz":
			state = bo.bit(m, offset) #measure the state of the bit
			prefactor = +1 if state==1 else -1
			return ( prefactor, m )

###################################################################################################

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

def operator_form_list(op_list : list) -> OPERATOR:
	"""
	This must be a list of (op : str, site : int, spin : str), for example ["cdag", 3, "UP"]
	"""
	return OPERATOR( *op_list )

###################################################################################################

class BASIS_STATE:
	"""
	A basis state contains an integer representing the occupation basis part and a dictionary of
	possible additional quantum numbers.
	"""
	def __init__(self, bitstring : int, **quantum_numbers) -> None:
		self.bitstring = bitstring
		self.quantum_numbers = quantum_numbers

		self.n = bo.countSetBits(bitstring)

	#The methods below have to be defined so that the basis can be ordered.
	#The ordering is done first by the bitstring and then by the value of the quantum numbers.
	#NOT ALL OF THOSE METHODS HAVE TO BE IMPLEMENTED, BY THE WAY.

	def __repr__(self):
		if self.quantum_numbers == {}:
				return f"{bin(self.bitstring)}"
		else:
			return f"{bin(self.bitstring)} |{str(self.quantum_numbers)[1:-1]}>"

	def __lt__(self, other):
		if self.bitstring == other.bitstring:
			for qn in self.quantum_numbers:
				if self.quantum_numbers[qn] != other.quantum_numbers[qn]:
					return self.quantum_numbers[qn] < other.quantum_numbers[qn]
		else:
			return self.bitstring < other.bitstring

	def __gt__(self, other):
		if self.bitstring == other.bitstring:
				for qn in self.quantum_numbers:
					if self.quantum_numbers[qn] != other.quantum_numbers[qn]:
						return self.quantum_numbers[qn] > other.quantum_numbers[qn]
		else:
			return self.bitstring > other.bitstring

	def __le__(self, other):
		if self.bitstring == other.bitstring:
				for qn in self.quantum_numbers:
					if self.quantum_numbers[qn] != other.quantum_numbers[qn]:
						return self.quantum_numbers[qn] <= other.quantum_numbers[qn]
		else:
			return self.bitstring <= other.bitstring

	def __ge__(self, other):
		if self.bitstring == other.bitstring:
			for qn in self.quantum_numbers:
				if self.quantum_numbers[qn] != other.quantum_numbers[qn]:
					return self.quantum_numbers[qn] >= other.quantum_numbers[qn]
		else:
			return self.bitstring >= other.bitstring

	def __eq__(self, other):
		if self.bitstring == other.bitstring:
			for qn in self.quantum_numbers:
				if self.quantum_numbers[qn] != other.quantum_numbers[qn]:
					return False
			return True #if bitstring is the same and all quantum numbers as well, the states are equal
		return False

class STATE:

	def __init__(self, vector : np.array, basis : np.array, N : int) -> None:
		self.vector = np.array(vector, dtype=complex) 	# np.array of amplitudes
		self.basis = np.array(basis, dtype=BASIS_STATE) 	# sorted np.array of BASIS_STATEs
		self.N = N 										# number of sites on which the state is defined.


	def add_amplitude_to_vector(self, amplitude : float, basis_state : int) -> None:
		"""
		Adds amplitude to the basis_state in state.
		Finds the index of the basis_state in the basis and adds the amplitude to that site.
		"""
		ndx = find_index(basis_state, self.basis)

		print("AAA", ndx)
		self.vector[ndx] += amplitude

	def __add__(self, other):
		if np.array_equal(self.basis, other.basis) and self.N == other.N:
			return STATE(self.vector + other.vector, self.basis, self.N)
		else:
			raise Exception("Error when adding two states - the basis do not match!")

	def __repr__(self):
		sorted_vec, sorted_basis = zip(*sorted(zip(self.vector, self.basis), key = lambda x : x[0]))
		res = ""
		for i in range(len(self.vector)):
			amp = self.vector[i]
			bas = self.basis[i]
			if amp != 0:
				res += f"{abs(amp)}	e^({round(cmath.phase(amp)/np.pi,3)}/pi)		{bas}\n"
		return res

	def pad_with_zeros(str, N):
		"""
		Pads the binary string with zeros so that its length is N.
		"""
		a = 1
		return
###################################################################################################

def find_index(basis_state : BASIS_STATE, basis : np.array) -> int:
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

		prefactor, new_bitstring = operator_string.apply_string_to_bitstring(basis_state.bitstring, state.N)
		if (prefactor, new_bitstring) != NONE:
			new_basis_state = BASIS_STATE( new_bitstring, **basis_state.quantum_numbers )
			new_state.add_amplitude_to_vector(amplitude * prefactor, new_basis_state)

	return new_state

def expectedValue(operator_string : OPERATOR_STRING, state : STATE) -> float:
	"""
	Calculates the expected value of an operator string to a state.
	"""
	Opsi = apply(operator_string, state)
	return np.dot( np.conjugate(state.vector), Opsi.vector )

def expectedValueQuantumNumber(quantum_number : str, state : STATE) -> float:
	"""
	Computes the expected value of a quantum number in the state.
	"""
	exVal = 0
	for i in range(len(state.vector)):
		amp = state.vector[i]
		basis_state = state.basis[i]

		exVal += np.conjugate(amp) * basis_state.quantum_numbers[quantum_number] * amp
	return exVal

def expectedValueQuantumNumberSquared(quantum_number : str, state : STATE) -> float:
	"""
	Computes the expected value of a quantum number in the state.
	"""
	exVal = 0
	for i in range(len(state.vector)):
		amp = state.vector[i]
		basis_state = state.basis[i]

		exVal += np.conjugate(amp) * (basis_state.quantum_numbers[quantum_number]**2) * amp
	return exVal

###################################################################################################
#TESTS

if 0:
	# spin operators test
	print("TESTING 1")
	opx = SPIN_OPERATOR(name = "Sx", site = 1)
	opy = SPIN_OPERATOR(name = "Sy", site = 0)
	opz = SPIN_OPERATOR(name = "Sz", site = 0)

	basis = np.array([BASIS_STATE(i) for i in range(5)])
	vector = [1, 10, 100, 1000, 10000]

	bb = STATE(vector, basis, 5)
	print(bb.vector)
	print([i for i in bb.basis])

	print("apply Sx")
	op = OPERATOR_STRING( opz )
	res = apply(op, bb)

	print(res)
	print(type(res))
	print(res.N)
	#print("apply Sx")
	#print("apply Sx")

if 0:
	# single site test

	op1 = OPERATOR(name = "cdag", site = 0, spin = "UP")
	op2 = OPERATOR(name = "c", site = 0, spin = "UP")
	op3 = OPERATOR(name = "n", site = 0, spin = "UP")
	op4 = OPERATOR(name = "Sz", site = 0)

	basis = np.array([BASIS_STATE(i) for i in range(4)])
	vector =[1, 10, 100, 1000]

	bb = STATE(vector, basis, 1)
	print(bb.vector)
	print([i for i in bb.basis])

	print("cdag c")

	op = OPERATOR_STRING( op1, op2 )
	res = apply(op, bb)

	print(res.vector)
	print(res.basis)

	print("n")

	op = OPERATOR_STRING( op3 )
	res = apply(op, bb)

	print(res.vector)
	print(res.basis)

	print("Sz")

	op = OPERATOR_STRING( op4 )
	res = apply(op, bb)

	print(res.vector)
	print(res.basis)

if 0:
	# single site with quantum numbers

	op1 = OPERATOR(name = "cdag", site = 0, spin = "DO")
	op2 = OPERATOR(name = "c", site = 0, spin = "DO")
	op3 = OPERATOR(name = "n", site = 0, spin = "DO")

	basis = np.array([BASIS_STATE(i, phi = 1, x = 3) for i in range(4)] + [BASIS_STATE(i, phi = 3, x = 0) for i in range(4)]  )
	basis = sorted(basis)
	vector =[1, 10, 100, 1000, 2, 20, 200, 2000]

	bb = STATE(vector, basis, 1)
	print(bb.vector)
	print([i for i in bb.basis])

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

