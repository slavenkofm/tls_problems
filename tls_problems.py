import numpy as np
from math import isclose
from enum import Enum
from scipy.linalg import block_diag

class TLS_Problem(object):
	SPECIAL_PROPERTIES = Enum('property', 'pretty_pivot long_tail general')
	PROBLEM_CLASSES = Enum('problem_class', 'F1_ranks F1_properties F2 F3 S')

	sigma_threshold = 10**(-6)

	def __init__(self, ba, u, s, v, n, sigma_threshold = None):
		
		#by default the static sigma_threshold is used.
		#If the threshold is specified, then assign it to the instance variable
		if sigma_threshold:
			self.sigma_threshold = sigma_threshold

		#data matrix
		self.ba = ba

		#calculating and assigning dimensions
		d = ba.shape[1] - n

		self.m = ba.shape[0]
		self.n = n
		self.d = d

		#extracting the left- and right-hand sides
		self.a = ba.T[self.d:].T
		self.b = ba.T[:self.d].T

		#assigning the SVD factors
		self.u = u
		self.s = s
		self.v = v

		#calculating and assigning the values for e-q partitioning
		e = self.calc_e()
		q = self.calc_q()

		self.e = e
		self.q = q

		#detecting special properties
		self.spec_property = self.det_property()

		#calculating e-q partitioning of the data matrix
		self.v_left = v.T[:n-q].T
		self.v_right = v.T[n-q:].T

		self.v1 = self.v_right[:d]
		self.v2 = self.v_right[d:]

		self.v11 = self.v_left[:d]
		self.v21 = self.v_left[d:]

		self.v12 = self.v1.T[:e+q].T
		self.v22 = self.v2.T[:e+q].T

		self.v13 = self.v1.T[e+q:].T
		self.v23 = self.v2.T[e+q:].T

		#determining the problem class
		self.problem_class = self.det_problem_class()

	#constructing a problem from the given data matrix and number of columns in matrix A
	@staticmethod
	def from_ba(ba, n, sigma_threshold = None):
		u, s, v = np.linalg.svd(ba)
		v = v.T

		return TLS_Problem(ba, u, s, v, n, sigma_threshold)

	#constructing the problem from the given svd factors of the data matrix BA and
	#number of columns in matrix A
	@staticmethod
	def from_svd(u, s, v, n, sigma_threshold = None):
		ba = TLS_problem.reconstruct_sigma(s, (u.shape[0], v.shape[1]))

		return TLS_Problem(ba, u, s, v, n, sigma_threshold)

	#reconstructing the matrix sigma in SVD from the given singular values and target dimensions
	#TODO: checking the dimensions are consistent with the number of given singular values
	@staticmethod
	def reconstruct_sigma_static(s, dims):
		m, n = dims[0], dims[1]

		sigma = np.diag(s)
		sigma = np.vstack((sigma, np.zeros((m-len(s), len(s))))) #adding zero rows
		sigma = np.hstack((sigma, np.zeros((m, n-len(s))))) #adding zero columns

		return sigma

	#reconstructing the factor sigma for the data matrix BA
	def reconstruct_sigma(self):
		return(TLS_Problem.reconstruct_sigma_static(self.s, self.ba.shape))

	#calculating the value e (e-q numbers) TODO: rewrite considering the numeric rank
	def calc_e(self):
		n = self.n
		d = self.d

		if not isclose(self.s[n-1], self.s[n], abs_tol = self.sigma_threshold):
			return 0

		e = 1

		for i in range(n+1, d+n):
			if not isclose(self.s[n-1], self.s[i], abs_tol = self.sigma_threshold):
				break
			e += 1

		return e

	#calculate the value e (e-q numbers) TODO: rewrite considering the numeric rank
	def calc_q(self):
		n = self.n

		if not isclose(self.s[n-1], self.s[n], abs_tol = self.sigma_threshold):
			return 0
		
		q = 1

		for i in range(n-2, -1, -1):
			if not isclose(self.s[n-1], self.s[i], abs_tol = self.sigma_threshold):
				break

			q += 1

		return q

	#detecting special properties of the problem
	def det_property(self):
		if self.e == 0:
			return TLS_Problem.SPECIAL_PROPERTIES.pretty_pivot
		elif self.e == self.d:
			return TLS_Problem.SPECIAL_PROPERTIES.long_tail
		else:
			return TLS_Problem.SPECIAL_PROPERTIES.general

	#determining the problem class
	def det_problem_class(self):
		sigma_threshold = self.sigma_threshold

		e = self.e
		d = self.d

		#checking the matrix ranks condition for F1
		if (np.linalg.matrix_rank(self.v12, sigma_threshold) == e and 
		np.linalg.matrix_rank(self.v13, sigma_threshold) == d-e):
			return TLS_Problem.PROBLEM_CLASSES.F1_ranks

		#if the matrix ranks condition for F1 is not fulfilled due to numerical effects, 
		# but the problem has a special property, we still consider it as belonging to F1
		elif self.spec_property != TLS_Problem.SPECIAL_PROPERTIES.general:
			return TLS_Problem.PROBLEM_CLASSES

		#checking the matrix ranks condition for F2
		elif np.linalg.matrix_rank(self.v12, sigma_threshold) > e and np.linalg.matrix_rank(self.v13, sigma_threshold) == d-e:
			return TLS_Problem.PROBLEM_CLASSES.F2

		#checking the matrix rank condition for F3
		elif np.linalg.matrix_rank(self.v12, sigma_threshold) > e and np.linalg.matrix_rank(self.v13, sigma_threshold) < d-e:
			return TLS_Problem.PROBLEM_CLASSES.F3

		else:
			raise RuntimeError("Troubles with determining the problem class")

	#calculating the TLS correction in the pretty pivot case
	def pretty_pivot_correction(self):
		return -self.u @ self.reconstruct_sigma() @ np.hstack((np.zeros((self.n+self.d, self.n)), self.v_right)).T

	#calculating the TLS corrected problem in the pretty pivot case
	def pretty_pivot_corrected(self):
		return self.u @ self.reconstruct_sigma() @ np.hstack((self.v_left, np.zeros((self.n+self.d, self.d)))).T

	#calculating the TLS solution in the pretty pivot case using the closed formula
	def pretty_pivot_solution_closed(self):
		return np.linalg.solve(self.v1.T, -self.v2.T).T

	#calculating the TLS correction in general case
	def general_correction(self):
		Q, R = np.linalg.qr(self.v1.T, mode = "complete")
		Q2 = Q.T[self.q:].T

		return  -self.ba @ self.v_right @ Q2 @ Q2.T @ self.v_right.T

	#calculating the TLS corrected problem in general case
	def general_corrected(self):
		Q, R = np.linalg.qr(self.v1.T, mode = "complete")
		Q2 = Q.T[self.q:].T

		return self.ba @ (np.identity(self.d + self.n) - self.v_right @ Q2 @ Q2.T @ self.v_right.T)

	#claculating the TLS solution in general case
	def general_solution_closed(self):
		X = np.linalg.lstsq(self.v1.T, -self.v2.T, rcond=None)[0]
		return X.T

	#calculating the TLS correction in the long tail case
	def long_tail_correction(self):
		return self.general_correction()

	#calculating the TLS corrected problem in the long tail case
	def long_tail_corrected(self):
		return self.general_corrected()

	#calculating the TLS solution in the long tail case using the closed formula
	def long_tail_solution_closed(self):
		return self.general_solution_closed()

	#calculating the correction automatically choosing the appropriate method
	def correction(self):
		#So far we focus on F1 TLS problems; if the problem is not from F1, raise an exception
		if not (self.problem_class == TLS_Problem.PROBLEM_CLASSES.F1_ranks or 
			self.problem_class == TLS_Problem.PROBLEM_CLASSES.F1_properties):
			raise NotImplementedError
		
		#if the problem is from F1, choose the appropriate method to calculate the correction
		#based on the special property
		else:
			if self.spec_property == TLS_Problem.SPECIAL_PROPERTIES.pretty_pivot:
				return self.pretty_pivot_correction()
			elif self.spec_property == TLS_Problem.SPECIAL_PROPERTIES.long_tail:
				return self.long_tail_correction()
			else:
				return self.general_correction()

	#calculating the corrected problem automatically choosing the appropriate method
	def corrected(self):
		#So far we focus on F1 TLS problems; if the problem is not from F1, raise an exception
		if not (self.problem_class == TLS_Problem.PROBLEM_CLASSES.F1_ranks or 
			self.problem_class == TLS_Problem.PROBLEM_CLASSES.F1_properties):
			raise NotImplementedError
		
		#if the problem is from F1, choose the appropriate method to calculate the corrected problem
		#based on the special property
		else:
			if self.spec_property == TLS_Problem.SPECIAL_PROPERTIES.pretty_pivot:
				return self.pretty_pivot_corrected()
			elif self.spec_property == TLS_Problem.SPECIAL_PROPERTIES.long_tail:
				return self.long_tail_corrected()
			else:
				return self.general_corrected()

	#calculating the TLS solution from the closed formula automatically choosing the appropriate method
	def solution_closed(self):
		#So far we focus on F1 TLS problems; if the problem is not from F1, raise an exception
		if not (self.problem_class == TLS_Problem.PROBLEM_CLASSES.F1_ranks or 
			self.problem_class == TLS_Problem.PROBLEM_CLASSES.F1_properties):
			raise NotImplementedError
		
		#if the problem is from F1, choose the appropriate method to calculate the corrected problem
		#based on the special property
		else:
			if self.spec_property == TLS_Problem.SPECIAL_PROPERTIES.pretty_pivot:
				return self.pretty_pivot_solution_closed()
			elif self.spec_property == TLS_Problem.SPECIAL_PROPERTIES.long_tail:
				return self.long_tail_solution_closed()
			else:
				return self.general_solution_closed()

	#calculating the TLS solution by solving the corrected problem
	def solve_corrected(self):
		system = self.corrected()
		return np.linalg.lstsq(system.T[self.d:].T, system.T[0:self.d].T, rcond = None)[0]

	#checking whether the numeric matrix rank of B equals d
	def b_full_rank(self):
		return np.linalg.matrix_rank(self.b, tol = self.sigma_threshold) == self.d

	#printing a report on the problem and checking if the solutions given by different approaches are consistent
	def check_integrity(self):
		print("PROBLEM PROPERTIES\n")
		print("m: %d, n: %d, d: %d" % (self.m, self.n, self.d))
		print("e: %d, q: %d" % (self.e, self.q))
		print("BA:")
		print(self.ba)

		print()

		print("BA Rank: %d" % np.linalg.matrix_rank(self.ba, tol = self.sigma_threshold))
		print("B is of full rank: %s" % self.b_full_rank())
		print("Problem class: %s" % self.problem_class.name)
		print("Special property: %s" % self.spec_property.name)

		print("\n***")

		print("CORRECTION INTEGRITY \n")

		print("Manual and Automatical coincide: %s " % np.allclose(self.corrected(), self.ba + self.correction()))
		print("BA = corrected - correction: %s " % np.allclose(self.ba, self.corrected() - self.correction()))

		print("***\n")
		print("corrected - correction:")
		print(self.corrected() - self.correction())

		print("***\n")

		print("Corrected automatically: ")
		print(self.corrected())

		print("")

		print("Automatical correction: ")
		print(self.corrected() - self.ba)
		print("Norm: %f" % np.linalg.norm(self.corrected() - self.ba, 'fro'))

		print("***\n")

		print("Corrected manually: ")
		print(self.ba + self.correction())

		print("")

		print("Manual correction: ")
		print(self.correction())
		print("Norm: %f" % np.linalg.norm(self.correction(), 'fro'))

		print("***\n")

		print("SOLUTIONS INTEGRITY \n")

		system = self.corrected()
		system = [system.T[self.d:].T, system.T[0:self.d].T]

		print("A':")
		print(system[0])
		print("B':")
		print(system[1])

		print("***\n")

		print("solution_closed:")
		print(self.solution_closed())
		print("Norm: %f" % np.linalg.norm(self.solution_closed(), 'fro'))
		print("\nA'.X':")
		print(system[0] @ self.solution_closed())
		print("Deviation norm: %f" % np.linalg.norm(system[1] - system[0] @ self.solution_closed(), 'fro'))

		print("***\n")

		print("solve_corrected():")
		print(self.solve_corrected())
		print("Norm: %f" % np.linalg.norm(self.solve_corrected(), 'fro'))
		print("\nA'.X':")
		print(system[0] @ self.solve_corrected())
		print("Deviation norm: %f" % np.linalg.norm(system[1] - system[0] @ self.solve_corrected(), 'fro'))

		print("***\n")

		print("solve_corrected_pinv():")
		sol = np.linalg.pinv(system[0]) @ system[1]
		print(sol)
		print("Norm: %f" % np.linalg.norm(sol, 'fro'))
		print("\nA'.X':")
		print(system[0] @ sol)
		print("Deviation norm: %f" % np.linalg.norm(system[1] - system[0] @ sol, 'fro'))

	#preprocessing the right-hand side, part of the classic CDR algorithm
	def svd_preprocess_b(self):
		r = np.linalg.svd(self.b)[2]

		self.R = r.T #storing the R transformation from the CDR algorithm

		return self.b @ r.T

	#transforming the matrix A, part of the classic CDR algorithm
	def transform_a(self, c):
		u, v = np.linalg.svd(self.a)[0::2]

		self.P = u
		self.Q = v.T

		return u.T @ c

	#preparing the partitioning of the matrix F, part of the classic CDR algorithm
	def f_partitioning(self, f):
		s = np.linalg.svd(self.a)[1]
		slices = []

		current_sigma = s[0]

		for i in range(1, self.r):
			if isclose(current_sigma, s[i], abs_tol = self.sigma_threshold):
				continue
			else:
				slices.append(i)
				current_sigma = s[i]

		if not self.r in slices:
			slices.append(self.r)

		return np.vsplit(f, slices)

	def transform_f(self, f):
		SL = []
		SR = []


		for Ft in self.f_partitioning(f):
			SL.append(np.linalg.svd(Ft)[0])

			self.PL_nums.append((np.linalg.matrix_rank(Ft, tol = self.sigma_threshold), 
				Ft.shape[0] - np.linalg.matrix_rank(Ft, tol = self.sigma_threshold)))

		SR.extend(SL[:-1])
		SR.append(np.identity(self.n - self.r))
		SR = block_diag(*SR)

		SL = block_diag(*SL)

		self.P = self.P @ SL
		self.Q = self.Q @ SR

		return SL.T @ f

	def final_permutation(self, g, s):
		PLL = []
		PLR = []

		for dims in self.PL_nums:
			PLL.append(np.vstack((np.identity(dims[0]),
				np.zeros(dims).T)))
			PLR.append(np.vstack((np.zeros((dims[0], dims[1])),
				np.identity(dims[1]))))

		PRL = PLL[0:-1]
		PRR = PLR[0:-1]

		PLL = block_diag(*PLL)
		PLR = block_diag(*PLR)

		PL = np.hstack((PLL, PLR))

		PRR.append(np.identity(self.n - self.r))
		PRR = block_diag(*PRR)
		PRL = block_diag(*PRL)

		PRL = np.vstack((PRL, np.zeros((self.n - self.r, PRL.shape[1]))))

		PR = np.hstack((PRL, PRR))

		#print(Q)
		self.P = self.P @ PL
		self.Q = self.Q @ PR
		#print(Q)

		PR = block_diag(np.identity(self.d), PR)

		BA_final = PL.T @ np.hstack((g, TLS_Problem.reconstruct_sigma_static(s, self.a.shape))) @ PR

		return BA_final

	def calculate_strokes(self):
		self.m_stroke, self.n_stroke = 0, 0
		for i in self.PL_nums:
			self.m_stroke += i[0]

		self.n_stroke = self.m_stroke - self.PL_nums[-1][0]

		return self.m_stroke, self.n_stroke

	def extract_core_problem(self):
		self.R = None
		self.P = None
		self.Q = None

		self.r = np.linalg.matrix_rank(self.a, tol = self.sigma_threshold)

		self.PL_nums = []

		self.m_stroke = None
		self.n_stroke = None

		c = self.svd_preprocess_b()
		f = self.transform_a(c)
		g = self.transform_f(f)
		transformed_problem = self.final_permutation(g, np.linalg.svd(self.a)[1])

		self.calculate_strokes()

		return TLS_Problem.from_ba(transformed_problem[0:self.m_stroke, 0:self.d + self.n_stroke], self.n_stroke)

	def reconstruct_original_solution(self, x):
		shape = x.shape
		x_dashed = block_diag(x, np.zeros((self.n-shape[0], self.d-shape[1])))

		return self.Q @ x_dashed @ self.R.T

if __name__ == '__main__':
	'''
	BA = np.array([[0.93192074, 0.10601966, 0.51514497, 0.48264215],
 		[0.43385596, 0.41059356, 0.15480585, 0.72322137],
 		[4.78807211, 3.77942793, 4.80190364, 3.69914083],
 		[2.74568604, 2.05703417, 2.77283469, 1.97658111],
 		[0.7717684,  0.67502296, 0.26617199, 0.16942655]])
	BA = np.array([[1, 0, 1, 0],
					[0, 1, 0, 1],
					[5, 4, 5, 4],
					[3, 2, 3, 2],
					[1, 1, 0, 0]])
	'''
	rng = np.random.default_rng(seed=33)

	A = rng.integers(low=0, high=50, size=(20, 5))
	B = A
	BA = np.hstack((B, A))
	noise = rng.standard_normal(size = (20, 10))
	BA = BA + noise

	print("Noise norm: %f" % np.linalg.norm(noise, "fro"))

	problem = TLS_Problem.from_ba(ba=BA, n=5)

	problem.check_integrity()

	core_problem = problem.extract_core_problem()
	print(core_problem.ba)
	#print(core_problem.solution_closed())
	print(problem.reconstruct_original_solution(core_problem.solution_closed()))
	print(np.linalg.norm(np.identity(5)-problem.solution_closed(), "fro"))
