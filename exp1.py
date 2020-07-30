from tls_problems import TLS_Problem
import numpy as np

if __name__ == '__main__':
	
	BA = np.array([[1, 0, 1, 0],
				   [0, 1, 0, 1],
				   [5, 4, 5, 4],
				   [3, 2, 3, 2],
				   [1, 1, 0, 0]])
	
	problem = TLS_Problem.from_ba(BA, 2)
	core_problem = problem.extract_core_problem()

	print(problem.spec_property)
	print(problem.problem_class)
	print()

	print("X'_a:")
	print(problem.solution_closed())

	print()

	print("X'_b:")
	print(problem.solve_corrected())

	print()

	print("X*_a:")
	print(problem.reconstruct_original_solution(core_problem.solution_closed()))

	print()

	print("X*_b:")
	print(problem.reconstruct_original_solution(core_problem.solve_corrected()))

	print()

	print("||I_2 - X'||_F: %f" % np.linalg.norm(np.identity(2) - problem.solution_closed(), "fro"))
	print("||I_2 - X'||_2: %f" % np.linalg.norm(np.identity(2) - problem.solution_closed(), 2))

