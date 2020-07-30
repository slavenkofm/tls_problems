from tls_problems import TLS_Problem
import numpy as np

if __name__ == '__main__':
	seed = 15
	exp_id = 3

	m, n, d = 700, 5, 4
	mu, sigma = 0, 40

	norm = "fro"

	rng = np.random.default_rng(seed=seed)

	A = rng.integers(low=0, high=50, size=(m, n))
	X = rng.integers(low=-5, high=6, size=(n, d))

	B = A @ X

	BA = np.hstack((B, A))

	noise_norms = []
	BA_norms = []
	true_BA_norms = []

	deviation_X_dash_a = []
	deviation_X_dash_b = []
	deviation_X_star_a = []
	deviation_X_star_b = []

	rng = np.random.default_rng(seed=seed+sigma)

	noise = rng.normal(mu, sigma, (m, n+d))

	for i in range(n+d, m):
		problem = TLS_Problem.from_ba(BA[:i]+noise[:i], n)
		true_BA_norms.append(np.linalg.norm(BA[:i], norm))
		noise_norms.append(np.linalg.norm(noise[:i], norm))
		BA_norms.append(np.linalg.norm(BA[:i]+noise[:i], norm))

		deviation_X_dash_a.append(np.linalg.norm(problem.solution_closed()-X, norm))
		deviation_X_dash_b.append(np.linalg.norm(problem.solve_corrected()-X, norm))

		core_problem = problem.extract_core_problem()

		deviation_X_star_a.append(np.linalg.norm(problem.reconstruct_original_solution(core_problem.solution_closed())-X, norm))
		deviation_X_star_b.append(np.linalg.norm(problem.reconstruct_original_solution(core_problem.solve_corrected())-X, norm))

		if (i % 100 == 0):
			print(i)

	with open("exp2_%s_noise_norms.txt"%exp_id, "w") as out_file:
		for i in noise_norms:
			print(i, file=out_file)

	with open("exp2_%s_ba_norms.txt"%exp_id, "w") as out_file:
		for i in BA_norms:
			print(i, file=out_file)

	with open("exp2_%s_true_ba_norms.txt"%exp_id, "w") as out_file:
		for i in true_BA_norms:
			print(i, file=out_file)

	with open("exp2_%s_deviation_X_dash_a.txt"%exp_id, "w") as out_file:
		for i in deviation_X_dash_a:
			print(i, file=out_file)

	with open("exp2_%s_deviation_X_dash_b.txt"%exp_id, "w") as out_file:
		for i in deviation_X_dash_b:
			print(i, file=out_file)

	with open("exp2_%s_deviation_X_star_a.txt"%exp_id, "w") as out_file:
		for i in deviation_X_star_a:
			print(i, file=out_file)

	with open("exp2_%s_deviation_X_star_b.txt"%exp_id, "w") as out_file:
		for i in deviation_X_star_b:
			print(i, file=out_file)

	with open("exp2_%s_params.txt"%exp_id, "w") as out_file:
		print("m = %d, n = %d, d = %d"%(m,n,d), file=out_file)
		print("mu = %f, sigma = %f"%(mu, sigma), file=out_file)
		print("norm = %s"%norm, file=out_file)
		print("seed = %s"%seed, file=out_file)

