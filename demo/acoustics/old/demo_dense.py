# # Solve dense system
# A_dense = np.identity(M*N) - \
#         ko**2 * np.matmul(np.diag(MR.reshape(M*N, 1, order='F')[:, 0]), MAT)

# it_count = 0

# sol1, info1 = gmres(A_dense, eInc, tol=1e-4, callback=iteration_counter)
# print("The linear system was solved in {0} iterations".format(it_count))

# circ_dense = np.zeros((M*N, M*N), dtype=np.complex128)
# for i in range(0, N-1):
#     i_temp = i*M
#     for j in range(0, N-1):
#         j_temp = j*M
#         temp_vec = MAT[i_temp, j_temp:j_temp+M]
#         c_temp = np.zeros(M, dtype='complex128')
#         c_temp[0] = temp_vec[0]
#         for ii in range(1, M):
#             c_temp[ii] = (M - ii) / M * temp_vec[ii] \
#                     + ii/M * temp_vec[(M - 1) - ii + 1]
#         circ_dense[i_temp:(i+1)*M, j_temp:(j+1)*M] = \
#             toeplitz(c_temp, c_temp)

# P_dense = np.identity(M*N) - \
#         ko**2 * np.matmul(np.diag(MR.reshape(M*N, 1, order='F')[:, 0]), circ_dense)


# mvp_eval = mvp_domain(solp, opCirc, M, N, MR)

# EINC = np.zeros((M * N, 1), dtype=np.complex128)
# EINC = np.exp(1j * ko * (np.real(x)*dInc[0] + np.imag(x*dInc[1])))

# E_tot = solp.reshape(M, N, order='F')

# E_tot = EINC.reshape(M, N, order='F') \
#     - mvp_eval.reshape(M, N, order='F') \
#     + solp.reshape(M, N, order='F')

# error_l2 = np.linalg.norm(u_exact - E_tot) / np.linalg.norm(u_exact)
# print('error = ', error_l2)


# it_count = 0
# start = time.time()
# solp, info = gmres(A, eInc, M=P_dense, tol=1e-4, callback=iteration_counter)
# print("The linear system was solved in {0} iterations".format(it_count))
# end = time.time()
# print('Solve time = ', end-start,'s')

# print('Relative residual = ',
#       np.linalg.norm(mvp(solp)-eInc)/np.linalg.norm(eInc))