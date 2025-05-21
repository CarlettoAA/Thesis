# implementation of example 5.2 - Bender & Kohlmann 2008
# European call otpion under non-convex borrowing constraints

import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import norm

# GLOBAL VARIABLES
# Parameters
r = 0.05  # Interest rate
mu = 0.07  # Expected return (10%)
sigma = 0.2  # Volatility (20%)
delta = 0.1  # dividends rate
S0 = 100  # Initial stock price
u = 9  # concavity param
rho = 1000  # special value
K = 110  # strike price
T = 0.5  # Time horizon (1 year)
N = 15  # Number of time steps
L = 10000  # Number of simulated paths
D = 5  # Value for Basis function; not relevant to be defined
n = 0     # just for example
q = 0.5   # just for example


def simulate_gbm(S0, mu, sigma, T, N, M, W):

    dt = T / N
    t = np.linspace(0, T, N)
    S = np.zeros((M, N))
    S[:, 0] = S0

    for i in range(1, N):
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * W[:, i])

    return t, S

# function needed for the non-convex constraint, it works componentwise
def f_con(x, u, q, rho):
    x = np.asarray(x)  # ensure x is a NumPy array
    y = np.zeros_like(x)

    # Conditions
    mask1 = x < 0
    mask2 = (x > 0) & (x <= 1)
    mask3 = (x > 1) & (x <= rho)
    mask4 = x > rho

    y[mask1] = 0
    y[mask2] = u * x[mask2]
    y[mask3] = u * x[mask3] ** (q + 1)
    y[mask4] = u * x[mask4] * q ** rho

    return y

# generator
def generator(t, s, y, z, weight):

    return r * y - (mu - r)*z/sigma - weight/10 * np.maximum(z/sigma - y - f_con(y, u, q, rho), 0)

def G_option(s, delta, K, T):

    return np.maximum(s * np.exp(-delta * T) - K, 0)

# --- Define the basis functions ---
def basis_1(x): return np.ones_like(x)
def basis_2(x): return x - S0
def basis_3(x): return (x - S0)**2
def basis_4(x): return (x - S0)**3

# --- Evaluate the basis functions on simulated paths ---
basis_functions = [
    lambda x: basis_1(x),
    lambda x: basis_2(x),
    lambda x: basis_3(x),
    lambda x: basis_4(x),
    lambda x: G_option(x, delta, K, T)
]


def orthonormalize_basis(S_tj, basis_functions):

    # Evaluate all basis functions on S_tj
    Phi = np.column_stack([f(S_tj) for f in basis_functions])  # Shape: (L, K)

    # SVD
    U, s, Vt = svd(Phi, full_matrices=False)

    # Retain columns corresponding to non-negligible singular values
    kappa_j = np.sum(s > 1e-10)
    Phi_orth = U[:, :kappa_j] * np.sqrt(L)  # scale to satisfy ⟨ψ_i, ψ_j⟩ = δ_ij

    return Phi_orth # that is L X kappa_j

def black_scholes_call_div(S, K, T, r, delta, sigma):
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-delta * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price



# MAIN CODE #################################################################
if __name__ == "__main__":

    # Set seed for reproducibility
    np.random.seed(86)

    BS_price = black_scholes_call_div(S0, K, T, r, delta, sigma)
    print("BS standard price:", BS_price)
    # STEP 1: choose a partition and a family of functions.

    # STEP 2: Stock price process
    # Run simulation
    W = np.random.standard_normal((L, N))
    t, S = simulate_gbm(S0, mu, sigma, T, N, L, W)

    #print("t", t)
    #print("Stock price simulations at time T", S[:, -1])
    #print("G_option of S_T", G_option(S[:, -1], delta, K, T))

    """
    # Plot results
    plt.figure(figsize=(10, 6))
    for i in range(L):
        plt.plot(t, S[i], lw=1)
    plt.title('Geometric Brownian Motion Simulation of Stock Prices')
    plt.xlabel('Time (Years)')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.show()
    """

    # the list that will contain all the final converging values
    Y_0_final = [0]
    max_iter = 1000

    #list_of_weights = [10 * i for i in range(0, 1)]  # [0, 10, 20, 30, 40, 50, ...]
    #list_of_weights = [1 * i for i in range(36, 56)]  # [0, 10, 20, 30, 40, 50, ...]
    list_of_weights = [0]
    for num_weight in list_of_weights:

        print("weight number:", num_weight)

        # now we iterate and we do not stop untile the condition at time 0 is satisfied.
        condition = False
        n_iter = 0

        b_vec = np.zeros((1, L, N))  # (iteration, simulation, time step)
        Y_j = np.zeros((1, L, N))  # first dimension for convergence, second for simulation
        #Y_j[-1, :, -1] = G_option(S[:, -1], delta, K, T) # not relevant, we do not work with the last j
        Z_j = np.zeros((1, L, N))  # first dimension for convergence, second for simulation

        while condition == False:
            n_iter += 1
            #print("This is iteration number:", n_iter)
            #if n_iter % 50 == 0 or n_iter == 1: print("This is iteration number:", n_iter)

            new_layer = np.zeros((1, L, N))
            b_vec = np.concatenate((b_vec, new_layer), axis=0)   # creating space for a new iteration
            Y_j = np.concatenate((Y_j, new_layer), axis=0)       # creating space for a new iteration
            Z_j = np.concatenate((Z_j, new_layer), axis=0)       # creating space for a new iteration
            #Y_j[-1, :, -1] = G_option(S[:, -1], delta, K, T)     # not relevant, we do not work with the last j
            b_vec[n_iter, :, N - 1] = G_option(S[:, -1], delta, K, T) #try.......................

            #print(b_vec[-1, :, N-1])
            # here should be starting from the last j, in the next j loop no!
            for j in range(N-2, -1, -1):  # python loops excludes the stop value (-1)

                delta_j = t[j+1]-t[j] # python parte da 0

                # To compute b_vec for each l, fixing j < N-1
                for l in range(0, L):
                    sum_generator = 0
                    # just to get the sum of the effect of the generator over time
                    for i in range(j, N-1): # check the indexes
                        sum_generator = sum_generator + generator(0, 0, Y_j[n_iter-1, l, i], Z_j[n_iter-1, l, i], num_weight)*delta_j

                    b_vec[n_iter, l, j] = b_vec[n_iter, l, N-1] - sum_generator

            for j in range(0, N-1):

                delta_j = t[j + 1] - t[j]

                orthonormal_basis_vectors = orthonormalize_basis(S[:, j], basis_functions)
                _, kappa_j = orthonormal_basis_vectors.shape
                Y_result = np.zeros(L)
                Z_result = np.zeros(L)

                for lam in range(L):  # loop over λ
                    sum_l_y = 0
                    sum_l_z = 0
                    for l in range(L):  # loop over l
                        sum_k_y = 0
                        sum_k_z = 0
                        for k in range(kappa_j):  # loop over k
                            psi_k_lam = orthonormal_basis_vectors[lam, k]
                            psi_k_l = orthonormal_basis_vectors[l, k]
                            sum_k_y += psi_k_lam * psi_k_l * b_vec[-1, l, j]
                            sum_k_z += psi_k_lam * psi_k_l * b_vec[-1, l, j+1] * (W[l, j + 1] - W[l, j]) / t[j+1] - t[j]
                        sum_l_y += sum_k_y
                        sum_l_z += sum_k_z
                    Y_result[lam] = sum_l_y / L
                    Z_result[lam] = sum_l_z / L

                # Assign manually computed result
                Y_j[n_iter, :, j] = Y_result
                Z_j[n_iter, :, j] = Z_result


            # END OF THE j-LOOP OVER TIME
            #######################################################################

            # Now I have everything until j = 0.
            # to check condition for iteration
            if n_iter >= 1:
                if np.abs(np.mean(Y_j[n_iter-1, :, 0]) - np.mean(Y_j[n_iter, :, 0])) <= 0.0001:
                    condition = True
                    Y_0_final.append(np.mean(Y_j[n_iter, :, 0]))
                    print("Setting condition to True. Iteration n:", n_iter)
                    print("Y at time t=0:\n", np.mean(Y_j[n_iter, :, 0]))
                else:
                    print("n_iter = ", n_iter, "; Y_0 value = ", np.mean(Y_j[n_iter, :, 0]))


            # to break in case of errors:
            if n_iter >= max_iter:
                max_iter = max_iter * 2
                condition = True
                Y_0_final.append(np.mean(Y_j[n_iter, :, 0]))
                print("NO CONVERGENCE, CHECK RESULTS, APPROXIMATIVE VALUE FOR Y")


    print("Y_0_final", Y_0_final)
    plt.plot(Y_0_final[1:], marker='o')  # 'o' marks each point
    plt.title("Line Plot of List Elements")
    plt.xlabel("penalization")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
