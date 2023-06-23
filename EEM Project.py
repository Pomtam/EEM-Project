from numpy import *
from matplotlib import pyplot as plt
import time



# Define pre-supposed steady-state abundunces (n), and mean & variance of Gaussian distribution p(beta|n)
n = array([[0.9203],
    [0.0527],
    [0.7379],
    [0.2691],
    [0.4228]])
beta_mean = 0.4
beta_variance = 0.2

S = len(n)

# How many samples we want to get
Samples = 10000

# A four-dimensional matrix to store main results (sampled matrices Beta)
betas = zeros((S,S,Samples))
betas_cong = betas








# Function outputs three properties of the multivariate Gaussian
# distribution p(Beta|n) that Barbier et al. (2021) derive
# Solves analytically

def calculateBarbierDistProperties(betaMean, betaVariance, n):
    S = len(n)
    beta_mean_matrix = zeros((S, S - 1))
    beta_covariance_matrix = zeros((S - 1, S - 1, S))
    beta_variance_matrix = zeros((S, S - 1))

    for i in range(S):
        n_i = array([delete(n, i)]).T

        # Mean vectors mu_{Beta_i}
        beta_mean_matrix[i, :] = ((1-n[i]) / ((linalg.norm(n_i))**2) * n_i
                                 + betaMean * ones((S-1,1))
                                 - matmul(betaMean * (n_i * transpose(n_i)) / ((linalg.norm(n_i))**2), ones((S-1, 1)))).T

        # Covariance matrices Sigma_{Beta_i}
        beta_covariance_matrix[:, :,  i] = betaVariance * (eye(S - 1) - (n_i * transpose(n_i)) / ((linalg.norm(n_i))**2))

        # Extract the diagonal elements of covariance matrices Sigma_{Beta_i}
        # to obtain a matrix of variances.
        beta_variance_matrix[i, :] = transpose(diag(beta_covariance_matrix[:, :, i]))

    # Overall mean of all non-diagonal Beta elements, calculated from the
    # individual means of these elements.
    beta_mean_overall = mean(beta_mean_matrix)

    return beta_mean_overall, beta_mean_matrix, beta_variance_matrix


# Function to sample from Gaussian distribution defined by Barbier et al.(2021)
# Perform the Barbier et al. (2021) algorithm to obtain one sample from p(Beta|n)


def sampleBetaBarbier(beta_mean,beta_variance,n,all_Rs):
    S = len(n)
    beta = ones((S,S))

    all_Rs = rotationMatrix()

    for i in range(S):
        n_i = array([delete(n, i)]).T


        #Step 1. Use the relevant rotation matrix.
        R = all_Rs[:,:, i]

        #Step 2. Calculate the first element of x, i.e. x_1.
        x_1 = (1 - n[i]) / linalg.norm(n_i)

        # Step 3. Calculate the remaining elements of x, i.e. x_2:x_{S-1}.
        x_mean = matmul(beta_mean * R[1:shape(R)[1],:], ones((S - 1, 1)))
        x_cov = beta_variance * eye(S - 2)
        x_excl_1 = random.multivariate_normal(x_mean.T.tolist()[0], x_cov).T

        # Step 4. Step 4. Join x together, and reverse transform to get all
        # non-diagonal elements of the ith row of beta
        beta_i = matmul(R.T, array([concatenate((x_1, x_excl_1))]).T)

        # Step 5. Place the found elements of Beta into the matrix Beta.
        beta[i, 0: i] = beta_i[0: i].T
        beta[i, i + 1: shape(beta)[1]] = beta_i[i: shape(beta_i)[0]].T

    return beta




def rotationMatrix():
    S = len(n)

    all_Rs = zeros((S - 1, S - 1, S))

    for i in range(S):
        n_i = array([delete(n, i)]).T

        all_Rs[:,:, i] = calculateR(n_i)


    return all_Rs


# Debug with a control instead of random
def calculateR(n_i):
    return gramschmidt( concatenate( (n_i, random.rand(len(n_i),len(n_i)-1)), axis=1 )).T

# V = array([[0.0527    , 0.84746291, 0.74926038, 0.29551553],
#        [0.7379    , 0.44482405, 0.23400031, 0.81735498],
#        [0.2691    , 0.16142829, 0.03814779, 0.04035871],
#        [0.4228    , 0.01926175, 0.06751803, 0.49741182]])

def gramschmidt(V):
    n,k = shape(V)

    U = zeros((n,k))
    U[:,0] = V[:,0] / linalg.norm(V[:,0])

    for i in range(1,k):
        U[:,i] = V[:,i]
        for j in range(0, i):
            # find orthogonal component
            U[:,i] = U[:,i] - dot(U[:,j], U[:,i]) * U[:,j]

        # normalise orthogonal componenet
        U[:,i] = U[:,i] / linalg.norm(U[:,i])
    return U




# Function to calculate sample mean and sample variances of non-diagonal elements of Beta matrix obtained through Barbier sampling approach
def CalculateElementWiseMeansVariances(betas):
    # Each Beta matrix has size GxG = (S-1)x(S-1).
    G = shape(betas)[0]


    # Create vectors to store the results we want to extract. Since Beta has
    # size GxG, we want to extra results associated only with the non-diagonal
    # elements, of which there are Gx(G-1) elements
    mu_out = zeros((G * (G - 1), 1))
    var_out = zeros((G * (G - 1), 1))

    t = 0
    for g in range(G):
        for h in range(g):
            mu_out[t] = mean(betas[g,h,:])
            var_out[t] = (std(betas[g,h,:]))**2
            t = t+1

        # Skip over diagonal elements
        for h in range(g+1,G):
            mu_out[t] = mean(betas[g,h,:])
            var_out[t] = (std(betas[g,h,:]))**2
            t = t+1

    return mu_out,var_out



def cong(beta_mean,beta_variance, n):
    S = len(n)
    beta = ones((S, S))

    for i in range(S):
        n_i = array([delete(n, i)]).T

        # Step 1. Sample y from normal distribution
        y_mean = beta_mean * ones((S - 1, 1))
        y_cov = beta_variance * eye(S - 1)


        y = random.multivariate_normal(y_mean.T.tolist()[0], y_cov).T

        # Step 2. Obtain Beta_i
        beta_i = (eye(S - 1) - n_i@n_i.T/linalg.norm(n_i)**2) @ y + (n_i/linalg.norm(n_i)**2).T[0] * (1 - n[i])
        beta[i, 0: i] = beta_i[0: i].T
        beta[i, i + 1:] = beta_i[i:]

        # Step 3. Repeat for all rows to obtain a sampled matrix of Beta

    return beta





# Obtain means and variances through Barbier Sampling (and time)
    # Begin timing
start_time = time.time()
    # Generate samples of Beta matrix using Barbier method
for i in range(Samples):
    betas[:,:, i] = sampleBetaBarbier(beta_mean, beta_variance, n, rotationMatrix())
    # Calculate means & variances of non-diagonal elements of Beta
mu_sample,var_sample = CalculateElementWiseMeansVariances(betas)
    # End Timer
duration_barbier = time.time() - start_time



# Obtain means and variances through Cong (and time)
start_time = time.time()
for i in range(Samples):
    betas_cong[:,:,i] = cong(beta_mean, beta_variance, n)

mu_sample2,var_sample2 = CalculateElementWiseMeansVariances(betas_cong)

duration_cong = time.time() - start_time



# Obtain means and variances analytically
_, beta_analytical_matrix, var_analytical_matrix = calculateBarbierDistProperties(beta_mean, beta_variance, n)

# Convert to analytical results to single column
betaAVector = matrix.flatten(beta_analytical_matrix)
varAVector = matrix.flatten(var_analytical_matrix)

# Print Results
print("Barbier algorithm: %.3f seconds" % (duration_barbier))
print("Cong algorithm: %.3f seconds" % (duration_cong))



# print(mu_sample), print(mu_sample2)
# print(var_sample), print(var_sample2)


#
# print(mu_sample-mu_sample2), print(var_sample-var_sample2)

# print(mu_sample2)



#
# for i in range(20):
#     print (varAVector[i])
#
# for i in range(20):
#     print (betaAVector[i])

