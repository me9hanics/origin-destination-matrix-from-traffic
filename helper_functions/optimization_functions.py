import numpy as np
import math
from scipy.optimize import minimize, Bounds, LinearConstraint
import functools

##################### Helper functions #####################
def log_factorial(n):
    """Roughly n*log(n) - n (their derivatives are asymptotically equal)
        - might as well use the simpler formula in some cases.
    """
    return np.sum(np.log(np.arange(1, n+1)))

def log_factorial_float(x):
    return math.lgamma(x+1)
    
def sign(x):
    if x > 0: return 1
    elif x < 0: return -1
    else: return 0

##################### Objective functions + gradients #####################
def entropy_max(odm):
    #Entropy maximizing (== minimizing the negative entropy)
    odm_nonzero = odm[odm != 0] #To avoid log(0) error
    return np.sum(odm_nonzero * np.log(odm_nonzero))

def entropy_max_gradient(x):
    y = [-np.inf if i == 0 else np.log(i) + 1 for i in x]
    return np.array(y)


def entropy_min(odm):
    odm_nonzero = odm[odm != 0] #To avoid log(0) error
    return -np.sum(odm_nonzero * np.log(odm_nonzero))

def entropy_min_gradient(x):
    y = [np.inf if i == 0 else -np.log(i)-1 for i in x]
    return np.array(y)


def F_Bell(odm_vector, q):
    #Function to MAXIMIZE
    
    #The logarithm of the objective function, that maximizes the joint probability of observing t
    #subject to the constraints v_i = P_{ij}t_j, given the probabilities q and the assumption 
    #that trips are multinomially distributed.
    log_numerator = log_factorial_float(np.sum(odm_vector))
    log_denominator = np.sum([log_factorial_float(x) for x in odm_vector])
    log_probabilities = np.sum(odm_vector * np.log(q))
    return log_numerator - log_denominator + log_probabilities

def F_Bell_optimize(odm_vector, q):
    #Function to MINIMIZE
    return -F_Bell(odm_vector, q)

def F_Bell_gradient(odm_vector, q):
    t_sum = np.sum(odm_vector)
    return np.log(t_sum) - np.log(odm_vector) + np.log(q)

def F_Bell_optimize_gradient(odm_vector, q):
    #In case of minimization, we swap the signs
    return -F_Bell_gradient(odm_vector, q)


def F_Bell_modified(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05):
    errors = P_modified_loss @ odm_vector - v_modified_loss
    loss = np.sum(np.abs(errors)*np.log(np.abs(errors)+1))
    return F_Bell(odm_vector, q) - c*loss

def F_Bell_modified_optimize(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05):
    return -F_Bell_modified(odm_vector, q, P_modified_loss, v_modified_loss, c)

def F_Bell_modified_gradient(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05):
    #See here: https://www.wolframalpha.com/input?i=d%28%7Cx%7Clog%28%7Cx%7C%2B1%29%29%2Fdx
    loss_gradient = np.zeros_like(odm_vector)
    for i in range(len(v_modified_loss)): #All dependent equations, included in loss
        error_i = v_modified_loss[i]- P_modified_loss[i, :] @ odm_vector
        loss_i_derivative =  -(error_i/(np.abs(error_i)+1) + np.sign(error_i)*np.log(np.abs(error_i)+1))
        loss_gradient += P_modified_loss[i, :] * loss_i_derivative #Each column of P is an odm entry (so this is the vector already)
    return F_Bell_gradient(odm_vector, q) - c*loss_gradient

def F_Bell_modified_optimize_gradient(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05):
    #In case of minimization, we swap the signs
    return -F_Bell_modified_gradient(odm_vector, q, P_modified_loss, v_modified_loss, c)


def F_Bell_L2(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05): 
    #"Overfits" on the loss. Likely unneccessary overfitting, constraints are strong enough
    #Objective function grows ~xln(x), but the loss grows ~x^2 -> too much emphasis on the loss
    loss = np.sum((P_modified_loss @ odm_vector - v_modified_loss)**2)
    return F_Bell(odm_vector, q) - c*loss

def F_Bell_L2_optimize(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05):
    return -F_Bell_L2(odm_vector, q, P_modified_loss, v_modified_loss, c)

def F_Bell_L2_gradient(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05): #"Overfits" on the loss. Likely unneccessary overfitting, constraints are strong enough
    #Objective function grows ~xln(x), but the loss grows ~x^2 -> too much emphasis on the loss
    loss_gradient = np.zeros_like(odm_vector)
    for i in range(len(v_modified_loss)): #All dependent equations, included in loss
        error_i = v_modified_loss[i]- P_modified_loss[i, :] @ odm_vector
        loss_i_derivative =  2*error_i
        for k in range(len(odm_vector)): #Partial derivative by t_k
            loss_gradient[k] += P_modified_loss[i, k] * loss_i_derivative
    return F_Bell_gradient(odm_vector, q) - c*loss_gradient

def F_Bell_L2_optimize_gradient(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05):
    #In case of minimization, we swap the signs
    return -F_Bell_L2_gradient(odm_vector, q, P_modified_loss, v_modified_loss, c)


def F_Bell_L1_approximation(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05, k = 10000/9999):
    #A smooth approximation (L1 is non-differentiable at 0 errors), but interpretation isn't clear
    loss = np.sum(np.power((P_modified_loss @ odm_vector - v_modified_loss)**2, k/2)) #First square, so it's always non-negative
    return F_Bell(odm_vector, q) - c*loss

def F_Bell_L1_approximation_optimize(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05, k = 10000/9999):
    return -F_Bell_L1_approximation(odm_vector, q, P_modified_loss, v_modified_loss, c, k)

def F_Bell_L1_approximation_gradient(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05, k = 10000/9999):
    loss_gradient = np.zeros_like(odm_vector)
    for i in range(len(v_modified_loss)): #All dependent equations, included in loss
        error_i = v_modified_loss[i]- P_modified_loss[i, :] @ odm_vector
        loss_i_derivative =  k * np.power(error_i, k-1)*np.sign(error_i) #This is continuous. Sign is for the derivative of abs
        loss_gradient -= P_modified_loss[i, :] * loss_i_derivative #Each column of P is an odm entry (so this is the vector already)
    return F_Bell_gradient(odm_vector, q) - c*loss_gradient

def F_Bell_L1_approximation_optimize_gradient(odm_vector, q, P_modified_loss, v_modified_loss, c=0.05, k = 10000/9999):
    #In case of minimization, we swap the signs
    return -F_Bell_L1_approximation_gradient(odm_vector, q, P_modified_loss, v_modified_loss, c, k)

##################### Optimization functions #####################
def odm_linear_constraint(P, v):
    return LinearConstraint(P, lb=v, ub=v)

def bounds(lower_bound, upper_bound):
    return Bounds(lower_bound, upper_bound)

def optimize_odm(model_function, odm_initial, constraints_linear, runs=10+1, model_func_args=None, bounds=None, model_derivative=None, verbose=True, return_last=True):
    """
    Optimize the ODM using the given model function and constraints.

    Args:
        model_function (function): The function to optimize (e.g. F_Bell).
        odm_initial (np.array): The initial ODM, where the optimization starts.
        constraints_linear (LinearConstraint): The linear constraints, usually v = Pt.
        runs (int): The number of optimization runs to perform. Default is 11.
        model_func_args (dict): Additional arguments for the model function. Default is None.
        bounds (Bounds): The bounds for the optimization. Default is None.
        model_derivative (function): The derivative of the model function. Default is None.
        verbose (bool): Whether to print the optimization messages. Default is True.
        return_last (bool): Whether to return only the last optimized ODM. If False, returns all
            previous ODM estimations too. Default is True.
    """
    
    if model_func_args is not None: #Required for the lossful Bell models: P_modified_loss, v_modified_loss, c
        model_function = functools.partial(model_function, **model_func_args)
        model_derivative = functools.partial(model_derivative, **model_func_args)

    result = minimize(model_function, odm_initial, constraints=constraints_linear, bounds=bounds, jac = model_derivative)
    optimal_odm = result.x

    if verbose:
        print('First run success:', result.success)
        print('First run message:', result.message)
    optimal_odm_list = []

    for i in range(runs-1):
        res = minimize(model_function, optimal_odm, constraints=constraints_linear, bounds=bounds, jac = model_derivative)
        optimal_odm = res.x
        optimal_odm_list.append(optimal_odm)
        if res.message == 'Inequality constraints incompatible':
            if verbose:
                print(f'Run {i}: Inequality constraints incompatible - likely hit a local minimum.')
            break
        if res.message != 'Iteration limit reached':
            if verbose:
                print(f'Run {i} message:', res.message)
            
        if res.message == 'Positive directional derivative for linesearch':
            if verbose:
                print(f'Run {i}: Positive directional derivative for linesearch - likely hit a local minimum.')
            break
        if len(optimal_odm_list) > 1:
            if optimal_odm_list[-2].astype(int).tolist() == optimal_odm.astype(int).tolist():
                if verbose:
                    print(f'Run {i}: Optimal ODM did not change substantially (all changes < 1) - likely hit a local minimum.')
                break
        
    if return_last:
        return optimal_odm_list[-1]
    return optimal_odm_list