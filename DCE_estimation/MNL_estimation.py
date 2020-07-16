# %%

# coding: utf-8

# %%

import pandas as pd
import numpy as np
import scipy.stats as sc
from scipy import sparse
from collections import OrderedDict
from scipy.optimize import minimize
import pylogit as pl

import time


def add_intercept_to_df(df_long, specification_dict):
    
    if ("intercept" in specification_dict 
        and "intercept" not in df_long.columns):
        df_long["intercept"] = 1
    
    return None


# %%

def create_design_matrix(df_long, specification_dict,
                         names_dict, alternative_id_col):
    
    add_intercept_to_df(df_long,specification_dict)
    
    columns = []
    for col in specification_dict:
        for group in specification_dict[col]:
            if type(group) == list:
                columns.append(df_long[alternative_id_col].isin(group)
                               *df_long[col])
            else:
                columns.append((df_long[alternative_id_col]==group)
                               *df_long[col])
    
    design_matrix = np.stack(columns,axis = 1)
    
    var_names = []
    for variable in names_dict:
        for name in names_dict[variable]:
            var_names.append(name)
    
    return design_matrix, var_names


# %%

def calculate_utilities(betas, design_matrix):
    
    limit_max = 700
    limit_min = -700 
    
    utility = design_matrix.dot(betas)
    utility[utility>limit_max] = limit_max
    utility[utility<limit_min] = limit_min
    
    utilities = np.exp(utility)
    
    return utilities


# %%

def create_mapping_matrix(df_long, observation_id_col):
    row_to_col_matrix = pd.get_dummies(df_long[observation_id_col]).values
#     row_to_col_matrix = (df_long[observation_id_col].values[:,None] == 
#                          np.sort(df_long[observation_id_col].unique())[None,:]).astype(int) 
    sparse_row_to_col_matrix = sparse.csr_matrix(row_to_col_matrix)
    
    mapping_matrix = sparse_row_to_col_matrix.dot(sparse_row_to_col_matrix.T)
    
    return mapping_matrix


# %%

def calculate_probabilities(betas,design_matrix, mapping_matrix):
    
    utilities = calculate_utilities(betas, design_matrix)
    denominator = mapping_matrix.dot(utilities)
    probabilities = utilities/denominator
    probabilities[probabilities==0] = 1e-300
    
    
    return probabilities


# %%

def calculate_likelihood(betas,design_matrix,mapping_matrix,df_long, choice_col,obs_id_column,const_pos):
    
    probabilities = calculate_probabilities(betas,design_matrix, mapping_matrix)
    y_i = df_long[choice_col].values
    LL = np.log(probabilities).T.dot(y_i)
    
    return LL


# %%

def calculate_gradient(betas,design_matrix,mapping_matrix,df_long, choice_col,obs_id_column,const_pos):
    
    y_i = df_long[choice_col].values
    probabilities = calculate_probabilities(betas,design_matrix, mapping_matrix)
    gradient = design_matrix.T.dot(y_i - probabilities)
    
    return gradient


# %%

def calculate_hessian(df_long, design_matrix, probabilities,obs_id_column, constrained_pos):
    
    probabilities = probabilities[:,None]
    design_matrix_df = pd.DataFrame(probabilities * design_matrix)
    design_matrix_df.index = df_long[obs_id_column]
    new_design_matrix = design_matrix_df.groupby(design_matrix_df.index).sum().values
    
    hess = - (probabilities * design_matrix).T.dot(design_matrix)           + ((new_design_matrix).T.dot(new_design_matrix)) 
    
    if constrained_pos is not None:
        for pos in constrained_pos:
            hess[pos,:] = 0
            hess[:,pos] = 0
            hess[pos,pos] = -1
    return hess


# %%

def calculate_neg_LL_and_grad(betas,design_matrix,mapping_matrix,df_long, choice_col,obs_id_column,const_pos):
    
    neg_LL = -1 * calculate_likelihood(betas,design_matrix,mapping_matrix,df_long, choice_col,obs_id_column,const_pos)
    neg_graident = -1 * calculate_gradient(betas, design_matrix, mapping_matrix,df_long, choice_col,obs_id_column,const_pos)
    
    if const_pos is not None:
        neg_graident[const_pos] = 0
        
    return neg_LL,neg_graident


# %%

def estimate(df_long,betas,design_matrix,mapping_matrix,choice_col, obs_id_column,max_iterations, const_pos):
       
    results = minimize(calculate_neg_LL_and_grad,x0=betas,method='BFGS',jac=True,
                       args = (design_matrix, mapping_matrix, df_long, 
                               choice_col, obs_id_column, const_pos),
                       options={'gtol': 0.001,
                                "maxiter": max_iterations})
    
    return results


# %%

# def newtons_method(df_long, design_matrix, obs_id_column, 
#                    choice_col, alternative_id_col,
#                    betas, mapping_matrix, max_iterations):

#     iterator = 0
#     converged = False
    
#     while (converged == False) & (iterator<max_iterations):
#         probabilities = calculate_probabilities(betas, design_matrix, mapping_matrix)
#         gradient = calculate_gradient(df_long, design_matrix, probabilities, choice_col)
#         hess = calculate_hessian(df_long, design_matrix, probabilities,obs_id_column)
        
#         betas_new = betas - np.linalg.inv(hess).dot(gradient)
#         betas = betas_new
#         #print (betas.T)
#         iterator+=1
#         #print (iterator)
#         if all(abs(gradient)<0.0001):
#             converged = True
            
#     return betas_new, probabilities, gradient, hess         


# %%

def calculate_std_errors(hess):
    
    std_errors = np.sqrt(np.diag(np.linalg.inv(-hess)))
    return std_errors


# %%

def calculate_tstats(betas, std_errors):
    
    t_stats = betas/std_errors
    return np.round(t_stats,2)


# %%

def calculate_pvalues(tstats):
    
    p_values = 2 * (1 - sc.norm.cdf(abs(tstats)))
    return np.round(p_values,4)


# %%

def fit_MNL(df_long, 
            specification_dict,
            names_dict,
            obs_id_column, 
            choice_col,
            alternative_id_col,
            initial_values,
            max_iterations = 1000,
            const_pos = None):
    
    
    design_matrix, var_names = create_design_matrix(df_long,specification_dict, 
                                         names_dict, alternative_id_col)
    mapping_matrix = create_mapping_matrix(df_long, obs_id_column)
    num_vars = sum([len(specification_dict[x]) 
                    for x in specification_dict])
    
    betas = initial_values
    
    probabilities0 = calculate_probabilities(np.zeros(num_vars),
                                             design_matrix,mapping_matrix)
    LL0 = calculate_likelihood(np.zeros(num_vars),design_matrix,
                               mapping_matrix,df_long, 
                               choice_col,obs_id_column,
                               const_pos)

    initial_probabilities = calculate_probabilities(betas,design_matrix,mapping_matrix)
    LL_initial = calculate_likelihood(betas,design_matrix,mapping_matrix,
                                      df_long, 
                                      choice_col,
                                      obs_id_column,const_pos)
    
    time1 = time.time()
    results = estimate(df_long,betas,design_matrix,
                       mapping_matrix,
                       choice_col, obs_id_column,
                       max_iterations, const_pos)
    time2 = time.time()
    
    final_betas = results['x']
    final_probabilities = calculate_probabilities(final_betas,design_matrix,mapping_matrix)
    final_hess = calculate_hessian(df_long,design_matrix,final_probabilities,obs_id_column, const_pos)
        
    std_errors = calculate_std_errors(final_hess)
    t_stats = calculate_tstats(final_betas, std_errors)
    p_values = calculate_pvalues(t_stats)
    final_ll = -1 * results['fun']
    
    output = pd.DataFrame(index=var_names)
    output["Estimates"] = final_betas
    output["std errors"] = std_errors
    output["t_stat"] = t_stats
    output["p_values"] = p_values
    
    if const_pos is not None:
        output.iloc[const_pos,1:] = np.nan
    
    print ("Initial Log-likelihood:  {}".format(np.round(LL_initial,4)))
    print ("Null Log-likelihood(LL0):{}".format(np.round(LL0,4)))
    print ('Estimation time: ',np.round(time2 - time1,2))
    print ("Final Log-likelihood:    {}".format(np.round(final_ll,4)))
    print ("Rho_bar sqrd =            {}".format(np.round(1-final_ll/LL0,4)))
    
    return output
