import sys, os
sys.path.append('/Users/clancy/repos/hermite_trust_region/')

import numpy as np
from numpy.linalg import norm

import hermite_tr_solver
from trophy import DynTR
import pycutest_for_trophy as pycutest
import util_func
import pandas as pd
import copy



# get unconstrained problems
problems = pycutest.find_problems(constraints='U')

np.random.seed(1)
ii = 0
mindim = 10
maxdim = 150
hermlist = list()
reglist = list()
sr1list = list()
dim = None


"""
These files will change as follows: 
    1. Number of iterations before gradient
    2. BFGS or minimum norm
    3. Number of gradients to use for interpolation (1, 2, or 3)
"""
#bfgs_updating = True
new_grad_every = 1
min_grads = 1 
max_grads = 1
#hessian_type = 'BFGS' if bfgs_updating else 'minnorm'


file_suffix = '_gradevery' + str(new_grad_every) + 'it_using' + str(min_grads) + 'grad.csv'


# see if the file we are looking at has any problems included already 
#file_suffix = '_gradevery_1it_' + str(mindim) + '_to_' + str(maxdim) + '_BFGS.csv'

#os.chdir('/Users/clancy/Research/hermite_trust/experiments/gradient_every_iteration')
print(os.getcwd())
os.chdir('/Users/clancy/repos/hermite_trust/')

if os.path.isfile('data/hitr_mns' + file_suffix):
    temp_df = pd.read_csv('data/hitr_bfgs' + file_suffix)
else:
    with open('data/hitr_mns' + file_suffix, 'w') as fp:
        pass

try:
    # read the last problem that we've solved
    latest_problem = temp_df.iloc[-1,0]
except:
    # if an error was thrown, just set to a name that isn't in the problem set
    latest_problem = 'RUMPLESTILTSKON'
found_prob = False

# look through unconstrained problems.
for i, prob in enumerate(problems):
    if prob == latest_problem:
        found_prob = True
        break

# was the problem found
if found_prob:
    # if so, just don't try solving it again, start looking at the first unsolved problem
    problems = problems[(i+1):]
else:
    # if not, construct columns at print to file
    df_columns = ['problem', 'dim', 'f', 'normg', 'iteration','fevals','gevals','success','stopping_criteria']  
    alldf = pd.DataFrame(columns=df_columns)
    hermmnsdf = pd.DataFrame(columns=df_columns)
    hermbfgsdf = pd.DataFrame(columns=df_columns)
    regdf = pd.DataFrame(columns=df_columns)
    sr1df = pd.DataFrame(columns=df_columns)

    alldf.to_csv('data/combined'+file_suffix, index=False)
    hermmnsdf.to_csv('data/hitr_mns'+file_suffix, index=False)
    hermbfgsdf.to_csv('data/hitr_bfgs'+file_suffix, index=False)
    regdf.to_csv('data/int'+file_suffix, index=False)
    sr1df.to_csv('data/sr1'+file_suffix, index=False)

print('Writing to', file_suffix)
    


    
    


maxit = 1000
# loop through unsolved problems 
for prob in problems:
    ii += 1
    if prob not in ['COATINGNE', 'DEVGLA2NE', 'JIMACK', 'S308NE'] and 'BA-' not in prob:
        # get objective function handle
        prob_name = prob + "_double"
        p = pycutest.import_problem(prob_name)
        x0 = p.x0
        dim = x0.shape[0]
        if (mindim <= dim) and (dim <= maxdim):
            func = lambda z: p.obj(z, gradient=True)
            print('\n \n'+str(ii)+". Problem", prob, "of dim", dim, '====================================================')
            _, gg = func(x0)

            # get function handle for TROPHY solver 
            precision_dict = {'double':0}
            func2 = lambda z, prec: util_func.pycutest_wrapper(z, prec, p, p)

            del_init = norm(gg)/10
            #del_init = 10

            # solve using hermite interpolation TR solver
            print('Hermite interpolation - Minimum Norm')
            herm_mns = hermite_tr_solver.dfo_tr_solver(x0, func, delta_init=del_init, delta_max=1e6, eta_ok=.001, 
                                        eta_great=0.1, gamma_inc=2, gamma_dec=0.25, ftol = 1e-16, gtol=1e-6,max_it=maxit, 
                                        min_num_grads=min_grads, max_num_grads=max_grads, gradient_every=new_grad_every, print_every=10, bfgs_update=False, 
                                        write_to='data/hitr_mns_'+prob+file_suffix)
            print('\n \n \n \n \n')

            # solve using hermite interpolation TR solver
            print('Hermite interpolation - BFGS update')
            herm_bfgs = hermite_tr_solver.dfo_tr_solver(x0, func, delta_init=del_init, delta_max=1e6, eta_ok=.001, 
                                        eta_great=0.1, gamma_inc=2, gamma_dec=0.25, ftol = 1e-16, gtol=1e-6,max_it=maxit, 
                                        min_num_grads=min_grads, max_num_grads=max_grads, gradient_every=new_grad_every, print_every=10, bfgs_update=True, 
                                        write_to='data/hitr_bfgs_'+prob+file_suffix)



            print('\n \n \n \n \n')        

            # sovle using interpolation
            print('Intepolation')
            reg = hermite_tr_solver.dfo_tr_solver(x0, func, delta_init=del_init, delta_max=1e6, eta_ok=.001, 
                                        eta_great=0.1, gamma_inc=2, gamma_dec=0.25, ftol = 1e-16, gtol=1e-6,max_it=maxit, 
                                        min_num_grads=0, max_num_grads=0, gradient_every=np.inf, print_every=10, bfgs_update=False, 
                                        write_to='data/itr_'+prob+file_suffix)
            print('\n \n \n \n \n')

            # solve using TR SR1 update
            print("Gradient and Hessian approximation")
            sr1 = DynTR( x0, func2, precision_dict, gtol=1.0e-6, max_iter=maxit, eta_good=0.001, 
                                eta_great=0.1, gamma_dec=0.25, gamma_inc=2, max_memory=30, tr_tol=1.0e-6, max_delta=1e6, 
                                sr1_tol=1.e-4, delta_init=10, verbose=True, store_history=False, norm_to_use=2,
                                write_to='data/sr1_'+prob+file_suffix)

            
            # create list to append to existing data
            herm_mns_temp = [prob, dim, herm_mns.fun, norm(herm_mns.grad), herm_mns.niter, herm_mns.fevals, herm_mns.gevals, herm_mns.success, herm_mns.stop_criteria]
            herm_bfgs_temp = [prob, dim, herm_bfgs.fun, norm(herm_bfgs.grad), herm_bfgs.niter, herm_bfgs.fevals, herm_bfgs.gevals, herm_bfgs.success, herm_bfgs.stop_criteria]
            reg_temp = [prob, dim, reg.fun, norm(reg.grad), reg.niter, reg.fevals, reg.gevals, reg.success, reg.stop_criteria]
            sr1_temp = [prob, dim, sr1.fun, norm(sr1.jac), sr1.nit, sr1.nfev, sr1.nfev, sr1.success, sr1.message]
            
            # make list to write to combined file
            all_temp = copy.copy(herm_mns_temp)
            all_temp.extend(herm_bfgs_temp)
            all_temp.extend(reg_temp)
            all_temp.extend(sr1_temp)

            # construct a data frame for each of the different solvers
            hermmnsdf = pd.DataFrame([herm_mns_temp])
            hermbfgsdf = pd.DataFrame([herm_mns_temp])
            regdf = pd.DataFrame([reg_temp])
            sr1df = pd.DataFrame([sr1_temp])
            alldf = pd.DataFrame([all_temp])
            
            # append new data frame to existing file.   
            alldf.to_csv('data/combined'+file_suffix, mode='a', header=False, index=False)
            hermmnsdf.to_csv('data/hitr_mns'+file_suffix, mode='a', header=False, index=False)
            hermbfgsdf.to_csv('data/hitr_bfgs'+file_suffix, mode='a', header=False, index=False)
            regdf.to_csv('data/int'+file_suffix, mode='a', header=False, index=False)
            sr1df.to_csv('data/sr1'+file_suffix, mode='a', header=False, index=False)
            

        else:
            print('\n', ii, '. Problem ' + prob + ' has dimension ' + str(dim) + '.  Outside interval (',mindim, ',', maxdim, ').')
    else:
        print(ii, 'Problem', prob, 'of dimension', dim, 'is being skipped')
