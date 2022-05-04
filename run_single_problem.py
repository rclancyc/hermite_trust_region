import numpy as np
from numpy.linalg import norm
import sys, os

os.chdir('/Users/clancy/repos/hermite_trust_region/')
sys.path.append('/Users/clancy/repos/hermite_trust_region/')

# To call our solver, we will need to make changes to standard interpolation. In particular, the input parameters
# will be different, so make sure to check 
import hermite_tr_solver
from trophy import DynTR
import pycutest_for_trophy as pycutest
import util_func
import pandas as pd
#import scipy.optimize




np.random.seed(1)
min_dim = 100
max_dim = 100

prob_list = ['SPIN2LS','QING', 'LUKSAN21LS','VANDANMSLS','SENSORS','COATING','DMN15333LS','DIAMON3DLS',
'ARGTRIGLS','VARDIM','LUKSAN22LS','CHNROSNB','HYDCAR6LS','TOINTQOR','STRTCHDV','STRATEC','HILBERTB',
'DMN15332LS','LUKSAN15LS','BROWNAL','METHANL8LS','LUKSAN17LS','TOINTPSP','METHANB8LS','WATSON',
'ARGLINA','LUKSAN13LS','DMN37143LS','LUKSAN11LS','ARGLINC','HATFLDGLS','DIAMON2DLS','COATINGNE','ERRINRSM'
'ARGLINB','LUKSAN16LS','ERRINROS','PENALTY2','PENALTY3','TOINTGOR','OSBORNEB','HYDC20LS','LUKSAN14LS','MANCINO',
'DMN15103LS','DMN37142LS','TRIGON1','PARKCH','CHNRSNBM','TRIGON2','LUKSAN12LS']


'''
Tells us size of different problems for reference.

SPIN2LS, 102
QING_double dimension 100
LUKSAN21LS_double dimension 100
VANDANMSLS_double dimension 22
SENSORS_double dimension 100
COATING_double dimension 134
DMN15333LS_double dimension 99
DIAMON3DLS_double dimension 99
ARGTRIGLS_double dimension 200
VARDIM_double dimension 200
LUKSAN22LS_double dimension 100
CHNROSNB_double dimension 50
HYDCAR6LS_double dimension 29
TOINTQOR_double dimension 50
STRTCHDV_double dimension 10
STRATEC_double dimension 10
HILBERTB_double dimension 10
DMN15332LS_double dimension 66
LUKSAN15LS_double dimension 100
BROWNAL_double dimension 200
METHANL8LS_double dimension 31
LUKSAN17LS_double dimension 100
TOINTPSP_double dimension 50
METHANB8LS_double dimension 31
WATSON_double dimension 12
ARGLINA_double dimension 200
LUKSAN13LS_double dimension 98
DMN37143LS_double dimension 99
LUKSAN11LS_double dimension 100
ARGLINC_double dimension 200
HATFLDGLS_double dimension 25
DIAMON2DLS_double dimension 66
COATINGNE_double dimension 134
ERRINRSM_double dimension 50
ARGLINB_double dimension 200
LUKSAN16LS_double dimension 100
ERRINROS_double dimension 50
PENALTY2_double dimension 200
PENALTY3_double dimension 200
TOINTGOR_double dimension 50
OSBORNEB_double dimension 11
HYDC20LS_double dimension 99
LUKSAN14LS_double dimension 98
MANCINO_double dimension 100
DMN15103LS_double dimension 99
DMN37142LS_double dimension 66
TRIGON1_double dimension 10
PARKCH_double dimension 15
CHNRSNBM_double dimension 50
TRIGON2_double dimension 10
LUKSAN12LS_double dimension 98
'''

# select problem ot solve
prob = prob_list[1]
prob = 'TRIGON1'
prob =  'SPIN2LS'
print('prob')
prob_name = prob + '_double'
ii = 0
hermlist = list()
reglist = list()
sr1list = list()
dim = None


# load objective function handle
p = pycutest.import_problem(prob_name)
x0 = p.x0
dim = x0.shape[0]
func = lambda z: p.obj(z, gradient=True)
print('\n \n'+str(ii)+". Problem", prob, "of dim", dim, '====================================================')

# need to do this to use trophy algorithm
precision_dict = {'double':0}
func2 = lambda z, prec: util_func.pycutest_wrapper(z, prec, p, p)


maxit = 100

new_grad_every = 5
min_n_g = 1
max_n_g = 2
bfgs_updating=True
ff, gg = func(x0)
del_init = norm(gg)/10

# call hermite interpolation solver
print('Hermite interpolation')

# call the hermite TR solver. Main inputs to change are min_num_grads, max_num_grads and gradient_every which control for the minimum/maximum number of gradients to use for a problem 
# and how often a new gradient should be incorporated into the TR subproblem model. 
herm = hermite_tr_solver.dfo_tr_solver(x0, func, delta_init=10, delta_max=1e6, eta_ok=.001, 
                            eta_great=0.1, gamma_inc=2, gamma_dec=0.25, ftol = 1e-16, gtol=1e-6,max_it=maxit, 
                            min_num_grads=min_n_g, max_num_grads=max_n_g, gradient_every=new_grad_every, print_every=10, bfgs_update=False) #x0.shape[0])

                        

print('\n \n \n \n \n')
print('===============================================================================================')

# call normal interpolation solver...this is enforced by ensuring that min and max number of gradients is zero, i.e., no gradients are used 
print('Intepolation')
reg = hermite_tr_solver.dfo_tr_solver(x0, func, delta_init=10, delta_max=1e6, eta_ok=.001, 
                            eta_great=0.1, gamma_inc=2, gamma_dec=0.25, ftol = 1e-16, gtol=1e-6,max_it=maxit, 
                            min_num_grads=0, max_num_grads=0, gradient_every=np.inf, print_every=10)

print('\n \n \n \n \n')
print('===============================================================================================')

# call trust region method with SR1 update 
print("Gradient and Hessian approximation")
sr1 = DynTR( x0, func2, precision_dict, gtol=1.0e-6, max_iter=maxit, eta_good=0.001, 
                    eta_great=0.1, gamma_dec=0.25, gamma_inc=2, max_memory=30, tr_tol=1.0e-6, max_delta=1e6, sr1_tol=1.e-4, delta_init=10, verbose=True, 
                    store_history=False, norm_to_use=2)



##### I think all of what follows is for the purpose of running through multiple problems and storing them 

# construct lists with details about the current solver. 
df_columns = ['problem', 'dim', 'f', 'normg', 'iteration','fevals','gevals','success','stopping_criteria']
herm_temp = [prob, dim, herm.fun, norm(herm.grad), herm.niter, herm.fevals, herm.gevals, herm.success, herm.stop_criteria]
reg_temp = [prob, dim, reg.fun, norm(reg.grad), reg.niter, reg.fevals, reg.gevals, reg.success, reg.stop_criteria]
sr1_temp = [prob, dim, sr1.fun, norm(sr1.jac), sr1.nit, sr1.nfev, sr1.nfev, sr1.success, sr1.message]

# append details for each solver
hermlist.append(herm_temp)
reglist.append(reg_temp)
sr1list.append(sr1_temp)

# write to data frame 
hermdf = pd.DataFrame(data=hermlist, columns=df_columns)
regdf = pd.DataFrame(data=reglist, columns=df_columns)
sr1df = pd.DataFrame(data=sr1list, columns=df_columns)


write_to = False
if write_to:
    # write results to a csv
    file_suffix = str(min_n_g) + '_every_' + str(new_grad_every) + '.csv'
    hermdf.to_csv('data/hermite'+file_suffix, index=False)
    regdf.to_csv('data/regular'+file_suffix, index=False)
    sr1df.to_csv('data/sr1'+file_suffix, index=False)

    # construct column headers to print all details in a combined file
    hermnames = [h+'_herm' for h in df_columns]
    regnames = [h+'_reg' for h in df_columns]
    sr1names = [h+'_sr1' for h in df_columns]
    allnames = hermnames
    allnames.extend(regnames)
    allnames.extend(sr1names)

    # construct data to fill the data frame
    alldf = pd.concat([hermdf, regdf, sr1df], axis=1)
    alldf.columns = allnames

    # write the data frame to the csv file
    alldf.to_csv('data/combined'+file_suffix, index=False)