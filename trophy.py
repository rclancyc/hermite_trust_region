from os import write
import pandas as pd
import numpy as np
from numpy.linalg import norm
import util_func
import time


# change name from DynTR_for_pydda to DynTR
def DynTR(x0, fun, precision_dict, gtol=1.0e-5, max_iter=1000, eta_good=0.001, eta_great=0.75, gamma_dec=0.25,
          gamma_inc=2, max_memory=30, tr_tol=1.0e-6, max_delta=1e6, sr1_tol=1.e-4, delta_init=None, gamma=None,
          verbose=False, store_history=False, norm_to_use=2, write_folder=None, print_every=10, write_to=None):
    """
    :param x0: initial estimate
    :param fun: function handle with gradient, i.e, f, g = fun(x, precision)
    :param precision_dict: precision dictionary with index/sting pairs, i.e, ex: {'half': 1, 'single': 2, 'double': 3}
    :param gtol: first order condition tolerance, that is algo stops when, ||g|| < gtol (default 1e-5)
    :param max_iter: maximum number of iterations to perform (default 1000)
    :param eta_good: smallest value of predicted_decrease/actual_decrease that TR step is accepted (default 0.001)
    :param eta_great: value of predicted_decrease/actual_decrease such that TR radius is expanded
    :param gamma_dec: ratio by which TR radius is shrunk
    :param gamma_inc: ratio by which TR is expanded
    :param max_memory: maximum number of curvature pairs to store (default 30 but changes to # variables if less)
    :param tr_tol: trust region subproblem tolerance (default 1e-6)
    :param max_delta: maximum allowable trust region radius (default 1e6)
    :param sr1_tol: determines helps sets curvature pair update rule for SR1
    :param delta_init: intial trust region radius, if not specified, delta_init = ||g(x0)||
    :param gamma: constant by which identity in approximate hessian is scaled by (default None)
    :param verbose: print intermediate values (default False)
    :param store_history: store additional information on problem being solves (default False). If true, it is
                            recommended to specify a write_folder otherwise it will print to the current directory
    :param norm_to_use: which norm to use for stopping criteria (default is 2-norm)
    :param write_folder: specifies the folder to which history will be written
    :return: matlab like structure object, in particular return util_func.structtype(x=x, fun=f, jac=g,
                message=message, success=success, nfev=sum(precision_counter.values()), nit=i,
                precision_counts=precision_counter, time_counter=time_counter)
    """

    # initialize history lists
    f_hist = list()
    prec_hist = list()
    inf_norm_hist = list()
    two_norm_hist = list()

    it_f = list()
    it_prec_hist = list()
    it_norm_hist = list()
    it_two_norm_hist = list()
    it_step_size_hist = list()
    it_delta_hist = list()
    it_criteria_hist = list()

    # Set the initial iterate
    x = x0
    n = x.shape[0]
    machine_eps = np.finfo(float).eps
    fplus = []

    # initialize counters
    precision_counter = {}
    time_counter = {}
    inv_precision_dict = {}

    # construct and initialize dictionaries for use later
    for key, value in precision_dict.items():
        precision_counter[key] = 0
        inv_precision_dict[value] = key
        time_counter[key] = 0.0

    # always start at the lowest precision (for curr_prec) and set high precision to the max
    high_prec_idx = max(precision_dict.values())
    curr_prec_idx = min(precision_dict.values())

    # set string value of "numerical precision" and starting precision
    high_prec_str = inv_precision_dict[high_prec_idx]
    curr_prec_str = inv_precision_dict[curr_prec_idx]

    # first evaluation
    ti = time.time()
    f, g = fun(x, curr_prec_str)
    tf = time.time()
    time_counter[curr_prec_str] += tf - ti
    precision_counter[curr_prec_str] += 1

    if store_history:
        # store values if instructed (this will store every function evaluation)
        f_hist.append(f)
        prec_hist.append(curr_prec_str)
        inf_norm_hist.append(norm(g, np.inf))
        two_norm_hist.append(norm(g))


    # set initial TR radius to gradient norm if not passed in
    delta = norm(g)/100 if delta_init is None else delta_init

    # Initialize approximate Hessian, using max memory or one more than size of problem
    memory = min(max_memory, n+1)
    S = []
    Y = []

    first_fail = True
    first_success = 0
    theta = 0.0
    gamma = 1
    
    if write_to is not None:
        df = pd.DataFrame(columns=['iter','funcval', 'gradnorm', 'x'])
        df.loc[0] = [0, f, norm(g), x]

    # begin main algorithm
    for i in range(max_iter):
        # criticality check
        if norm(g, norm_to_use) <= gtol or delta/gamma <= np.sqrt(machine_eps):
            # are we on the highest precision?
            if curr_prec_idx == high_prec_idx:
                if norm(g, norm_to_use) <= gtol:
                    # terminate, because we're done
                    message = 'First order condition met'
                    success = True
                    print(message) if verbose else None
                else:
                    # can't get sufficient decrease with small radius so just quit
                    message = 'TR radius too small'
                    success = False
                    print(message) if verbose else None

                # break out of loop if first order condition is met
                break
            else:
                # when not at highest precision, increase precision for more accuracy
                curr_prec_idx += 1
                curr_prec_str = inv_precision_dict[curr_prec_idx]
                print("Permanently switching evaluations to precision level ", curr_prec_str) if verbose else None

                # function evaluation
                ti = time.time()
                f, g = fun(x, curr_prec_str)
                tf = time.time()
                time_counter[curr_prec_str] += tf - ti
                precision_counter[curr_prec_str] += 1

                if store_history:
                    f_hist.append(f)
                    prec_hist.append(curr_prec_str)
                    inf_norm_hist.append(norm(g, np.inf))
                    two_norm_hist.append(norm(g))

                # does the current iterate satisfy first order condition
                if norm(g, norm_to_use) <= gtol:
                    # terminate, because we're done
                    message = 'First order condition met'
                    success = True
                    print(message) if verbose else None

        # solve TR subproblem
        s, crit = util_func.CG_Steinhaug_matFree(tr_tol, g, delta, S, Y, gamma, verbose=False, max_it=10*max_memory)

        # estimate the obj. reduction predicted by TR minimizer (should be greater than zero here)
        predicted_reduction = np.sum(-0.5*(s.T@util_func.Hessian_times_vec(Y, S, gamma, s)) - s.T@g)

        s = s.reshape((n,))
        if predicted_reduction <= 0:
            print('Step gives model function increase of', -predicted_reduction) if verbose else None

        # evaluate function at new trial point
        ti = time.time()
        fplus, gplus = fun(x + s, curr_prec_str)
        tf = time.time()
        time_counter[curr_prec_str] += tf - ti
        precision_counter[curr_prec_str] += 1

        # calculate actual reduction (should be greater than zero) and set rho
        actual_reduction = f - fplus   #
        rho = actual_reduction/predicted_reduction

        # for Hessian updating:
        gprev = g
        #print('rho =', rho) if verbose else None

        # does the model give enough of a decrease for acceptance?
        if (rho < eta_good) or (predicted_reduction < 0):
            # the iteration was a failure
            xnew = x

            # do we update the precision?
            if first_fail:
                first_fail = False
                if curr_prec_idx < high_prec_idx:
                    temp_prec_str = high_prec_str  #
                    print("Probed a pair of function evaluations at precision level ", temp_prec_str) if verbose else None
                    ti = time.time()
                    ftemp, gtemp = fun(x, temp_prec_str)
                    ftempplus, gtempplus = fun(x+s, temp_prec_str)
                    tf = time.time()

                    time_counter[temp_prec_str] += tf - ti
                    precision_counter[temp_prec_str] += 2

                    # initial theta (how different are evals at different precision, if big, increase precision)
                    theta = abs((f-fplus)-(ftemp-ftempplus))
                    print('Initial theta is ', theta) if verbose else None
                    if np.isnan(theta) or np.isinf(theta):
                        theta_old = theta
                        theta = np.sqrt(machine_eps)
                        print('Since initial theta was', theta_old, 
                              'which is a problem, theta has been changed to', theta) if verbose else None


            # check that the predicted reduction is quite small and precision is less than max and TR radius is small.
            # reason for casting into array was to satisfy type mismatches when using JAX on a GPU
            #if np.asarray(theta).item() > np.asarray(eta_good).item()*np.asarray(predicted_reduction).item() \
            #        and np.asarray(curr_prec_idx).item() < np.asarray(high_prec_idx).item() \
            #       and np.asarray(delta) < min(1.0, np.asarray(norm(g)).item()):
            if float(theta) > float(eta_good)*float(predicted_reduction) \
                    and float(curr_prec_idx) < float(high_prec_idx) and float(delta) < min(1.0, float(norm(g))):

                temp_prec_str = high_prec_str
                print("Probed a pair of function evaluations at ", temp_prec_str) if verbose else None

                # function evaluation at next precision
                ti = time.time()
                ftemp, gtemp = fun(x, temp_prec_str)
                ftempplus, gtempplus = fun(x+s, temp_prec_str)
                tf = time.time()
                time_counter[temp_prec_str] += tf - ti
                precision_counter[temp_prec_str] += 2

                theta = abs((f-fplus)-(ftemp-ftempplus))
                print('Changing theta to', theta) if verbose else None

                # is the model suspect due to limitations in precision?
                if theta > eta_good*predicted_reduction:
                    # predicted reduction is below precision switching floor
                    curr_prec_idx += 1
                    curr_prec_str = inv_precision_dict[curr_prec_idx]
                    print("Permanently switching to precision level ", curr_prec_str) if verbose else None
                    f = ftemp
                    g = gtemp
                    gplus = gtempplus
                    gprev = gtemp
                    delta = min(1.0, norm(g))
                    Y = []
                    S = []
            else:
                # the model quality isn't suspect, standard reject
                delta = gamma_dec*delta

        else:
            # the step is at least good
            if first_success == 0:
                first_success = 1

            # set new iterates
            xnew = x + s
            f = fplus
            g = gplus

            # is the step great (do we get desired reduction from model and are we close to TR radius)?
            if (norm(s) > 0.8*delta) and (rho > eta_great):
                delta = min(max_delta, gamma_inc*delta)

        # set gamma if it's not specified
        y = gplus-gprev
        if first_success == 1:
            first_success = 2
            if gamma is None:
                gamma = (norm(y)**2) / np.dot(s,y)

        # update the Hessian
        if first_success > 0:
            # ensure that data types are compatible
            if y.dtype != s.dtype:
                y = np.array(y, dtype=s.dtype)

            Y, S = util_func.updateYS(Y, S, y, s, memory, gamma, sr1_tol=sr1_tol, verbose=verbose)


        # get ready for next iteration
        x = xnew

        norm_s = 0 if (f != fplus) else norm(s)
        if store_history:
            # note: f, g, and delta are used in next iteration but s, crit, and curr_prec were used to get f and g
            norm_g_inf = norm(g, np.inf)
            norm_g_two = norm(g)

            f_hist.append(np.ndarray.item(np.array(f)))
            prec_hist.append(curr_prec_str)
            inf_norm_hist.append(norm_g_inf)
            two_norm_hist.append(norm_g_two)

            it_f.append(np.ndarray.item(np.array(f)))
            it_prec_hist.append(curr_prec_str)
            it_norm_hist.append(norm_g_inf)
            it_two_norm_hist.append(norm_g_two)
            it_step_size_hist.append(norm_s)
            it_delta_hist.append(delta)
            it_criteria_hist.append(crit)

        if write_to is not None:
            df.loc[i] = [i+1, f, norm(g), x]

        if verbose:
            # f, g, and delta are values used for next TR solve norm_s and criteria are both from previous iterate
            #print("iter: ", i+1, "rho: ", np.round(rho, 4), ", f: ", f, ", new_delta: ", delta, ", norm(g): ", norm(g), "norm(s): ", norm_s,
            #      ' stopping criteria:', crit, "||g||_inf:", norm(g, np.inf))
            if i % print_every == 0:
                print( str(i)+'. f(x) =', '{:6.2e}'.format(f), ', ||g(x)|| =', '{:6.2e}'.format(norm(g)), '||s|| =', '{:6.2e}'.format(norm_s), 'delta =', '{:6.2e}'.format(delta), ', rho =', '{:6.2e}'.format(rho))
        # are we actually getting numbers from function evaluations?
        if np.isnan(f) or np.isinf(f):
            message = 'Obj is nan or +/-inf'
            success = False
            print('Obj is nan or +/-inf') if verbose else None
            break

        # early termination if progress seems unlikely
        if norm(g, norm_to_use) < gtol:
            if curr_prec_idx == high_prec_idx:
                message = 'First order condition met'
                success = True
                print(message) if verbose else None
                break
    # end of main TR loop

    # did we stop because max number of iterations was realized?
    if i == (max_iter-1):
        message = "Exceed max iterations"
        success = False

    # return structure type
    ret = util_func.structtype(x=x, fun=f, jac=g, message=message, success=success, nfev=sum(precision_counter.values()),
                                  nit=i, precision_counts=precision_counter, time_counter=time_counter)

    if write_to is not None:
        df.to_csv(write_to, index=False)

    if store_history:
        # unlike other file, this store every
        ret.f_hist = np.array(f_hist)
        ret.prec_hist = prec_hist
        ret.g_inf_norm_hist = inf_norm_hist
        ret.g_two_norm_hist = two_norm_hist

        ret.it_f_hist = np.array(it_f)
        ret.it_prec_hist = it_prec_hist
        ret.it_g_inf_norm_hist = it_norm_hist
        ret.it_g_two_norm_hist = it_two_norm_hist
        ret.it_step_size_hist = it_step_size_hist
        ret.it_delta_hist = it_delta_hist
        ret.it_criteria_hist = it_criteria_hist

    return ret






