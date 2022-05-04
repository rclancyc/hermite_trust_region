import sys
import numpy as np
import pandas as pd
import pickle
from numpy import hstack, vstack, ones, zeros
from numpy.linalg import norm
import numpy.linalg as LA
import trustregion
import util_func

sys.path.append('/Users/clancy/repos/trophy/python')

fmess = "Model decrease below f tolerance"
gmess = "First order condition met"
tr_mess = "TR radius too small"
it_mess = "Max iteration"

def dfo_tr_solver(x0, objfun, delta_init=10, delta_max=1e8, eta_ok=0.001, eta_great=0.75, gamma_inc=2, 
                  gamma_dec=0.5, gtol=1e-5, max_it=10000, ftol=1e-12, Htol=1e-6, delta_min=1e-12, gradient_every=10,
                  min_num_grads=1, max_num_grads=10, print_every=10, bfgs_update=False, write_to=None):
    """
    REQUIRED
    :param x0: (vector) initial iterate
    :param objfun: (function handle) objective that returns objective function and its gradient
    
    OPTIONAL
    :param delta_init=10: (positive real) initial TR radius 
    :param delta_max=1e8: (positive real) maximum possible TR radius
    :param eta_ok=0.001: (real in [0,eta_great) ) lower bound on a successful iteration
    :param eta_great=0.75: (real in (eta_ok, 1] ) threshold for a very successful iteration
    :param gamma_inc=2: (real > 1) factor to increase TR radius by for great iteration
    :param gamma_dec=0.5: (real < 1) factor to decrease TR radius by for failed iteration
    :param gtol=1e-5: (real > 0) gradient stopping criteria
    :param max_it=10000: (positive integer) maximum number of iterations
    :param ftol=1e-12: (positive real) threshold for change in objective value between iterations
    :param Htol=1e-6: (real matrix) minimum Hessian norm to solve interpolation TR subproblem rather than Cauchy step
    :param delta_min=1e-12: (positive real) minimum allowable TR radius before stopping solver
    :param gradient_every=10: (positive integer) specify how often to use gradient 
    :param min_num_grads=1: (non-negative integer < max_num_grads) specifies the min number of gradients to use for interpolation when available
    :param max_num_grads=3: (integer > min_num_grads) maximum number of gradients to use for interpolation
    :param print_every=10: (non-negative integer) print to terminal after this many iterations
    :param bfgs_update=False: (boolean) use BFGS matrix update for regularized interpolation problem
    :param write_to=None: (string or None) file path to print output file

    :return ret:
        x: (real vector) solution or latest iteration if solver failed
        f: (real scalar) final objective function values
        f: (real vector) final gradient
        niter: (positive integer) number of iteration solver took
        fevals: (positive integer) number of function evaluations
        gevals: (positive integer) number of gradient evaluations
        success: (boolean) specifies if solver was successful in finding minimizer 
        stop_criteria: (string) gives stopping condition reached
    """ 

    # used code from https://github.com/TheClimateCorporation/dfo-algorithm/blob/master/blackbox_opt/DFO_src/dfo_tr.py on github to clarify some ideas
    stop_criteria = None
    fevals = 0
    gevals = 0
    x = x0
    delta = delta_init
    n = x0.shape[0]
    N = n*(n+1)/2

    # set number of data points we can store before becoming over-determined
    nYmax = N + n

    # generate initial points (change as we see fit, maybe not best method but at least a place holder)
    Y = initial_points(x, delta_init)
    Grads = zeros(Y.shape)

    # stack initial point and others just created then evaluate function values
    Y = vstack((x,Y))
    F = zeros((Y.shape[0],))
    for i, y in enumerate(Y):
        F[i], _ = objfun(y) 
        fevals += 1
    
    # use smallest value for initial location then sort by objective size
    idx = np.argsort(F)
    Y = Y[idx,:]
    F = F[idx] 
    x = Y[0,:]
    f = F[0]
    Y = np.delete(Y, 0, axis=0)
    F = np.delete(F, 0, axis=0)
    
    # the following conditional is for testing purposes although we might want to exponse initial gradient to solver call
    B = None
    use_initial_gradient = True
    if use_initial_gradient is True:
        # get gradient
        _, grad = objfun(x)
        gevals += 1

        # when using BFGS update, need curvature pairs, so get them
        if bfgs_update:
            pt_bfgs_curr = x
            grad_bfgs_curr = grad
            B = np.eye(n)
    else:
        # when the initial gradient is not available, set gradient and point to null values
        grad = zeros(n)
        if bfgs_update:
            grad_bfgs_curr = None
            pt_bfgs_curr = None



    # now sort by distance from current location
    Y, idx = sort_by_distance(Y,x)
    F = F[idx]
    Grads = Grads[idx,:]

    # are we writing to file, if so, store current values for function value, gradient norm, and current x value 
    if write_to is not None:
        df = pd.DataFrame(columns=['iter', 'funcval', 'gradnorm', 'x'])
        df.loc[0] = [0, F[0], norm(grad), x]
        
        
    iter = 0
    while iter < max_it:
        iter += 1

        # interpolate polynomial centered on minimum found so far. We include current gradient. 
        # If it's zero, we will ignore it within interpolate function
        g, H = interpolate_data(Y, F, Grads, x, f, min_num_grads=min_num_grads, 
                                max_num_grads=max_num_grads, approx_hess=B, gradient_at_center=grad)

        
        # check for stopping criteria
        normg = norm(g)
        if normg < gtol or delta < delta_min:
            # did we stop because the model gradient is really small?
            if normg < gtol:
                print(gmess)
                stop_criteria = gmess
                final_success = True
            else:
                # or is it because the trust region shrunk too much
                print(tr_mess)
                stop_criteria = tr_mess
                final_success = False
            break
        
        # check size of hessian, if it's big enough solve TR problem or else take Cauchy step
        if LA.norm(H, 'fro') > Htol:
            # solve tr subproblem (pip installed trustregion package, git repo at https://github.com/lindonroberts/trust-region)
            s = trustregion.solve(g, H, delta)
        else:
            # take Cauchy step
            s = -(delta/normg)*g

        # calculate model value at optimizer of TR subproblem
        mtrial = f + g.T@s + 0.5*s.T@H@s
        
        # check decrease versus model 
        xtrial = x + s

        # is it time for a gradient update
        if iter % gradient_every == 0:
            ftrial, gradtrial = objfun(xtrial)      # calculate both objective and gradient
            fevals += 1                             # increment function and gradient eval count
            gevals += 1

            # are we using a BFGS update or just a minimum norm solution?
            if bfgs_update:
                if pt_bfgs_curr is not None:
                    # set previous/current values to be used in curvature pair for approx Hessian update
                    pt_bfgs_prev = pt_bfgs_curr             
                    pt_bfgs_curr = xtrial
                    grad_bfgs_prev = grad_bfgs_curr
                    grad_bfgs_curr = gradtrial

                    # calculate curvature pair
                    ss = pt_bfgs_curr - pt_bfgs_prev
                    yy = grad_bfgs_curr - grad_bfgs_prev

                    # update Hessian approximation
                    B = update_B(B, yy, ss)
                else:
                    grad_bfgs_curr = gradtrial
                    pt_bfgs_curr = xtrial
                    
                    # use minimum norm solution at when BFGS update isn't used. We therefore penalize 
                    # Hessian approx for begin far away from zero.
                    B = np.zeros((n,n))
        else: 
            # when gradient isn't being updated...just function value and set trial gradient to zero
            ftrial, _ = objfun(xtrial)
            gradtrial = zeros(n)
            fevals += 1

        # evaluate model agreement on trust region
        ared = f - ftrial
        pred = f - mtrial
        rho = ared / pred

        # if the predicted model decrease is really small, we are probably done
        if abs(pred) < ftol and pred > 0:
            print("Predicted model decrease is less than", np.float32(ftol))
            stop_criteria = fmess 
            final_success = True
            break

        # was it a failed step?
        if rho < eta_ok:
            # failed step
            success = False

            # do we still have extra space to store data?
            if Y.shape[0] < nYmax:
                # yes: just add it to set of interpolating points
                F = hstack((F,ftrial))
                Y = vstack((Y,xtrial))
                Grads = vstack((Grads,gradtrial))
            else: 
                # no: then if the new point is closer than furthest point, overwrite furthest point with new one
                if norm(xtrial-x) < norm(Y[-1,:]-x):
                    Y[-1,:] = xtrial
                    F[-1] = ftrial 
                    Grads[-1,:] = gradtrial

            # since the step failed...we beleive the model is poor so shrink size of trust region
            delta = delta*gamma_dec
        else:
            # succeful step: it was at least ok, so accept it and add to interpolating set
            success = True

            # do we still have room?
            if  Y.shape[0] < nYmax:
                # yes: just add it to set of interpolating points
                F = hstack((F,f))
                Y = vstack((Y,x))  
                Grads = vstack((Grads,grad))  
            else:
                # no: overwrite furthest point in interpolating set
                F[-1] = f
                Y[-1,:] = x
                Grads[-1,:] = grad

            # set current iterate to trial iterate when successful
            x = xtrial
            f = ftrial
            grad = gradtrial

            # was it a great step to the edge of the trust region? If so, model is good so expand trust region
            if rho > eta_great and norm(s) > 0.9*delta:
                delta = min(delta*gamma_inc, delta_max)  
        
        
        if np.isnan(f):
            # when the algorithm returns a NaN, we want the code to break
            print('Algorithm returns NaN')
            stop_criteria = "Objective is NaN"
            final_success = False
            break

        # sort interpolating points and func/grad values by distance from the current iterate
        Y, idx = sort_by_distance(Y,x)
        F = F[idx]
        Grads = Grads[idx,:]


        # when step size and delta are too small, something is wrong so discard old function values for
        # iterates that are too far away from the current location...this should improve conditioning
        critical_step =  1e-9
        critical_delta = 1e-8
        if norm(s) < critical_step and delta < critical_delta:
            idx = norm(Y-np.tile(x, (Y.shape[0],1)), axis=1) < 100*delta
            if sum(idx) > n+1:
                Y = Y[idx,:]
                F = F[idx]
                Grads = Grads[idx,:]
            else:
                Y = Y[0:n+1,:]
                F = F[0:n+1]
                Grads = Grads[0:n+1,:]

        # write to command line or file when desired
        if iter % print_every == 0:
            print(str(iter)+'. f(x) =', '{:6.2e}'.format(f), ', ||g(x)|| =', '{:6.2e}'.format(normg), '||s|| =', '{:6.2e}'.format(norm(s)), 'delta =', '{:6.2e}'.format(delta), ', rho =', '{:6.2e}'.format(rho), 'Success =', success)
        if write_to is not None:
            df.loc[iter] = [iter, f, normg, x]
            
    # kill the process if maximum number of iterations have been exceeded
    if iter == max_it:
        stop_criteria = it_mess 
        print(it_mess)
        final_success = False


    ret = util_func.structtype(x=x, fun=f, grad=g, niter=iter, fevals=fevals, gevals=gevals, 
                              success=final_success, stop_criteria=stop_criteria)

    if write_to is not None:
        df.to_csv(write_to, index=False)
                                  
    return ret





def interpolate_data(Yraw, fraw, Grads, x_center, f_center, min_num_grads=1, max_num_grads=3, 
                     approx_hess=None, gradient_at_center=zeros((2,))):
    """
    :param Yraw: (matrix) points use for interpolation (each row is a point)
    :param fraw: (vector) function values to interpolate
    :param x_center: (vector) current iterate
    :param f_center: (scalar) function values at current iterate
    :param(optional) min_num_grads:  (default=1, non-negative integer) specifies minimum number of gradients to use for interpolation if available
    :param(optional) max_num_grads:  (default=3, non-negative integer) specifies maximum number of gradients to use for interpolation
    
    :return coeff_L: (vector) linear interpolation terms. If current iterate provided...this will just be gradient
    :return coeff_Q: (vector) returns coefficients for quadratic terms of interpolating polynomial (same info as Hessian)
    """
    # m is number of interpolation points and n is problem dimension
    m, n = Yraw.shape    
    
    # degrees of freedom in the Hessian
    N = int(n*(n+1)/2)

    # center data to conform with standard trust region setup, i.e., m_k(p) = f(x_k) + f'(x_k)^T p + 1/2 p^T f''(x_k) p
    # where p is a displacement vector from current iterate. Note that constant term is fixed, no need for
    # for affine offset (solving for intercept constant)
    # construct centered linear (and constant) portion of Vandermonde matrix
    Y = Yraw - np.tile(x_center, (m,1))
    f = fraw - f_center

    # take index of non-zero gradient rows
    Gidx = np.sum(np.abs(Grads), axis = 1) > 0
    num_grads = np.sum(Gidx)

    # extract points (and gradients) corresponding to non-zero gradients (might both be zero)
    Y_for_grads = Y[Gidx,:]    
    grads = Grads[Gidx,:]
    
    # if model center has a gradient, add it to the stack
    if norm(gradient_at_center) > 0.:
        num_grads += 1
        if Y_for_grads.shape[0] > 0:
            Y_for_grads = vstack((zeros(n), Y_for_grads))
            grads = vstack((gradient_at_center, grads))
        else:
            Y_for_grads = zeros((1,n))
            grads = gradient_at_center.reshape((1,n))
    
    # ensure that there are at most the maximum number of grads used
    if num_grads > max_num_grads:
        Y_for_grads = Y_for_grads[0:max_num_grads,:]
        grads = grads[0:max_num_grads,:]
        num_grads = max_num_grads
    
    # will interpolation matrix be over-determined? we favor function evaluations here over gradient evaluations. 
    # preference for gradients can be adjusted by choosing min_num_grads
    #if m + n*num_grads > (N + n):  # this is the original size before we start culling points from interpolating set. 
    if m + num_grads*(n+1) - num_grads**2/2 > (N + n):
        # over-determined? allocate enough rows for minimum number of gradient constraints.
        used_by_min_num_grads =  int(min_num_grads*(2*n+1)/2 - min_num_grads**2/2)
        avail_constraints = (N + n) - used_by_min_num_grads

        # can we fit all of the function interpolations in after using minimum number of gradient constraints?
        if m <= avail_constraints:
            # yes: then include all interpolation points and fit in as many gradients as possible
            avail_constraints = avail_constraints - m
            pts_to_use = m
            #grads_to_use = (avail_constraints // n) + min_num_grads  # check how many gradient slots remain then add gradients for constraints already used

            #grads_to_use = np.floor(  (n+1) +np.sqrt(n**2+2*n-2*avail_constraints)   )

            u = used_by_min_num_grads
            b = min_num_grads
            c = avail_constraints
            grads_to_use =int( (2*(n-b)+1 - np.sqrt( (2*(n-b)+1)**2 - 4*(2*(c+u) - (2*n+1)*b + b**2) ))//2 )

        else:
            # no: then take as many function interpolations as we can and the minimnum number of gradient constraints
            pts_to_use = avail_constraints
            grads_to_use = min_num_grads

        Y_for_grads = Y_for_grads[0:grads_to_use,:]
        grads = grads[0:grads_to_use,:]
        Y = Y[0:pts_to_use,:]
        f = f[0:pts_to_use]
        m = pts_to_use
    

    d = grads.shape[0]
    KKT, RHS, Slist = construct_sketched_KKT_system(Y, f, Y_for_grads, grads, approx_hess=approx_hess)
    
    try:
        U, sig, Vt = LA.svd(KKT, full_matrices=False)
        sol =  Vt.T@(np.diag(1/sig)@(U.T@RHS)) 
    except:
        try:
            Q_, R_ = LA.qr(KKT)
            sol = LA.inv(R_)@(Q_.T@RHS)
        except: 
            sol = LA.pinv(KKT, RHS)
            print('KKT system is singular, using psuedo inverse instead')

    # ell is model gradient
    ell = sol[0:n]
    lambdaQ = Slist[0].T@sol[n:]

    # construct model Hessian
    M = Y.T@np.diag(lambdaQ)@Y 
    M = M - 0.5*np.diag(np.diag(M))

    #sr = m
    if d > 0:
        for i in range(d):
            Si = Slist[i+1]        
            lami = Si.T@sol[n:]  # new set up for sketched dual variable
            M = M + np.outer(Y_for_grads[i,:], lami) + np.outer(lami, Y_for_grads[i,:]) - np.diag(lami*Y_for_grads[i,:])


    H = M if approx_hess is None else M + approx_hess
    H = 0.5*(H + H.T)


    # THIS CODE IS FOR TEST INTERPOLATION SOLUTION=================================================================
    testing_solutions = False
    if testing_solutions:
        mydict = {'Y':Y, 'f':f, 'grads':grads, 'Y_for_grads':Y_for_grads, 'Hprev':approx_hess, 'Htest':H}
        pickle.dump(mydict, open('test.pkl', 'wb'))
    return ell, H




def construct_KKT_system(Y, f, Y_for_grads, grads, approx_hess=None):
    """
    :param Y: (matrix) interpolating points centered on current iterate (each row is a point)
    :param f: (vector) offset function values for interpolation
    :param Y_for_grads: (matrix) centered points to use for gradient interpolation (each row is a point)
    :param grads: (matrix) gradients correspond to points in Y_for_grads (each row is a gradient)
    :param approx_hess: (matrix, defaults to none) optional argument for approximate hessian 
                        used for and offset for regularization

    :return KKT: (matrix) KKT matrix 
    :return RHS: (vector) right hand side for KKT equation
    """
    m,n = Y.shape
    # precompute matrices
    Yt = Y.T
    YYt = Y@Yt
    Y_sq = Y*Y
    QQt = 0.5*YYt*YYt - 0.25*Y_sq@Y_sq.T

    # construct upper left block of KKT matrix
    UL = zeros((n+m,n+m))
    UL[n:n+m, 0:n] = Y                          
    UL[0:n, n:n+m] = Y.T
    UL[n:n+m, n:n+m] = QQt
    URHS = hstack((zeros(n), f))            
    if approx_hess is not None:
        temp1 = np.asarray([0.5*y@(approx_hess@y) for y in Y])
        URHS[n:] = URHS[n:] - temp1
    
    # when gradients are to be used, construct other matrix blocks for KKT matrix
    d = grads.shape[0]
    if d > 0:
        # construct lower left block, and upper right block
        LL = zeros((n*d, n+m))
        UR = zeros((n+m, n*d))
        for i, x in enumerate(Y_for_grads):
            LL[i*n:(i+1)*n, 0:n] = np.eye(n)    ## OLD CONVENTION: LL[i*n:(i+1)*n, 0:n] = -np.eye(n)
            UR[0:n, i*n:(i+1)*n] = np.eye(n)

            DiQt = Yt * (np.outer(ones(n), x@Yt) - 0.5*Yt*np.outer(x, ones(m)))       
            LL[i*n:(i+1)*n, n:n+m] = DiQt
            UR[n:n+m, i*n:(i+1)*n] = DiQt.T
        
        # construct lower right block
        LR = zeros((n*d,n*d))    
        LRHS = zeros(n*d)
        for i, x in enumerate(Y_for_grads): #  range(d):
            LRHS[i*n:(i+1)*n] = grads[i]     ## OLD CONVENTION: LRHS[i*n:(i+1)*n] = -grads[i] 
            if approx_hess is not None:
                temp2 = approx_hess@x
                LRHS[i*n:(i+1)*n] = LRHS[i*n:(i+1)*n] - temp2

            for j in range(i,d):
                z = Y_for_grads[j,:]
                DiDj = np.outer(z, x) + np.dot(z, x)*np.eye(n) - np.diag(z*x)
                LR[i*n:(i+1)*n, j*n:(j+1)*n] = DiDj
                LR[j*n:(j+1)*n, i*n:(i+1)*n] = DiDj.T

        # construct KKT system to solve
        KKT = vstack((   hstack((UL, UR)) ,  hstack((LL, LR))     ))
        RHS = hstack((URHS, LRHS))
    else: 
        KKT = UL
        RHS = URHS
        #print("No gradients being used")


    return KKT, RHS





def construct_sketched_KKT_system(Y, f, Y_for_grads, grads, approx_hess=None):
    """
    :param Y: (matrix) interpolating points centered on current iterate (each row is a point)
    :param f: (vector) offset function values for interpolation
    :param Y_for_grads: (matrix) centered points to use for gradient interpolation (each row is a point)
    :param grads: (matrix) gradients correspond to points in Y_for_grads (each row is a gradient)
    :param approx_hess: (matrix, defaults to none) optional argument for approximate hessian 
                        used for and offset for regularization

    :return KKT: (matrix) KKT matrix 
    :return RHS: (vector) right hand side for KKT equation



    WE SHOULD ADD MORE OPTIONAL ARGS LIKE NUMBER OF FUNCTION EVALS TO USE, FINAL DIMENSION OF SKETCH, ETC.
    """
    m,n = Y.shape
    # precompute matrices
    Yt = Y.T
    YYt = Y@Yt
    Y_sq = Y*Y
    QQt = 0.5*YYt*YYt - 0.25*Y_sq@Y_sq.T
    v = 0

    # determine number of rows to discard for each sketching block and indices they will appear in the sketching matrix.
    d = grads.shape[0]
    Slist = []
    if d > 1:
        rw = np.zeros(d+2, dtype=int)
        rw[1] = m

        for i in range(2, d+2):
            rw[i] = m + (i-1)*n - v
            v = v + (i-1)

        # number of dimensions for the sketch
        r = rw[-1]
        
        
        # GENERATE SKETCHING MATRICES. This can take a number of different forms. In my initial 
        # form, we have a sketching matrix for the function interpolation conditions, and then 
        # one for each derivative condition. The matrix will have the desired number of rows 
        # (subtract number of dimension that would result in dependence) but each sketching 
        # matrix will have the number of columns corresponding to number of interpolating
        # conditions given for the block.
        use_all_funvals = False
        if use_all_funvals:
            temp = np.zeros((r, m))
            temp[0:m,:] = np.eye(m)    
        else:       
            # need some way to adjust for number of rows, i.e., m will change
            temp = np.random.randn(m,m)
            W, _ = LA.qr(temp)
            temp = np.zeros((r, m))
            temp[0:m,:] = W.T
        Slist.append(temp)

        for i in range(1,d+1):
            sr = rw[i]      # start row
            er = rw[i+1]    # end row
            temp = np.random.randn(n,er-sr) # draw random Gaussian matrix
            W, _ = LA.qr(temp)              # take QR decomp to get Haar matrix
            temp = np.zeros((r, n))         # construct stetching matrix
            temp[sr:er,:] = W.T             # set rows we want to sketch to be columns of Haar matrix
            Slist.append(temp)              # append matrix to sketch matrix list
    else:
        if d == 0:
            r = m
            Slist.append(np.eye(m))
        else:
            r = m+n
            Slist.append(vstack((np.eye(m), zeros((n,m)))))
            Slist.append(vstack((zeros((m,n)), np.eye(n))))

    temp = None

    
    # construct SL and S_ 0 QQ.T S_0.T (we can thing of this as the upper left block)
    S0 = Slist[0]
    Lsum = S0@Y
    QQsum = (S0@QQt)@S0.T
    fsum = S0@f
    if d > 0:
        # loop through all the gradients we are storing
        # this is the beginning of the lower right block. Treat it in a seperate loop since using components of L and Q
        # differs from the structure of derivative constraints, i.e., I and DiDjT
        for i in range(1,d+1):
            x = Y_for_grads[i-1,:]
            g = grads[i-1,:]
            S = Slist[i]

            # construct linear sum and right hand side from sketched constraints
            Lsum = Lsum + S@np.eye(n)
            fsum = fsum + S@g

            DiQt = Yt * (np.outer(ones(n), x@Yt) - 0.5*Yt*np.outer(x, ones(m))) 
            SiDi_QtS0t = (S@DiQt)@S0.T
            QQsum = QQsum + SiDi_QtS0t + SiDi_QtS0t.T
        

        # the nested loop constructs and adds all constraints from the derivative portion
        for p in range(1, d+1):
            xp = Y_for_grads[p-1,:]
            for q in range(i,d+1):
                xq = Y_for_grads[q-1,:]
                xpxq = np.dot(xp,xq)
                DpDq = np.outer(xq, xp) + xpxq*np.eye(n) - np.diag(xp*xq)
                SpDp_DqSq = S[p]@DpDq@S[q].T
                QQsum = QQsum + SpDp_DqSq
                if p != q:
                    QQsum = QQsum + SpDp_DqSq.T
        

    # construct upper left block of KKT matrix (standard setup)
    KKT = zeros((n+r,n+r))
    KKT[n:n+r, 0:n] = Lsum                        
    KKT[0:n, n:n+r] = Lsum.T
    KKT[n:n+r, n:n+r] = QQsum
    RHS = hstack((zeros(n), fsum))            
    if approx_hess is not None:
        # construct term to subtract from RHS that is based on the shift from function interpolation conditions
        temp = np.asarray([0.5*y@(approx_hess@y) for y in Y])
        S0 = Slist[0]
        RHSoffset = S0@temp

        if d > 0:
            # loop through points that are being used to interpolate gradients and accumulate them for shift
            for i in range(1,d+1):
                Si = Slist[i]
                x = Y_for_grads[i-1,:]
                temp = approx_hess@x
                RHSoffset = RHSoffset + Si@temp

        RHS[n:] = RHS[n:] - RHSoffset 
    return KKT, RHS, Slist




def hess_to_vec(H):
    """
    :param H: (matrix) Hessian matrix

    :return q: (vector) vectorized matrix of Hessian (only looks at upper triangular 
                portion with appropriate scaling)
    """
    n = H.shape[0]
    N = n*(n+1)//2
    k = -1
    q = np.zeros(N)
    for i in range(n):
        k += 1
        q[i] = H[i,i]
    

    for i in range(n-1):
        for j in range(i+1,n):
            k += 1
            q[k] = H[i,j]
    assert k + 1 == N, "Issue with dimension for q"

    return q




def initial_points(x0, step):
    """
    :param x0: (vector) center point which we perturb 
    :param step: (positive real) step length by which we perturb along each component
    :return X: (matrix) matrix of initial interpolation points

    This function takes a point x0 of length n then perturbs each component to produce a set of n 
    points to use for an initialization of the algorithm. Also adds a very small Gaussian perturbation
    to make sure that the points are a least slightly different if we were to include plus and minus 
    each perturbation
    """
    # get problem size 
    n = x0.shape[0]                                                           

    # repeat x0 n time then perturb along diagonal by randomly chosen + or - step
    X = np.tile(x0, (n,1)) + 0.5*step*np.diag(2*(np.random.binomial(1,0.5,n)-0.5))  
    
    # perturb by small Gaussian to avoid undesired corner cases
    X = X + np.random.normal(0,1e-6,X.shape)
    return X


def sort_by_distance(X, x_current):
    """
    :param X: (matrix) each row is an interpolation point
    :param x_current: (vector) current point we are measuring distance from

    :return X_sorted: (matrix) points sorted by distance from x_current 
    :return idx: (vector of non-negative integers) index used to sort X
    """
    # constuct matrix of offsets from current point
    D = X - np.tile(x_current, (X.shape[0],1))

    # calculate each points Euclidean distance from the current point
    dist = norm(D, axis=1)

    # get sorted indices for points, closest to farthest
    idx = np.argsort(dist)

    # sort original uncentered data
    X_sorted = X[idx,:]
    return X_sorted, idx


def get_quad_mapping(n):
    """ 
    :params n: (positive integer) dimension of the problem

    :return rows: for a column of quad Vandermonde, tells which row component of hessian it corresponds to
    :return cols: for a column of quad Vandermonde, tells which column component of hessian it corresponds to
    :return mydict: for a particular component j, specificies which columns of Vandermonde it appears in 

    Takes dimension of problem and provides mapping to and from Hessian
    """
    rows = {}
    cols = {} 
    mydict = {}
    k = -1

    # get mapping for diagonal components of Hessian
    for i in range(n):
        k += 1
        rows[k] = i
        cols[k] = i
        mydict[k] = [i]
    
    # get mapping for cross terms (above diagonal)
    for i in range(n-1):
        for j in range(i+1,n):
            k += 1
            rows[k] = i
            cols[k] = j
            mydict[i].append(k)
            mydict[j].append(k)
    return rows, cols, mydict


def hessian_from_vector(q):
    """
    :param q: (vector) quadratic coefficients from solving interpolation problem
    :return H: (matrix) return Hessian matrix based on coefficient vector
    """
    N = len(q)
    n = int(0.5*(-1 + np.sqrt(1+8*N)))
    rows, cols, _ = get_quad_mapping(n)
    H = np.zeros((n,n))
    for k, H_ij in enumerate(q):
        H[rows[k],cols[k]] = H_ij
        if rows[k] != cols[k]:
            H[cols[k],rows[k]] = H_ij
    return H


def get_phi_vec(x):
    """
    :param x: (vector) point in space

    :return (phi_L, phi_Q): (vector) corresponding row of vandermonde for the given point x
    """
    n = len(x)
    N = n*(n+1)//2
    rows, cols, _ = get_quad_mapping(n)
    phi_L = x
    phi_Q = np.zeros(N)

    # loop through x and place components where appropriate
    for k in range(N):
        i = rows[k]  # this the first component of Y of the current k
        j = cols[k]  # this is the second component of Y of the current k
        const = 0.5 if k < n else 1.0  # for diagonals, multiply by 1/2 (first n columns), otherwise, by 1
        phi_Q[k] = const*x[i]*x[j]       # construct current column of Q
    return np.hstack((phi_L, phi_Q))



def get_grad_phi(x):
    """
    :param x: (vector) point in space
    
    :return phi_grad: (matrix) matrix used corresponding to linearized version of quadratic polynomial, i.e., for point
                        x and coefficients alpha, grad m(x) = <phi_grad, alpha>
    """
    n = x.shape[0]
    rows, cols, mydict = get_quad_mapping(n)
    N = len(rows)
    mat = np.zeros((n,N))
    for i in range(n):
        #active_set gives a list of which columns (k), component i is present in
        # in our case, this means that the gradients will only be active in these particular columns. 
        active_set = mydict[i]
        for j in active_set: 
            # get component indices present in the jth columns
            a = [rows[j], cols[j]]
        
            # get rid of the ith component since we don't need it
            a.remove(i)
            if a == []:
                a = [i]
            mat[i,j] = x[a[0]]
    phi_grad = np.hstack((np.eye(n), mat))
    return phi_grad



def update_B(B, y, s):
    """ 
    :param B: (matrix) current approximated Hessian
    :param y: (vector) gradient difference vector
    :param s: (vector) iterate difference vector

    :return B: (matrix) approximate Hessian updated according to BFGS rule
    """
    if LA.norm(B) > 0:
        v = B@s
        B = B + np.outer(y,y)/np.dot(y,s) - (1/np.dot(s,v))*np.outer(v,v)
    else: 
        B = np.eye(B.shape[0])
    return B




def form_Q(Y):
    """
    :param Y: (matrix) points to be used for interpolation (each row is a point)

    :return Q: (matrix) quadratic portion of Vandermonde matrix. Should be the same as applying get_phi_vec
                    to each point in Y, i.e., Q[i, :] = get_phi_vec(Y[i,:])[n+1:]
    """
    m, n = Y.shape
    N = n*(n+1)//2
    r, c, _ = get_quad_mapping(n)

    Q = np.zeros((m,N))
    for k in range(N):
        row = r[k]
        col = c[k]
        if row == col:
            Q[:,k] = 0.5*np.multiply(Y[:,row], Y[:,col])
        else:
            Q[:,k] = np.multiply(Y[:,row], Y[:,col])
    return Q
        
