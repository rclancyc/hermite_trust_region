#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019 Albert Berahas, Majid Jahani, Martin Takáč
#
# All Rights Reserved.
#
# Authors: Albert Berahas, Majid Jahani, Martin Takáč
#
# Please cite:
#
#   A. S. Berahas, M. Jahani, and M. Takáč, "Quasi-Newton Methods for
#   Deep Learning: Forget the Past, Just Sample." (2019). Lehigh University.
#   http://arxiv.org/abs/1901.09997


# Edits made by Richie Clancy, summer 2021
# ==========================================================================

import numpy as np
from numpy import linalg as LA
from numpy.linalg import norm

dbug = False
# ==========================================================================


def pycutest_wrapper(x, prec, single_func, double_func):
    if prec == 'single':
        f, g = single_func.obj(x, gradient=True)
    elif prec == 'double':
        f, g = double_func.obj(x, gradient=True)
    else:
        print('Only recognize single or double for pyCUTEst')
    return f, g


def rootFinder(a,b,c):
    """return the root of (a * x^2) + b*x + c =0"""
    r = b**2 - 4*a*c

    if r > 0:
        num_roots = 2
        x1 = ((-b) + np.sqrt(r))/(2*a+0.0)
        x2 = ((-b) - np.sqrt(r))/(2*a+0.0)
        x = max(x1,x2)
        if x>=0:
            return x
        else:
            print("no positive root!")
    elif r == 0:
        num_roots = 1
        x = (-b) / (2*a+0.0)
        if x>=0:
            return x
        else:
            print("no positive root!")
    else:
        print("No roots")


def updateYS(Y, S, y, s, memory, gamma, sr1_tol=1e-4, verbose=False):
    """
    :param Y: matrix of gradient differences
    :param S: matrix of displacements
    :param y: change in gradient
    :param s: change in x, i.e., most recent step
    :param memory: memory allocation
    :param gamma: scaling for identity, often just set to 1
    :param sr1_tol: helps determine when to update Y and S
    :param verbose: True or false. When True, will alert user when curvature pair has been rejected
    :return: updated Y and S
    """
    # are we storing any curvature pairs? if not, next step will be in direction of negative gradient
    if memory > 0:

        pred_grad = y - Hessian_times_vec(Y, S, gamma, s)
        dot_prod = np.dot(pred_grad, s)
        # will pair cause instability by being
        if abs(dot_prod) > sr1_tol*norm(pred_grad)*norm(s):

            # checks to see if there is currently a curvature pair stored (Y and S are empty lists if none stored)
            if not isinstance(Y, list):

                # have we maxed out storage?
                if Y.shape[1] >= memory:
                    # all memory used up, delete the oldest (y,s) pair,since indexed from zero, start at 1 not 2 and go to memory
                    Y = Y[:, 1:memory]
                    S = S[:, 1:memory]

            # add newest (y,s) pair as last columns of Y and S respectively
            if isinstance(Y, list):
                Y = y.reshape((y.shape[0],1))
                S = s.reshape((s.shape[0],1))
            else:
                Y = np.hstack((Y,y.reshape((y.shape[0],1))))
                S = np.hstack((S,s.reshape((s.shape[0],1))))


            # check to see if we should continue storing older curvature pairs based on positive deifinite criteria.
            # Used a rule here based on paper Matt provided, but of course, I can't find it now.
            keep_going = True
            while keep_going:
                if S.shape[1] > 0:
                    Psi = Y - gamma*S
                    Minv = np.matmul(S.T, Y) - gamma*np.matmul(S.T, S)
                    tmp = np.min(LA.eig(np.matmul(Psi.T, Psi))[0])
                    if tmp > 0 and LA.det(Minv) != 0:
                        keep_going = False
                    else:
                        S = S[:, 1::]
                        Y = Y[:, 1::]
                else:
                    keep_going = False

        else:
            print('Not updating Y and S') if verbose else None

    return Y, S


def Hessian_times_vec(Y, S, gamma, v):
    """
    :param Y: matrix of gradient differences
    :param S: matrix of displacements
    :param gamma: scaling for identity, often just set to 1
    :param v: vector to be multiplied
    :return B_v: low rank Hessian approximate times vector v

    Use form given in
    Byrd, Richard H., Jorge Nocedal, and Robert B. Schnabel.
    "Representations of quasi-Newton matrices and their use in limited memory methods."
    Mathematical Programming 63.1 (1994): 129-156.
    """
    nv = len(v)
    if not isinstance(Y, list):
        temp1 = np.matmul(S.T, Y)
        M = np.tril(temp1) + np.triu(temp1.T, 1) - gamma*np.matmul(S.T, S)
        try:
            Minv = LA.inv(M)
        except:
            # if inverse doesn't exist, use psuedo inverse instead (could probably just use this instead to be safe)
            Minv = LA.pinv(M)

    else:
        Minv = np.zeros((1, 1))
        Y = np.zeros((nv, 1))
        S = np.zeros((nv, 1))

    # Bk is approximation of Hessian...not it's inverse. This is ``matrix''-vector multiply here
    G = (Y - gamma*S)
    tmp1 = np.matmul(G.T, v)
    tmp2 = np.dot(Minv, tmp1)
    B_v = np.matmul(G, tmp2) + gamma*v
    return B_v


def CG_Steinhaug_matFree(eps, g, delta, S, Y, gamma, verbose=False, max_it=None):
    """
    :param eps: subproblem tolerance
    :param g: gradient for use in model function
    :param delta: trust region radius
    :param S: step/displacement matrix
    :param Y: gradient difference matrix
    :param gamma: identity scaling factor
    :param verbose: print details if set to True
    :param max_it: maximum number of iterations to take for conjugate gradient
    :return s, criteria: step vector and stopping criteria that was satisfied
    """
    nv = len(g)

    # since CG should converge in at most nv iterations, prevent invinite loop from instability (usually set much lower)
    if max_it is None:
        max_it = 3*nv

    # initialize vectors
    zOld = np.zeros((nv, 1))
    try:
        g.shape[1]
    except:
        g = g.reshape((nv, 1))
    rOld = g
    dOld = -g
    keep_going = True
    norm_rOld = norm(rOld)

    # dide we start with the solution? if so, we're already done
    if norm_rOld < eps:
        p = zOld
        return p, "small residual"

    # use compact limited form to generate matrix vector product
    if not isinstance(Y, list):
        temp1 = np.matmul(S.T, Y)
        M = np.tril(temp1) + np.triu(temp1.T, 1) - gamma*np.matmul(S.T, S)
        try:
            Minv = LA.inv(M)
        except:
            Minv = LA.pinv(M)
    else:
        Minv = np.zeros((1, 1))
        Y = np.zeros((nv, 1))
        S = np.zeros((nv, 1))

    # set G and G_T so we don't need to keep transposing for big matrices.
    j = 0
    G = Y-gamma*S
    G_T = np.array(G.T, order='C')

    while keep_going:
        # calculate ``Hessian''-vector product
        temp1 = np.matmul(G_T, dOld)
        temp2 = np.matmul(Minv, temp1)
        B_dOld = np.matmul(G, temp2) + gamma*dOld
        dBd = np.matmul(dOld.T, B_dOld)

        # does direction have negative curvature?
        if dBd <= 0:
            # find tau that gives minimizer
            tau = rootFinder(norm(dOld)**2, 2*np.dot(zOld.T, dOld), (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld

            if dBd == 0:
                print("The matrix is indefinite") if verbose else None
            return p, "neg. curve",

        alphaj = norm_rOld**2 / dBd
        zNew = zOld + alphaj*dOld

        # stop if we exceed trust region radius (we know the norm of the step will continue increasing beyond radius)
        # It's work noting that this should be a descent direction so shouldn't harm us by stopping early.
        if norm(zNew) >= delta:
            tau = rootFinder(norm(dOld)**2, 2*np.dot(zOld.T, dOld), (norm(zOld)**2 - delta**2))
            p = zOld + tau*dOld
            return p, "exceed TR"

        rNew = rOld + alphaj*B_dOld
        norm_rNew = norm(rNew)

        # have we converged or taken to many iterations
        if norm_rNew <= eps or j > max_it:
            p = zNew
            if norm_rNew > eps:
                print('CG should have converged by now') if verbose else None
                return p, "Too many CG iterations"
            else:
                return p, "Success in TR"

        betaNew = norm_rNew**2/norm_rOld**2
        dNew = -rNew + betaNew*dOld

        dOld = dNew
        rOld = rNew
        zOld = zNew
        norm_rOld = norm_rNew
        j += 1

"""
def get_quad_mapping(n):
    rows = {}
    cols = {} 
    mydict = {}
    k = -1
    for i in range(n):
        k = k + 1
        rows[k] = i
        cols[k] = i
        mydict[k] = [i]
    
    for i in range(n-1):
        for j in range(i+1,n):
            k = k + 1
            rows[k] = i
            cols[k] = j
            mydict[i].append(k)
            mydict[j].append(k)
    return rows, cols, mydict
         

def hessian_from_vector(q):
    N = len(q)
    n = int(0.5*(-1 + np.sqrt(1+8*N)))
    rows, cols, _ = get_quad_mapping(n)
    H = np.zeros((n,n))
    for k, H_ij in enumerate(q):
        H[rows[k],cols[k]] = H_ij
        if rows[k] != cols[k]:
            H[cols[k],rows[k]] = H_ij

    return H
"""

class structtype():
    # pulled from https://stackoverflow.com/questions/11637045/complex-matlab-like-data-structure-in-python-numpy-scipy
    # this just allows us to return structure type return from solver like scipy.optimize (should probably figure out
    # how to do this using exact same class type aa scipy
    def __init__(self, **kwargs):
        self.Set(**kwargs)
    def Set(self, **kwargs):
        self.__dict__.update(kwargs)
    def SetAttr(self, lab, val):
        self.__dict__[lab] = val
    def ListVariables(self):
        names = dir(self)
        for name in names:
            # Print the item if it doesn't start with '__'
            if '__' not in name and 'Set' not in name and 'SetAttr' not in name and 'ListVariables' not in name:
                myvalue = self.__dict__[name]
                print(name, ':', myvalue)

