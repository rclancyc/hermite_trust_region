This repo includes code written in the Spring of 2022 towards the completion of my dissertation. The main contribution here is `hermite_tr_solver.py` which is a trust region solver that interpolates function and gradients to build models for the trust region subproblem. It is worth noting that this is experimental code which likely contains bugs I am unaware. I make no guarantees on the accuracy. 


The rest of the repo is code to either test the solver via experiments, supporting code for the experiments, or to check the solvers components. I have culled 90% of the code from my local directory to make this more pallatable, but am happy to answer any questions you may have. 

This repos contains the following code:
1. `hermite_tr_solver.py`: main solver that interpolates function and gradient values for trust region subproblem 
2. `trophy.py`: TROPHY algorithm which is a mixed precision trust region method that uses SR1 updates for the Hessian. This is what I worked on at Argonne but only used for comparison purposes here. The method uses gradients at every iteration so we expect it should do better than our method. Not that this needs a "precision dictionary" for precision hierarchy...we do this by treating double precision as though it's the only precision in the dictionary. 
3. `util_func`: utility functions that support TROPHY and other scripts.
4. `run_single_problem`: runs hermite interpolation based method against standard inteprolation and SR1 solver and writes results if desired. Mainly used for trouble shooting, debugging and sanity checking. 
5. `runall_problems`: run all problems to generate plots and check summary statistics for different parameter setups like min or max number of gradients and how frequently new gradients should be incorporated into the TR subproblem.
6. `pycutest_for_trophy` folder: python interface to CUTEst test set with changes made to accomodate multiple precisions. 
7. `plotting_for_problems_by_iteration.ipynb`: Jupyter notebook generating plots used in dissertation. 

----

Since there is considerable overhead getting the pycutest set up and running as well as building the problems using `pycutest_for_trophy`, it might be worthwhile to use a readily available function handle in  `run_single_problem` and drop the use of the TROPHY algorithm. I will try adding instructions here in the coming days concerning how to build the problems used in my disseration. 

