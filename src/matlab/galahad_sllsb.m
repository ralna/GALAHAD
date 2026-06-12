% GALAHAD_SLLSB -
%
%  Given an o by n matrix A_o, an o by o diagonal scaling matrix W,
%  an o-vector b, a constant sigma >= 0, and n-vectors x_l <= x_u,
%  find a local minimizer of the (REGULARIZED) SIMPLEX-CONSTRAINED
%  LINEAR LEAST-SQUARES problem
%    minimize 0.5 * || A_o x - b||_W^2 + 0.5 * sigma ||x-x_s||^2
%    subject to sum_{C_j} x_i = 1, x_{C_j} >= 0 for j = 1,...,m
%  where ||v||^2 = v' v and ||v||_W^2 = v' W v, using an interior-point method
%  Advantage is taken of sparse A_o.
%
%  Simple usage -
%
%  to solve the simplex-constrained linear least-squares problem
%   [ x, inform, aux ]
%    = galahad_sllsb( A_o, b, sigma, cohort, w, x_s, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%    = galahad_sllsb( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform, aux ]
%    = galahad_sllsb( 'existing', A_o, b, sigma, cohort, w, x_s, control )
%
%  to remove data structures after solution
%   galahad_sllsb( 'final' )
%
%  Usual Input -
%    A_o: the o by n matrix A_o
%    b: the o-vector b
%    sigma: the regularization parameter sigma >= 0
%
%  Optional Input - (either or both may be given, with w before control)
%    cohort: the cohorts, so that variable x_i is in cohort C_j if 
%       cohort[i] = j, and x_i is not constrained if cohort[i] = 0 
%    w: the o-vector of weights w for which W=diag(w) (= 1 if w is 
%       not specified)
%       ** N.B. If n=o and both w and cohort are provided, cohort 
%          must proceed w in the calling sequence
%    x_s: the n-vector of shifts x_s (= 0 if x_s is not specified)
%       ** N.B. If x_s is required and n=o, cohort, w and x_s should 
%          all be provided in that order, even if defaults are used. 
%          Otherwise, if cohort and x_s are both required, cohort 
%          must proceed x_s in the calling sequence
%    control: a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type SLLSB_CONTROL as described in the
%      manual for the fortran 90 package GALAHAD_SLLSB.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/sllsb.pdf
%
%  Usual Output -
%   x: a local minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type SLLSB_INFORM as described in the manual for
%      the fortran 90 package GALAHAD_SLLSB.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/sllsb.pdf
%  aux: a structure containing Lagrange multipliers and constraint status
%   aux.r: values of the residuals A_o * x - b
%   aux.z: dual variables corresponding to the bound constraints
%        x_l <= x <= x_u
%   aux.x_stat: vector indicating the status of the bound constraints
%           x_stat(i) < 0 if (x_l)_i = (x)_i
%           x_stat(i) = 0 if (x_i)_i < (x)_i < (x_u)_i
%           x_stat(i) > 0 if (x_u)_i = (x)_i
%
% This version copyright Nick Gould for GALAHAD productions 24/March/2026
