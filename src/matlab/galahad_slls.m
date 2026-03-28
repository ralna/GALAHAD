% GALAHAD_SLLS -
%
%  Given an o by n matrix Ao, an o-vector b, and a constant sigma >= 0, find
%  a local mimimizer of the REGULARIZED SIMPLEX_CONSTRAINED LINER 
%  LEAST-SQUARES problem
%    minimize 0.5 || Ao x - b ||_W^2 + 0.5 sigma || x - x_s ||^2
%    subject to sum_{C_j} x_i = 1, x_{C_j} >= 0 for j = 1,...,m
%    where ||v||^2 = v' v and ||v||_W^2 = v' W v, using a projection method.
%  Advantage is taken of sparse Ao.
%
%  Simple usage -
%
%  to solve the simplex_constrained liner least-squares problem
%   [ x, inform, aux ]
%     = galahad_slls( Ao, b, sigma, x_0, cohort, w, x_s, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_slls( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform, aux ]
%     = galahad_slls( 'existing', Ao, b, sigma, x_0, cohort, w, x_s, control )
%
%  to remove data structures after solution
%   galahad_slls( 'final' )
%
%  Usual Input -
%    A: the o by n matrix A
%    b: the o-vector b
%    sigma: the regularization weight (sigma>0)
%    x_0: an initial estimate of the solution
%
%  Optional Input -
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
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type SLLS_CONTROL as described in the
%      manual for the fortran 90 package GALAHAD_SLLS.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/slls.pdf
%
%  Usual Output -
%   x: a global minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type SLLS_INFORM as described in the manual for
%      the fortran 90 package GALAHAD_SLLS.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/slls.pdf
%   aux: a structure containing Lagrange multipliers and constraint status
%    aux.y: Largrange multipliers y corresponding to the simplex constraints
%    aux.z: dual variables z corresponding to the non-negativity constraints
%         x_i >= 0
%    aux.r: values of the residuals r(x) = Ao x - b
%    aux.g: values of the gradients g(x) = Ao^T W r(x)
%    aux.x_status: vector indicating the status of the bound constraints
%            x_status(i) < 0 if (x)_i = 0
%            x_status(i) = 0 if (x)_i > 0
%
% This version copyright Nick Gould for GALAHAD productions 30/December/2023
