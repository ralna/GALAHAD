% GALAHAD_BLLS -
%
%  Given an o by n matrix Ao, an o-vector b, and a constant sigma >= 0, find
%  a local mimimizer of the BOUND_CONSTRAINED LINEAR LEAST-SQUARES problem
%    minimize 0.5 || Ao x - b ||_W^2 + 0.5 sigma ||x-x_s||^2
%    subject to x_l <= x <= x_u,
%  where ||v||^2 = v' v and ||v||_W^2 = v' W v, using a projection method.
%  Advantage is taken of sparse Ao.
%
%  Simple usage -
%
%  to solve the bound-constrained liner least-squares problem
%   [ x, inform, aux ]
%     = galahad_blls( Ao, b, sigma, x_l, x_u, x_0, w, x_s, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_blls( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform, aux ]
%     = galahad_blls( 'existing', Ao, b, sigma, x_l, x_u, x_0, w, x_s, control )
%
%  to remove data structures after solution
%   galahad_blls( 'final' )
%
%  Usual Input -
%    Ao: the o by n matrix Ao
%    b: the o-vector b
%    sigma: the regularization weight (sigma>0)
%    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
%    x_u: the n-vector x_u. The value inf should be used for infinite bounds
%    x_0: an initial estimate of the solution
%
%  Optional Input -
%    w: the o-vector of weights w for which W=diag(w) (= 1 if w is 
%       not specified)
%    x_s: the n-vector of shifts x_s (= 0 if x_s is not specified)
%       ** N.B. If x_s is required and n=o, w and x_s should 
%          both be provided in that order, even if defaults are used
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type BLLS_CONTROL as described in the
%      manual for the fortran 90 package GALAHAD_BLLS.
%      In particular if the weight sigma is nonzero, it
%      should be passed via control.weight.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/blls.pdf
%
%  Usual Output -
%   x: a global minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type BLLS_INFORM as described in the manual for
%      the fortran 90 package GALAHAD_BLLS.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/blls.pdf
%   aux: a structure containing Lagrange multipliers and constraint status
%    aux.z: dual variables corresponding to the bound constraints
%           x_l <= x <= x_u
%    aux.r: values of the residuals r(x) = Ao x - b
%    aux.g: values of the gradients g(x) = Ao^T W r(x)
%    aux.x_status: vector indicating the status of the bound constraints
%           x_status(i) < 0 if (x_l)_i = (x)_i
%           x_status(i) = 0 if (x_i)_i < (x)_i < (x_u)_i
%           x_status(i) > 0 if (x_u)_i = (x)_i
%
% This version copyright Nick Gould for GALAHAD productions 25/March/2026
