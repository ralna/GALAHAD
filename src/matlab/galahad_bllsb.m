% GALAHAD_BLLSB -
%
%  Given an o by n matrix A_o, an o by o diagonal scaling matrix W,
%  an o-vector b, a constant sigma >= 0, and n-vectors x_l <= x_u ,
%  find a local minimizer of the (REGULARIZED) BOUND-CONSTRAINED
%  LINEAR LEAST-SQUARES problem
%    minimize 0.5 * || A x - b||_W^2 + 0.5 * sigma ||x||^2
%    subject to x_l <= x <= x_u
%  and where ||v||^2 = v' v and ||v||_W^2 = v' W v.
%  Advantage is taken of sparse A_o.
%
%  Simple usage -
%
%  to solve the constrained linear least-squares problem
%   [ x, inform, aux ]
%    = galahad_bllsb( A_o, b, sigma, x_l, x_u, w, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%    = galahad_bllsb( 'initial' )
%
%  to solve the convex quadratic program using existing data structures
%   [ x, inform, aux ]
%    = galahad_bllsb( 'existing', A_o, b, sigma, x_l, x_u, w, control )
%
%  to remove data structures after solution
%   galahad_bllsb( 'final' )
%
%  Usual Input -
%    A_o: the o by n matrix A_o
%    b: the o-vector b
%    sigma: the regularization parameter sigma >= 0
%    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
%    x_u: the n-vector x_u. The value inf should be used for infinite bounds
%
%  Optional Input - (either or both may be given, with w before control)
%    w: the (diagonal) components of the diagonal scaling matrix W
%    control: a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type BLLSB_CONTROL as described in the
%      manual for the fortran 90 package GALAHARS_BLLSB.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/bllsb.pdf
%
%  Usual Output -
%   x: a local minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type BLLSB_INFORM as described in the manual for
%      the fortran 90 package GALAHARS_BLLSB.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/bllsb.pdf
%  aux: a structure containing Lagrange multipliers and constraint status
%   aux.r: values of the residuals A_o * x - b
%   aux.z: dual variables corresponding to the bound constraints
%        x_l <= x <= x_u
%   aux.x_stat: vector indicating the status of the bound constraints
%           x_stat(i) < 0 if (x_l)_i = (x)_i
%           x_stat(i) = 0 if (x_i)_i < (x)_i < (x_u)_i
%           x_stat(i) > 0 if (x_u)_i = (x)_i
%
% This version copyright Nick Gould for GALAHAD productions 24/December/2023
