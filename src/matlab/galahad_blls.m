% GALAHAD_BLLS -
%
%  Given an o by n matrix Ao, an o-vector b, and a constant sigma >= 0, find
%  a local mimimizer of the BOUND_CONSTRAINED LINER LEAST-SQUARES problem
%    minimize 0.5 || Ao x - b ||^2 + 0.5 sigma ||x||^2
%    subject to x_l <= x <= x_u
%  using a projection method.
%  Advantage is taken of sparse Ao.
%
%  Simple usage -
%
%  to solve the bound-constrained liner least-squares problem
%   [ x, inform, aux ]
%     = galahad_blls( Ao, b, x_l, x_u, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_blls( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform, aux ]
%     = galahad_blls( 'existing', Ao, b, x_l, x_u, control )
%
%  to remove data structures after solution
%   galahad_blls( 'final' )
%
%  Usual Input -
%    Ao: the o by n matrix Ao
%    b: the o-vector b
%    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
%    x_u: the n-vector x_u. The value inf should be used for infinite bounds
%
%  Optional Input -
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
%         x_l <= x <= x_u
%    aux.x_stat: vector indicating the status of the bound constraints
%            x_stat(i) < 0 if (x_l)_i = (x)_i
%            x_stat(i) = 0 if (x_i)_i < (x)_i < (x_u)_i
%            x_stat(i) > 0 if (x_u)_i = (x)_i
%
% This version copyright Nick Gould for GALAHAD productions
% 30/December/2023
