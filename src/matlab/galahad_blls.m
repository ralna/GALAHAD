% GALAHAD_BLLS -
%
%  Given a symmetric n by n matrix H, an m by n matrix A, an n-vector
%  g, a constant f, and n-vectors x_l <= x_u, find a local mimimizer
%  of the BOUND_CONSTRAINED LINER LEAST-SQUARES problem
%    minimize 0.5 || A x - b ||^2 + 0.5 sigma ||x||^2
%    subject to x_l <= x <= x_u
%  using a projection method.
%  Advantage is taken of sparse A.
%
%  Simple usage -
%
%  to solve the bound-constrained quadratic program
%   [ x, inform, aux ]
%     = galahad_blls( A, b, x_l, x_u, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_blls( 'initial' )
%
%  to solve the bound-constrained QP using existing data structures
%   [ x, inform, aux ]
%     = galahad_blls( 'existing', A, b, x_l, x_u, control )
%
%  to remove data structures after solution
%   galahad_blls( 'final' )
%
%  Usual Input -
%    A: the m by n matrix A
%    b: the m-vector b
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
% 4/December/2009
