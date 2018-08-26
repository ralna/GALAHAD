% GALAHAD_LPB -
%
%  Given an m by n matrix A, an n-vector g, a constant f, n-vectors
%  x_l <= x_u and m-vectors c_l <= c_u, find a local minimizer of the
%  LINEAR PROGRAMMING problem
%    minimize g' * x + f
%    subject to c_l <= A * x <= c_u and x_l <= x <= x_u
%  Advantage is taken of sparse A.
%
%  Simple usage -
%
%  to solve the linear program
%   [ x, inform, aux ]
%     = galahad_lpb( g, f, A, c_l, c_u, x_l, x_u, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_lpb( 'initial' )
%
%  to solve the linear program using existing data structures
%   [ x, inform, aux ]
%     = galahad_lpb( 'existing', g, f, A, c_l, c_u, x_l, x_u, control )
%
%  to remove data structures after solution
%   galahad_lpb( 'final' )
%
%  Usual Input -
%    g: the n-vector g
%    f: the scalar f
%    A: the m by n matrix A
%    c_l: the m-vector c_l. The value -inf should be used for infinite bounds
%    c_u: the m-vector c_u. The value inf should be used for infinite bounds
%    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
%    x_u: the n-vector x_u. The value inf should be used for infinite bounds
%
%  Optional Input -
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type LPB_CONTROL as described in the
%      manual for the fortran 90 package GALAHARS_LPB.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/lpb.pdf
%
%  Usual Output -
%   x: a local minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type LPB_INFORM as described in the manual for
%      the fortran 90 package GALAHARS_LPB.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/lpb.pdf
%  aux: a structure containing Lagrange multipliers and constraint status
%   aux.c: values of the constraints A * x
%   aux.y: Lagrange multipliers corresponding to the general constraints
%        c_l <= A * x <= c_u
%   aux.z: dual variables corresponding to the bound constraints
%        x_l <= x <= x_u
%   aux.c_stat: vector indicating the status of the general constraints
%           c_stat(i) < 0 if (c_l)_i = (A * x)_i
%           c_stat(i) = 0 if (c_i)_i < (A * x)_i < (c_u)_i
%           c_stat(i) > 0 if (c_u)_i = (A * x)_i
%   aux.b_stat: vector indicating the status of the bound constraints
%           b_stat(i) < 0 if (x_l)_i = (x)_i
%           b_stat(i) = 0 if (x_i)_i < (x)_i < (x_u)_i
%           b_stat(i) > 0 if (x_u)_i = (x)_i
%
% This version copyright Nick Gould for GALAHAD productions 1/January/2010
