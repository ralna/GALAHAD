% GALAHAD_QPC -
%
%  Given a symmetric n by n matrix H, an m by n matrix A, an n-vector 
%  g, a constant f, n-vectors x_l <= x_u and m-vectors c_l <= c_u, 
%  find a local minimizer of the QUADRATIC PROGRAMMING problem
%    minimize 0.5 * x' * H * x + g' * x + f
%    subject to c_l <= A * x <= c_u and x_l <= x <= x_u
%  using a crossover interior-point/active-set method.
%  H need not be definite. Advantage is taken of sparse A and H. 
%
%  Simple usage -
%
%  to solve the quadratic program
%   [ x, inform, aux ] 
%     = galahad_qpc( H, g, f, A, c_l, c_u, x_l, x_u, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_qpc( 'initial' )
%
%  to solve the quadratic program using existing data structures
%   [ x, inform, aux ]
%     = galahad_qpc( 'existing', H, g, f, A, c_l, c_u, x_l, x_u, control )
%
%  to remove data structures after solution
%   galahad_qpc( 'final' )
%
%  Usual Input -
%    H: the symmetric n by n matrix H
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
%      the derived type QPC_CONTROL as described in the 
%      manual for the fortran 90 package GALAHARS_QPC.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/qpc.pdf
%
%  Usual Output -
%   x: a local minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type QPC_INFORM as described in the manual for 
%      the fortran 90 package GALAHARS_QPC.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/qpc.pdf
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
% This version copyright Nick Gould for GALAHAD productions 26/July/2007
