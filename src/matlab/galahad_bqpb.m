% GALAHAD_BQPB -
%
%  Given a symmetric n by n matrix H, an m by n matrix A, an n-vector 
%  g, a constant f, and n-vectors x_l <= x_u, find a local mimimizer
%  of the BOUND_CONSTRAINED QUADRATIC PROGRAMMING problem
%    minimize 0.5 * x' * H * x + g' * x + f
%    subject to x_l <= x <= x_u
%  using an interior-point method.
%  H need must be positive semi-definite. Advantage is taken of sparse H. 
%
%  Simple usage -
%
%  to solve the bound-constrained quadratic program
%   [ x, inform, aux ] 
%     = galahad_bqpb( H, g, f, x_l, x_u, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_bqpb( 'initial' )
%
%  to solve the bound-constrained QP using existing data structures
%   [ x, inform, aux ]
%     = galahad_bqpb( 'existing', H, g, f, x_l, x_u, control )
%
%  to remove data structures after solution
%   galahad_bqpb( 'final' )
%
%  Usual Input -
%    H: the symmetric n by n matrix H
%    g: the n-vector g
%    f: the scalar f
%    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
%    x_u: the n-vector x_u. The value inf should be used for infinite bounds
%
%  Optional Input -
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type BQPB_CONTROL as described in the 
%      manual for the fortran 90 package GALAHAD_BQPB.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/bqpb.pdf
%
%  Usual Output -
%   x: a local minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type BQPB_INFORM as described in the manual for 
%      the fortran 90 package GALAHAD_BQPB.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/bqpb.pdf
%   aux: a structure containing Lagrange multipliers and constraint status
%    aux.z: dual variables corresponding to the bound constraints
%         x_l <= x <= x_u
%    aux.b_stat: vector indicating the status of the bound constraints
%            b_stat(i) < 0 if (x_l)_i = (x)_i
%            b_stat(i) = 0 if (x_i)_i < (x)_i < (x_u)_i 
%            b_stat(i) > 0 if (x_u)_i = (x)_i
%
% This version copyright Nick Gould for GALAHAD productions
% 4/December/2009

