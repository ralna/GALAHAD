% GALAHAD_QPA -
%
%  Given a symmetric n by n matrix H, an m by n matrix A, an n-vector g, 
%  constants f, rho_g & rho_b, n-vectors x_l <= x_u and m-vectors c_l <= c_u, 
%  find a local minimizer of the L1-QUADRATIC PROGRAMMING problem
%    minimize 0.5 * x' * H * x + g' * x + f
%             + rho_g min( A x - c_l , 0 ) + rho_g max( A x - c_u , 0 )
%             + rho_b min(  x  - x_l , 0 ) + rho_b max( x  - x_u , 0 )  
%  or the QUADRATIC PROGRAMMING problem
%    minimize 0.5 * x' * H * x + g' * x + f
%    subject to c_l <= A * x <= c_u and x_l <= x <= x_u
%  using an active-set method.
%  H need not be definite. Advantage is taken of sparse A and H. 
%
%  Simple usage -
%
%  to solve the l1 quadratic program
%   [ x, inform, aux ] 
%     = galahad_qpa( H, g, f, A, c_l, c_u, x_l, x_u, rho_g, rho_b, control )
%
%  to solve the quadratic program
%   [ x, inform, aux ] 
%     = galahad_qpa( H, g, f, A, c_l, c_u, x_l, x_u, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_qpa( 'initial' )
%
%  to solve the l1 quadratic program using existing data structures
%   [ x, inform, aux ]
%     = galahad_qpa( 'existing', H, g, f, A, c_l, c_u, x_l, x_u, 
%                     rho_g, rho_b, control )
%
%  to solve the quadratic program using existing data structures
%   [ x, inform, aux ]
%     = galahad_qpa( 'existing', H, g, f, A, c_l, c_u, x_l, x_u, control )
%
%  to remove data structures after solution
%   galahad_qpa( 'final' )
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
%    rho_g: the scalar rho_g. If one of rho_g and rho_g is given, they
%    rho_b: the scalar rho_b                              must both be
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type QPA_CONTROL as described in the 
%      manual for the fortran 90 package GALAHAD_QPA.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/qpa.pdf
%
%  Usual Output -
%   x: a local minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type QPA_INFORM as described in the manual for 
%      the fortran 90 package GALAHAD_QPA.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/qpa.pdf
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
% This version copyright Nick Gould for GALAHAD productions 17/Feb/2010
