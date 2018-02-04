% GALAHAD_QPC -
%
%  Given an m by n matrix A, n-vectors g, x_l <= x_u and m-vectors c_l <= c_u, 
%  find a well-centered interior point within the polytope
%          c_l <= A x <= c_u and x_l <=  x <= x_u,
%  for which the dual feasibility conditions
%           g = A' y + z
%  for Lagrange multipliers y and z are satisfied, using an infeasible-point 
%  primal-dual method. Advantage is taken of sparse A.
%
%  Simple usage -
%
%  to find a well-centered feasible point
%   [ x, inform, aux ] 
%     = galahad_wcp( A, c_l, c_u, x_l, x_u, g )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_wcp( 'initial' )
%
%  to find a well-centered feasible point using existing structures
%   [ x, inform, aux ]
%     = galahad_wcp( 'existing, g, A, c_l, c_u, x_l, x_u, g, control )
%
%  to remove data structures after solution
%   galahad_wcp( 'final' )
%
%  Usual Input -
%    A: the m by n matrix A
%    c_l: the m-vector c_l. The value -inf should be used for infinite bounds
%    c_u: the m-vector c_u. The value inf should be used for infinite bounds
%    x_l: the n-vector x_l. The value -inf should be used for infinite bounds
%    x_u: the n-vector x_u. The value inf should be used for infinite bounds
%
%  Optional Input -
%    g: the n-vector g. If absent, g = 0 is presumed
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type WCP_CONTROL as described in the 
%      manual for the fortran 90 package GALAHAD_WCP.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/wcp.pdf
%
%  Usual Output -
%   x: a local minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type WCP_INFORM as described in the manual for 
%      the fortran 90 package GALAHAD_WCP.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/wcp.pdf
%  aux: a structure containing Lagrange multipliers and constraint status
%   aux.c: values of the constraints A * x
%   aux.y: Lagrange multipliers corresponding to the general constraints 
%        c_l <= A * x <= c_u 
%   aux.z: dual variables corresponding to the bound constraints
%        x_l <= x <= x_u
%   aux.c_status: vector indicating the status of the general constraints
%           c_status(i) < 0 if (c_l)_i = (A * x)_i
%           c_status(i) = 0 if (c_i)_i < (A * x)_i < (c_u)_i 
%           c_status(i) > 0 if (c_u)_i = (A * x)_i
%   aux.x_status: vector indicating the status of the bound constraints
%           x_status(i) < 0 if (x_l)_i = (x)_i
%           x_status(i) = 0 if (x_i)_i < (x)_i < (x_u)_i 
%           x_status(i) > 0 if (x_u)_i = (x)_i
% This version copyright Nick Gould for GALAHAD productions 06/November/2008
