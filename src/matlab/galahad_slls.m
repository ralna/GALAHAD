% GALAHAD_SLLS -
%
%  Given an o by n matrix Ao, an o-vector b, and a constant sigma >= 0, find
%  a local mimimizer of the SIMPLEX_CONSTRAINED LINER LEAST-SQUARES problem
%    minimize 0.5 || Ao x - b ||^2 + 0.5 sigma ||x||^2
%    subject to sum x_i = 1, x_i >= 0
%  using a projection method.
%  Advantage is taken of sparse Ao.
%
%  Simple usage -
%
%  to solve the simplex_constrained liner least-squares problem
%   [ x, inform, aux ]
%     = galahad_slls( Ao, b, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_slls( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform, aux ]
%     = galahad_slls( 'existing', Ao, b, control )
%
%  to remove data structures after solution
%   galahad_slls( 'final' )
%
%  Usual Input -
%    A: the o by n matrix A
%    b: the o-vector b
%
%  Optional Input -
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type SLLS_CONTROL as described in the
%      manual for the fortran 90 package GALAHAD_SLLS.
%      In particular if the weight sigma is nonzero, it
%      should be passed via control.weight.
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
%    aux.z: dual variables corresponding to the non-negativity constraints
%         x_i >= 0
%    aux.x_stat: vector indicating the status of the bound constraints
%            x_stat(i) < 0 if (x)_i = 0
%            x_stat(i) = 0 if (x)_i > 0
%
% This version copyright Nick Gould for GALAHAD productions
% 30/December/2023
