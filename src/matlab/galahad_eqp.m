% GALAHAD_EQP -
%
%  Given a symmetric n by n matrix H, an m by n matrix A, an n-vector 
%  g, a constant f and an m-vector c, find the minimizer of the 
%  EQUALITY-CONSTRAINED QUADRATIC PROGRAMMING problem
%    minimize 0.5 * x' * H * x + g' * x + f
%    subject to  A * x + c = 0.
%  An additional trust-region constraint may be imposed to prevent unbounded
%  solutions. H need not be definite. Advantage is taken of sparse A and H. 
%
%  Simple usage -
%
%  to solve the quadratic program
%   [ x, inform, aux ] 
%     = galahad_eqp( H, g, f, A, c, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_eqp( 'initial' )
%
%  to solve the quadratic program using existing data structures
%   [ x, inform, aux ]
%     = galahad_eqp( 'existing', H, g, f, A, c, control )
%
%  to remove data structures after solution
%   galahad_eqp( 'final' )
%
%  Usual Input -
%    H: the symmetric n by n matrix H
%    g: the n-vector g
%    f: the scalar f
%    A: the m by n matrix A
%    c: the m-vector c
%
%  Optional Input -
%    control, a structure containing control parameters.
%      The components are of the form control.value, where
%      value is the name of the corresponding component of
%      the derived type EQP_CONTROL as described in the 
%      manual for the fortran 90 package GALAHAD_EQP.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/eqp.pdf
%
%  Usual Output -
%   x: a local minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type EQP_INFORM as described in the manual for 
%      the fortran 90 package GALAHAD_EQP.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/eqp.pdf
%  aux: a structure containing Lagrange multipliers and constraint status
%   aux.y: Lagrange multipliers corresponding to the constraints A x + c = 0
%
% This version copyright Nick Gould for GALAHAD productions 18/February/2010
