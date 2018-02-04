% GALAHAD_LLST -
%
%  Given an m by n matrix A, an m-vector b, a scalar radius, and possibly
%  a symmetric, diagonally dominant n by n matrix S, find the minimum
%  S-norm solution of the LEAST-SQUARES TRUST-REGION subproblem
%    minimize || A x - b ||_2
%    subject to ||x||_S <= radius
%  Here ||x||_S^2 = x' * S * x; if S is not given, S=I and ||x||_S is
%  thus taken to be the Euclidean (l_2-)norm sqrt(x' * x). 
%  Advantage is taken of sparse A and S.
%
%  Simple usage -
%
%  to solve the least-squares trust-region subproblem in the Euclidean norm
%   [ x, obj, inform ] 
%     = galahad_llst( A, b, radius, control, S )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_llst( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, obj, inform ]
%     = galahad_llst( 'existing', A, b, radius, control, S )
%
%  to remove data structures after solution
%   galahad_llst( 'final' )
%
%  Usual Input -
%          A: the m by n matrix A
%          b: the m-vector b
%     radius: the trust-region radius (radius>0)
%
%  Optional Input -
%    control: a structure containing control parameters.
%            The components are of the form control.value, where
%            value is the name of the corresponding component of
%            the derived type LLST_CONTROL as described in the 
%            manual for the fortran 90 package GALAHAD_LLST.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/llst.pdf
%          S: the n by n symmetric, diagonally-dominant matrix S
%
%  Usual Output -
%          x: the minimizer of least S-norm
%        obj: the optimal value of the objective function ||Ax-b||_2
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%           The components are of the form inform.value, where
%           value is the name of the corresponding component of the
%           derived type LLST_INFORM as described in the manual for 
%           the fortran 90 package GALAHAD_LLST. 
%           See: http://galahad.rl.ac.uk/galahad-www/doc/llst.pdf
%           Note that as the objective value is already available
%           the component r_norm from LLST_inform is omitted.
%
% This version copyright Nick Gould for GALAHAD productions 1/March/2014
