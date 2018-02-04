% GALAHAD_GLRT -
%
%  Given a symmetric n by n matrix H (and possibly M), an n-vector g, 
%  a constant f, and scalars p and sigma, find an approximate solution 
%  of the REGULARISED quadratic subproblem
%    minimize  1/p sigma ||x||_M^p + 1/2 <x, H x> + <c, x> + f
%  using an iterative method.
%  Here ||x||_M^2 = x' * M * x and M is positive definite; if M is
%  not given, M=I and ||x||_M is thus taken to be the Euclidean (l_2-)norm 
%  sqrt(x' * x). H need not be definite. Advantage is taken of sparse H. 
%
%  Simple usage -
%
%  to solve the regularized quadratic subproblem
%   [ x, obj, inform ] 
%     = galahad_glrt( H, c, f, p, sigma, control, M )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_glrt( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, obj, inform ]
%     = galahad_glrt( 'existing', H, c, f, p, sigma, control, M )
%
%  to remove data structures after solution
%   galahad_glrt( 'final' )
%
%  Usual Input -
%          H: the symmetric n by n matrix H
%          c: the n-vector c
%          f: the scalar f
%          p: the regularisation order, p (p>=2)
%     sigma: the regularisation weight, sigma (sigma>=0)
%
%  Optional Input -
%    control: a structure containing control parameters.
%            The components are of the form control.value, where
%            value is the name of the corresponding component of
%            the derived type GLRT_control as described in the 
%            manual for the fortran 90 package GALAHAD_GLRT.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/glrt.pdf
%          M: the n by n symmetric, positive-definite matrix M
%
%  Usual Output -
%          x: the global minimizer
%        obj: the optimal value of the objective function
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of
%      the derived type GLRT_inform as described in the manual for 
%      the fortran 90 package GALAHAD_GLRT. 
%      See: http://galahad.rl.ac.uk/galahad-www/doc/glrt.pdf
%      Note that as the objective value is already available
%      the component obj from GLRT_inform is omitted.
%
% This version copyright Nick Gould for GALAHAD productions 5/March/2009
