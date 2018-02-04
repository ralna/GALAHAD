% GALAHAD_LSRT -
%
%  Given an m by n matrix A, an m-vector b, and scalars p and sigma, find 
%  an approximate solution of the REGULARISED LEAST-SQUARES subproblem
%    minimize 1/2 || A x - b ||^2_2 + 1/p sigma ||x||^p_2 
%  using an iterative method. Here ||.||_2 is the Euclidean (l_2-)norm.
%  Advantage is taken of sparse A. 
%
%  Simple usage -
%
%  to solve the regularised least-squares subproblem
%   [ x, obj, inform ] 
%     = galahad_lsrt( A, b, p, sigma, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ] 
%     = galahad_lsrt( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, obj, inform ]
%     = galahad_lsrt( 'existing', A, b, p, sigma, control )
%
%  to remove data structures after solution
%   galahad_lsrt( 'final' )
%
%  Usual Input -
%          A: the m by n matrix A
%          b: the m-vector b
%          p: the regularisation order, p (p>=2)
%      sigma: the regularisation weight, sigma (sigma>=0)
%
%  Optional Input -
%    control: a structure containing control parameters.
%            The components are of the form control.value, where
%            value is the name of the corresponding component of
%            the derived type LSRT_CONTROL as described in the 
%            manual for the fortran 90 package GALAHAD_LSRT.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/lsrt.pdf
%
%  Usual Output -
%          x: the global minimizer
%        obj: the optimal value of the objective function
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%           The components are of the form inform.value, where
%           value is the name of the corresponding component of the
%           derived type LSRT_INFORM as described in the manual for 
%           the fortran 90 package GALAHAD_LSRT. 
%           See: http://galahad.rl.ac.uk/galahad-www/doc/lsrt.pdf
%           Note that as the objective value is already available
%           the component obj from LSRT_inform is omitted.
%
% This version copyright Nick Gould for GALAHAD productions 5/March/2009
