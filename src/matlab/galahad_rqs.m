% GALAHAD_RQS -
%
%  Given a symmetric n by n matrix H (and possibly M), optionally an
%  m by n matrix A, an n-vector g, a constant f, and scalars p and sigma,
%  find the solution of the REGULARISED QUADRATIC subproblem
%    minimize sigma/p ||x||_M^p + 0.5 * x' * H * x + c' * x + f
%    (perhaps subject to Ax=0).
%  Here ||x||_M^2 = x' * M * x and M is diagonally dominant; if M is
%  not given, M=I and ||x||_M is thus taken to be the Euclidean (l_2-)norm
%  sqrt(x' * x). H need not be definite. Advantage is taken of sparse A and H.
%
%  Simple usage -
%
%  to solve the regularised quadratic subproblem in the Euclidean norm
%   [ x, inform ]
%     = galahad_rqs( H, c, f, p, sigma, control, M, A )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%     = galahad_rqs( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform ]
%     = galahad_rqs( 'existing', H, c, f, p, sigma, control, M, A )
%
%  to remove data structures after solution
%   galahad_rqs( 'final' )
%
%  Usual Input -
%          H: the symmetric n by n matrix H
%          c: the n-vector c
%          f: the scalar f
%          p: the order of the regularisation (p>2)
%      sigma: the regulaisation weight (sigma>0)
%
%  Optional Input -
%    control: a structure containing control parameters.
%            The components are of the form control.value, where
%            value is the name of the corresponding component of
%            the derived type RQS_control_type as described in
%            the manual for the fortran 90 package GALAHAD_RQS.
%            See: http://galahad.rl.ac.uk/galahad-www/doc/rqs.pdf
%          M: the n by n symmetric, diagonally-dominant matrix M
%          A: the m by n matrix A
%
%  Usual Output -
%          x: the global minimizer
%
%  Optional Output -
%   control: see above. Returned values are the defaults
%   inform: a structure containing information parameters
%      The components are of the form inform.value, where
%      value is the name of the corresponding component of the
%      derived type RQS_inform_type as described in the manual
%      for the fortran 90 package GALAHAD_RQS. The components
%      of inform.time, inform.history, inform.IR_inform and
%      inform.SLS_inform are themselves structures, holding the
%      components of the derived types RQS_time_type, RQS_history_type,
%      IR_inform_type and SLS_inform_type, respectively.
%      See: http://galahad.rl.ac.uk/galahad-www/doc/rqs.pdf
%
% This version copyright Nick Gould for GALAHAD productions 18/February/2009
