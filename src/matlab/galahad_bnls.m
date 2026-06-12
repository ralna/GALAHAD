% GALAHAD_BNLS
%
%  find a local (unconstrained) minimizer of a differentiable, possibly
%  weighted, nonlinear least-squares objective function
%    f(x) = 1/2 sum_i=1^m_r w_i r_i^2(x)
%  where the n real variables x are required to satisfy the simple 
%  bound constraints
%    x_l <= x <= x_u,
%  using a regularization method. Advantage may be taken of sparsity 
%  in the problem
%
%  Terminology -
%
%  r_i(x) is the ith residual, and the vector r(x) are the residuals.
%  Weights w_i>0 may be provided, but otherwise will be assumed to be 1.
%  The matrix Jr(x) for which Jr_i,j = d r_i(x) / dx_j is the Jacobian
%  of the residuals
%
%  Simple usage -
%
%  to find the minimizer
%   [ x, inform ]
%    = galahad_bnls( pattern, x_l, x_u, x_0, eval_r, eval_jr, w, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%    = galahad_bnls( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform ] = galahad_bnls( 'existing', pattern, x_l, x_u, x_0, ...
%                                  eval_r, eval_jr, w, control )
%
%  to remove data structures after solution
%   galahad_bnls( 'final' )
%
%  Usual Input -
%     pattern: a structure that indicates the sparsity pattern of
%              the Jacobian matrix. Components are -
%                 m_r: number of residuals (compulsory)
%                 jr_row, jr_col: row and column indices of the nonzeros
%                 in the Jacobian of the residuals Jr(x) (optional).
%                 If absent, Jr(x) is assumed dense and stored by rows
%      x_l: the n-vector x_l. The value -inf should be used for infinite bounds
%      x_u: the n-vector x_u. The value inf should be used for infinite bounds
%      x_0: an initial estimate of the minimizer
%      eval_r: a user-provided subroutine named eval_r.m for which
%                [r,status] = eval_r(x)
%              returns the value of the vector of residual functions
%              r at x; r(i) contains r_i(x).
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%      eval_jr: a user-provided subroutine named eval_jr.m for which
%                [jr_val,status] = eval_jr(x)
%              returns a vector of values of the Jacobian Jr(x) of the
%              residuals stored by rows. If Jr(x) is dense, the n*(i-1)+j-th
%              conponent of jr_val should contain the derivative
%              dr_i(x)/dx_j dx_j at x, 1<=i<=m, 1<=j<=n. If Jr(x) is sparse,
%              the k-th component of jr_val contains the derivative
%              dr_i(x)/dx_i dx_j for which i=jr_row(k) and j=jr_col(k),
%              as set in the structure pattern (see above).
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%
%  Optional Input -
%     w: the m_r-vector of weights w for which W=diag(w). If absent, 
%        weights of one will be used.
%     control: a structure containing control parameters.
%              The components are of the form control.value, where
%              value is the name of the corresponding component of
%              the derived type BNLS_control_type as described in
%              the manual for the fortran 90 package GALAHAD_BNLS.
%              See: http://galahad.rl.ac.uk/galahad-www/doc/bnls.pdf
%
%  Usual Output -
%          x: a first-order criticl point that is usually a local minimizer.
%
%  Optional Output -
%     control: see 'initial' above. Returned values are the defaults
%     inform: a structure containing information parameters
%             The components are of the form inform.value, where
%             value is the name of the corresponding component of the
%             derived type BNLS_inform_type as described in the manual
%             for the fortran 90 package GALAHAD_BNLS. The components
%             of inform.time, inform.BLLS_inform and inform.BLLSB_inform
%             are themselves structures, holding the components of the 
%             derived types BNLS_time_type, BLLS_inform_type and
%             BLLSB_inform_type, respectively.
%             See: http://galahad.rl.ac.uk/galahad-www/doc/bnls.pdf
%     aux: a structure containing Lagrange multipliers and constraint status
%      aux.z: dual variables z corresponding to the simple-bound constraints
%           x_l <= x <= x_u
%      aux.r: values of the residuals r(x)
%      aux.g: values of the gradients g(x) = Jr(x)^T W r(x)
%      aux.x_status: vector indicating the status of the bound constraints
%              x_status(i) < 0 if (x)_i = (x_l)_i
%              x_status(i) = 0 if (x)_i > 0
%              x_status(i) > 0 if (x)_i = (x_u)_i
%            x_status(i) = 0 if (x)_i > 0
%
% This version copyright Nick Gould for GALAHAD productions 21/May/2026
