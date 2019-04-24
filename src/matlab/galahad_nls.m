% GALAHAD_NLS
%
%  find a local (unconstrained) minimizer of a differentiable, possibly
%  weighted, nonlinear least-squares objective function
%    f(x) = 1/2 sum_i=1^m w_i r_i^2(x)
%  of n real variables x, using a regularized tensor-Newton method.
%  Advantage may be taken of sparsity in the problem.
%
%  Terminology -
%
%  r_i(x) is the ith residual, and the vector r(x) are the residuals.
%  Weights w_i>0 may be provided, but otherwise will be assumed to be 1.
%  The matrix J(x) for which J_i,j = d r_i(x) / dx_j is the Jacobian
%  of the residuals. For a specified m-vector y, the weighted residual
%  Hessian H(x,y) = sum_i=1^m y_i H_i(x), where (H_i(x))_j,k =
%  d^2 r_i(x) / dx_j dx_k is the Hessian of the ith residual. Finally,
%  for a given n-vector v, the residual-Hessians-vector product matrix
%  P(x,v) = (H_1(x) v, ...., H_m(x)v)
%
%  Simple usage -
%
%  to find the minimizer
%   [ x, inform ]
%    = galahad_nls( pattern, x0, eval_r, eval_j, eval_h, eval_p, control )
%   [ x, inform ]
%    = galahad_nls( pattern, x0, eval_r, eval_j, eval_h, control )
%   [ x, inform ]
%    = galahad_nls( pattern, x0, eval_r, eval_j, eval_h )
%   [ x, inform ]
%    = galahad_nls( pattern, x0, eval_r, eval_j, control )
%   [ x, inform ]
%    = galahad_nls( pattern, x0, eval_r, eval_j )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%    = galahad_nls( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform ]
%    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j, eval_h, ...
%                   eval_p, control )
%   [ x, inform ]
%    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j, eval_h, control )
%   [ x, inform ]
%    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j, eval_h )
%   [ x, inform ]
%    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j, control )
%   [ x, inform ]
%    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j )
%
%  to remove data structures after solution
%   galahad_nls( 'final' )
%
%  Usual Input -
%     pattern: a structure that indicates the spartsity patterns of
%              the Jacobian, Hessian and residual-Hessian product
%              matrices, if any. Components are -
%                m: number of residuals (compulsory)
%                w: a vector of m positive weights (optional). If
%                  absent, weights of one will be used.
%                j_row, j_col: row and column indices of the nonzeros
%                  in the Jacobian of the residuals J(x) (optional).
%                  If absent, J(x) is assumed dense and stored by rows
%                h_row, h_col: row and column indices of the *lower-
%                  -triangular* part of the weighted residual Hessian
%                  H(x,y) (optional). If absent, H(x,y) is assumed dense
%                  and its lower triangle is stored by rows
%                p_row, p_col: row and column indices of the residual-
%                  -Hessians-vector product matrix P(x,v) stored by
%                  columns (i.e., the column indices are in non-decreasing
%                  order(optional). If absent, P(x,y) is assumed dense and
%                  is stored by columns
%          x0: an initial estimate of the minimizer
%      eval_r: a user-provided subroutine named eval_r.m for which
%                [r,status] = eval_r(x)
%              returns the value of the vector of residual functions
%              r at x; r(i) contains r_i(x).
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%      eval_j: a user-provided subroutine named eval_j.m for which
%                [j_val,status] = eval_j(x)
%              returns a vector of values of the Jacobian J(x) of the
%              residuals stored by rows. If J(x) is dense, the n*(i-1)+j-th
%              conponent of j_val should contain the derivative
%              dr_i(x)/dx_j dx_j at x, 1<=i<=m, 1<=j<=n. If J(x) is sparse,
%              the k-th component of j_val contains the derivative
%              dr_i(x)/dx_i dx_j for which i=j_row(k) and j=j_col(k),
%              as set in the structure pattern (see above).
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%
%  Optional Input -
%      eval_h: a user-provided subroutine named eval_h.m for which
%                [h_val,status] = eval_h(x,y)
%              returns a vector of values of the weighted residual
%              Hessian H(x,y) at (x,y) (if required) stored by rows.
%              If H(x,y) is dense, the i*(i-1)/2+j-th conponent of h_val
%              should contain the (H(x,y))_i,j at (x,y), 1<=j<=i<=n.
%              If H(x,y) is sparse, the k-th component of h_val contains
%              the component (H(x,y))_i,j for which i=h_row(k) and
%              j=h_col(k), as set in the structure pattern (see above).
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%              If eval_h is absent, the solver will resort to a
%              Gauss-Newton model
%      eval_p: a user-provided subroutine named eval_p.m for which
%                [p_val,status] = eval_p(x,v)
%              returns a vector of values of the residual-Hessians-vector
%              product matrix P(x,v) at (x,v) (if required) stored by
%              columns. If P(x,v) is dense, the i+m*(j-1)-th conponent
%              of p_val should contain (P(x,v))_i,j at (x,v), 1<=i<=n,
%              1<=j<=m. If P(x,v) is sparse, the k-th component of h_val
%              contains the component (P(x,v))_i,j for which i=p_row(k)
%              and j=p_col(k), as set in the structure pattern (see above).
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%              If eval_p is absent, the solver will resort to a
%              Newton or Gauss-Newton model
%     control: a structure containing control parameters.
%              The components are of the form control.value, where
%              value is the name of the corresponding component of
%              the derived type NLS_control_type as described in
%              the manual for the fortran 90 package GALAHAD_NLS.
%              See: http://galahad.rl.ac.uk/galahad-www/doc/nls.pdf
%
%  Usual Output -
%          x: a first-order criticl point that is usually a local minimizer.
%
%  Optional Output -
%     inform: a structure containing information parameters
%             The components are of the form inform.value, where
%             value is the name of the corresponding component of the
%             derived type NLS_inform_type as described in the manual
%             for the fortran 90 package GALAHAD_NLS. The components
%             of inform.time, inform.PSLS_inform, inform.GLRT_inform,
%             inform.RQS_inform, inform.BSC_inform and
%             inform.ROOTS_inform are themselves structures, holding the
%             components of the derived types NLS_time_type,
%             PSLS_inform_type, GLRT_inform_type, RQS_inform_type,
%             BSC_inform_type, and ROOTS_inform_type, respectively.
%             See: http://galahad.rl.ac.uk/galahad-www/doc/nls.pdf
%
% This version copyright Nick Gould for GALAHAD productions 7/March/2019
