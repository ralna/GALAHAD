% GALAHAD_SNLS
%
%  find a local minimizer of a differentiable, possibly
%  weighted, nonlinear least-squares objective function
%    f(x) = 1/2 sum_i=1^m_r w_i r_i^2(x)
%  where the n real variables x are required to lie within
%  the interection multiple non-overlapping regular simplices
%    sum_{C_j} x_i = 1, x_{C_j} >= 0 for j = 1,...,m_c
%  using a regularization method. Advantage may be taken of sparsity 
%  in the problem
%
%  Terminology -
%
%  r_i(x) is the ith residual, and the vector r(x) are the residuals.
%  Weights w_i>0 may be provided, but otherwise will be assumed to be 1.
%  The matrix Jr(x) for which J_i,j = d r_i(x) / dx_j is the Jacobian
%  of the residuals
%
%  Simple usage -
%
%  to find the minimizer
%   [ x, inform ]
%    = galahad_snls( pattern, x_0, eval_r, eval_jr, cohort, w, control )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%    = galahad_snls( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform ]
%    = galahad_snls( 'existing', pattern, x_0, eval_r, ...
                     cohort, w, eval_jr, control )
%
%  to remove data structures after solution
%   galahad_snls( 'final' )
%
%  Usual Input -
%     pattern: a structure that indicates the sparsity pattern of
%              the Jacobian matrices. Components are -
%                m_r: number of residuals (compulsory)
%                jr_row, jr_col: row and column indices of the nonzeros
%                  in the Jacobian of the residuals Jr(x) (optional).
%                  If absent, Jr(x) is assumed dense and stored by rows
%      x_0: an initial estimate of the minimizer
%      eval_r: a user-provided subroutine named eval_r.m for which
%                [r,status] = eval_r(x)
%              returns the value of the vector of residual functions
%              r at x; r(i) contains r_i(x).
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%      eval_jr: a user-provided subroutine named eval_jr.m for which
%                [j_val,status] = eval_jr(x)
%              returns a vector of values of the Jacobian Jr(x) of the
%              residuals stored by rows. If Jr(x) is dense, the n*(i-1)+j-th
%              conponent of j_val should contain the derivative
%              dr_i(x)/dx_j dx_j at x, 1<=i<=m, 1<=j<=n. If Jr(x) is sparse,
%              the k-th component of j_val contains the derivative
%              dr_i(x)/dx_i dx_j for which i=jr_row(k) and j=jr_col(k),
%              as set in the structure pattern (see above).
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%
%  Optional Input -
%     cohort: the n-vector of cohorts, so that variable x_i is in cohort C_j
%             if cohort[i] = j, and x_i is not constrained if cohort[i] = 0.
%             If absent, a single cohort containing all variables will be used
%     w: the m_c-vector of weights w for which W=diag(w). If absent, 
%        weights of one will be used. ** N.B. If n=m_r and both w and cohort 
%        are provided, cohort must proceed w in the calling sequence
%     control: a structure containing control parameters.
%              The components are of the form control.value, where
%              value is the name of the corresponding component of
%              the derived type SNLS_control_type as described in
%              the manual for the fortran 90 package GALAHAD_SNLS.
%              See: http://galahad.rl.ac.uk/galahad-www/doc/snls.pdf
%
%  Usual Output -
%          x: a first-order criticl point that is usually a local minimizer.
%
%  Optional Output -
%     control: see 'initial' above. Returned values are the defaults
%     inform: a structure containing information parameters
%             The components are of the form inform.value, where
%             value is the name of the corresponding component of the
%             derived type SNLS_inform_type as described in the manual
%             for the fortran 90 package GALAHAD_SNLS. The components
%             of inform.time, inform.SLLS_inform and inform.SLLSB_inform
%             are themselves structures, holding the components of the 
%             derived types SNLS_time_type, SLLS_inform_type and
%             SLLSB_inform_type, respectively.
%             See: http://galahad.rl.ac.uk/galahad-www/doc/snls.pdf
%   aux: a structure containing Lagrange multipliers and constraint status
%    aux.y: Largrange multipliers y corresponding to the simplex constraints
%    aux.z: dual variables z corresponding to the non-negativity constraints
%         x_i >= 0
%    aux.r: values of the residuals r(x) = Ao x - b
%    aux.g: values of the gradients g(x) = Ao^T W r(x)
%    aux.x_status: vector indicating the status of the bound constraints
%            x_status(i) < 0 if (x)_i = 0
%            x_status(i) = 0 if (x)_i > 0
%
% This version copyright Nick Gould for GALAHAD productions 26/March/2026
