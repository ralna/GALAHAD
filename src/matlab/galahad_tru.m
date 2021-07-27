% GALAHAD_TRU -
%
%  find a local (unconstrained) minimizer of a differentiable objective
%  function f(x) of n real variables x using a trust-region method.
%  Advantage may be taken of sparsity in the Hessian of f(x)
%
%  Simple usage -
%
%  to find the minimizer
%   [ x, inform ]
%    = galahad_tru( x0, eval_f, eval_g, eval_h, pattern_h, control )
%   [ x, inform ]
%    = galahad_tru( x0, eval_f, eval_g, eval_h, control )
%   [ x, inform ]
%    = galahad_tru( x0, eval_f, eval_g, eval_h, pattern_h )
%   [ x, inform ]
%    = galahad_tru( x0, eval_f, eval_g, eval_h )
%
%  Sophisticated usage -
%
%  to initialize data and control structures prior to solution
%   [ control ]
%    = galahad_tru( 'initial' )
%
%  to solve the problem using existing data structures
%   [ x, inform ]
%    = galahad_tru( 'existing', x0, eval_f, eval_g, eval_h, pattern_h, control )
%   [ x, inform ]
%    = galahad_tru( 'existing', x0, eval_f, eval_g, eval_h, control )
%   [ x, inform ]
%    = galahad_tru( 'existing', x0, eval_f, eval_g, eval_h, pattern_h )
%   [ x, inform ]
%    = galahad_tru( 'existing', x0, eval_f, eval_g, eval_h )
%
%  to remove data structures after solution
%   galahad_tru( 'final' )
%
%  Usual Input -
%          x0: an initial estimate of the minimizer
%      eval_f: a user-provided subroutine named eval_f.m for which
%               [f,status] = eval_f(x)
%              returns the value of objective function f at x.
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%      eval_g: a user-provided subroutine named eval_g.m for which
%                [g,status] = eval_g(x)
%              returns the vector of gradients of objective function
%              f at x; g(i) contains the derivative df/dx_i at x.
%              status should be set to 0 if the evaluation succeeds,
%              and a non-zero value if the evaluation fails.
%      eval_h: a user-provided subroutine named eval_h.m for which
%                [h_val,status] = eval_h(x)
%              returns a vector of values of the Hessian of objective
%              function f at x (if required). If H is dense, the
%              i*(i-1)/2+j-th conponent of h_val should contain the
%              derivative d^2f/dx_i dx_j at x, 1<=j<=i<=n. If H is sparse,
%              the k-th component of h_val contains the derivative
%              d^2f/dx_i dx_j for which i=pattern_h(k,1) and j=pattern_h(k,2),
%              see below. status should be set to 0 if the evaluation
%              succeeds, and a non-zero value if the evaluation fails.
%
%  Optional Input -
%   pattern_h: an integer matrix of size (nz,2) for which
%              pattern_h(k,1) and pattern_h(k,2) give the row and
%              column indices of the entries in the *lower-triangular*
%              part (i.e., pattern_h(k,1) >= pattern_h(k,2)) of
%              the Hessian for k=1:nz. This allows users to specify the
%              Hessian as a sparse matrix. If pattern_h is not present,
%              the matrix will be presumed to be dense.
%     control: a structure containing control parameters.
%              The components are of the form control.value, where
%              value is the name of the corresponding component of
%              the derived type TRU_control_type as described in
%              the manual for the fortran 90 package GALAHAD_TRU.
%              See: http://galahad.rl.ac.uk/galahad-www/doc/tru.pdf
%
%  Usual Output -
%          x: a first-order criticl point that is usually a local minimizer.
%
%  Optional Output -
%    control: see above. Returned values are the defaults
%     inform: a structure containing information parameters
%             The components are of the form inform.value, where
%             value is the name of the corresponding component of the
%             derived type TRU_inform_type as described in the manual
%             for the fortran 90 package GALAHAD_TRU. The components
%             of inform.time, inform.PSLS_inform, inform.GLTR_inform,
%             inform.TRS_inform, inform.LMS_inform and
%             inform.SHA_inform are themselves structures, holding the
%             components of the derived types TRU_time_type,
%             PSLS_inform_type, GLTR_inform_type, TRS_inform_type,
%             LMS_inform_type, and SHA_inform_type, respectively.
%             See: http://galahad.rl.ac.uk/galahad-www/doc/tru.pdf
%
% This version copyright Nick Gould for GALAHAD productions 2/March/2017
