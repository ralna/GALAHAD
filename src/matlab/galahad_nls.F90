#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.2 - 19/04/2019 AT 14:30 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_NLS
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  find a local (unconstrained) minimizer of a differentiable, possibly
!  weighted, nonlinear least-squares objective function
!    f(x) = 1/2 sum_i=1^m w_i r_i^2(x)
!  of n real variables x, using a regularized tensor-Newton method.
!  Advantage may be taken of sparsity in the problem.
!
!  Terminology -
!
!  r_i(x) is the ith residual, and the vector r(x) are the residuals.
!  Weights w_i>0 may be provided, but otherwise will be assumed to be 1.
!  The matrix J(x) for which J_i,j = d r_i(x) / dx_j is the Jacobian
!  of the residuals. For a specified m-vector y, the weighted residual
!  Hessian H(x,y) = sum_i=1^m y_i H_i(x), where (H_i(x))_j,k =
!  d^2 r_i(x) / dx_j dx_k is the Hessian of the ith residual. Finally,
!  for a given n-vector v, the residual-Hessians-vector product matrix
!  P(x,v) = (H_1(x) v, ...., H_m(x)v)
!
!  Simple usage -
!
!  to find the minimizer
!   [ x, inform ]
!    = galahad_nls( pattern, x0, eval_r, eval_j, eval_h, eval_p, control )
!   [ x, inform ]
!    = galahad_nls( pattern, x0, eval_r, eval_j, eval_h, control )
!   [ x, inform ]
!    = galahad_nls( pattern, x0, eval_r, eval_j, eval_h )
!   [ x, inform ]
!    = galahad_nls( pattern, x0, eval_r, eval_j, control )
!   [ x, inform ]
!    = galahad_nls( pattern, x0, eval_r, eval_j )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!    = galahad_nls( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, inform ]
!    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j, eval_h, ...
!                   eval_p, control )
!   [ x, inform ]
!    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j, eval_h, control )
!   [ x, inform ]
!    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j, eval_h )
!   [ x, inform ]
!    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j, control )
!   [ x, inform ]
!    = galahad_nls( 'existing', pattern, x0, eval_r, eval_j )
!
!  to remove data structures after solution
!   galahad_nls( 'final' )
!
!  Usual Input -
!     pattern: a structure that indicates the spartsity patterns of
!              the Jacobian, Hessian and residual-Hessian product
!              matrices, if any. Components are -
!                m: number of residuals (compulsory)
!                w: a vector of m positive weights (optional). If
!                  absent, weights of one will be used.
!                j_row, j_col: row and column indices of the nonzeros
!                  in the Jacobian of the residuals J(x) (optional).
!                  If absent, J(x) is assumed dense and stored by rows
!                h_row, h_col: row and column indices of the *lower-
!                  -triangular* part of the weighted residual Hessian
!                  H(x,y) (optional). If absent, H(x,y) is assumed dense
!                  and its lower triangle is stored by rows
!                p_row, p_col: row and column indices of the residual-
!                  -Hessians-vector product matrix P(x,v), stored by
!                  columns (i.e., the column indices are in non-decreasing
!                  order(optional). If absent, P(x,y) is assumed dense and
!                  is stored by columns
!          x0: an initial estimate of the minimizer
!      eval_r: a user-provided subroutine named eval_r.m for which
!                [r,status] = eval_r(x)
!              returns the value of the vector of residual functions
!              r at x; r(i) contains r_i(x).
!              status should be set to 0 if the evaluation succeeds,
!              and a non-zero value if the evaluation fails.
!      eval_j: a user-provided subroutine named eval_j.m for which
!                [j_val,status] = eval_j(x)
!              returns a vector of values of the Jacobian J(x) of the
!              residuals stored by rows. If J(x) is dense, the n*(i-1)+j-th
!              conponent of j_val should contain the derivative
!              dr_i(x)/dx_j dx_j at x, 1<=i<=m, 1<=j<=n. If J(x) is sparse,
!              the k-th component of j_val contains the derivative
!              dr_i(x)/dx_i dx_j for which i=j_row(k) and j=j_col(k),
!              as set in the structure pattern (see above).
!              status should be set to 0 if the evaluation succeeds,
!              and a non-zero value if the evaluation fails.
!
!  Optional Input -
!      eval_h: a user-provided subroutine named eval_h.m for which
!                [h_val,status] = eval_h(x,y)
!              returns a vector of values of the weighted residual
!              Hessian H(x,y) at (x,y) (if required) stored by rows.
!              If H(x,y) is dense, the i*(i-1)/2+j-th conponent of h_val
!              should contain the (H(x,y))_i,j at (x,y), 1<=j<=i<=n.
!              If H(x,y) is sparse, the k-th component of h_val contains
!              the component (H(x,y))_i,j for which i=h_row(k) and
!              j=h_col(k), as set in the structure pattern (see above).
!              status should be set to 0 if the evaluation succeeds,
!              and a non-zero value if the evaluation fails.
!              If eval_h is absent, the solver will resort to a
!              Gauss-Newton model
!      eval_p: a user-provided subroutine named eval_p.m for which
!                [p_val,status] = eval_p(x,v)
!              returns a vector of values of the residual-Hessians-vector
!              product matrix P(x,v) at (x,v) (if required) stored by
!              columns. If P(x,v) is dense, the i+m*(j-1)-th conponent
!              of p_val should contain (P(x,v))_i,j at (x,v), 1<=i<=n,
!              1<=j<=m. If P(x,v) is sparse, the k-th component of h_val
!              contains the component (P(x,v))_i,j for which i=p_row(k)
!              and j=p_col(k), as set in the structure pattern (see above).
!              status should be set to 0 if the evaluation succeeds,
!              and a non-zero value if the evaluation fails.
!              If eval_p is absent, the solver will resort to a
!              Newton or Gauss-Newton model
!     control: a structure containing control parameters.
!              The components are of the form control.value, where
!              value is the name of the corresponding component of
!              the derived type NLS_control_type as described in
!              the manual for the fortran 90 package GALAHAD_NLS.
!              See: http://galahad.rl.ac.uk/galahad-www/doc/nls.pdf
!
!  Usual Output -
!          x: a first-order criticl point that is usually a local minimizer.
!
!  Optional Output -
!     inform: a structure containing information parameters
!             The components are of the form inform.value, where
!             value is the name of the corresponding component of the
!             derived type NLS_inform_type as described in the manual
!             for the fortran 90 package GALAHAD_NLS. The components
!             of inform.time, inform.PSLS_inform, inform.GLRT_inform,
!             inform.RQS_inform, inform.BSC_inform and
!             inform.ROOTS_inform are themselves structures, holding the
!             components of the derived types NLS_time_type,
!             PSLS_inform_type, GLRT_inform_type, RQS_inform_type,
!             BSC_inform_type, and ROOTS_inform_type, respectively.
!             See: http://galahad.rl.ac.uk/galahad-www/doc/nls.pdf
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.2. March 6th 2019

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
!     USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_NLS_MATLAB_TYPES
      USE GALAHAD_USERDATA_double
      USE GALAHAD_NLS_double
      IMPLICIT NONE
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

! ------------------------- Do not change -------------------------------

!  Keep the above subroutine, argument, and function declarations for use
!  in all your fortran mex files.
!
      INTEGER * 4 :: nlhs, nrhs
      mwPointer :: plhs( * ), prhs( * )

      INTEGER, PARAMETER :: slen = 30
      LOGICAL :: mxIsChar, mxIsStruct
      mwSize :: mxGetString
      mwPointer :: mxGetPr, mxGetN, mxCreateDoubleMatrix
      mwPointer :: mxGetField, mxGetNumberOfElements
      INTEGER :: mexCallMATLABWithTrap
      REAL( KIND = wp ) :: mxGetScalar

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i, j, l, info
      INTEGER * 4 :: i4, m, n, nz, nz_col, alloc_stat
      INTEGER, PARAMETER :: int4 = KIND( i4 )

      mwSize :: status, m_mwsize, n_mwsize
      mwSize :: pat_arg, x0_arg, er_arg, ej_arg, eh_arg, ep_arg, con_arg, c_arg
      mwSize :: x_arg, i_arg, s_len

      mwPointer :: pat_in, x0_in, con_in, x0_pr
      mwPointer :: m_in, row_in, col_in, w_in
      mwPointer :: w_pr, row_pr, col_pr

      mwPointer input_x( 2 )
      mwPointer output_r( 2 ), output_j( 2 ), output_h( 2 ),output_p( 2 )
      mwPointer :: x_pr, y_pr, v_pr, s_in, s_pr
      mwPointer :: r_in, r_pr, j_in, j_pr, h_in, h_pr, p_in, p_pr

!     INTEGER :: iores, out = 11
      INTEGER :: iores, out = - 1
      CHARACTER ( len = 80 ) :: output_unit, filename
      CHARACTER ( len = 80 ) :: eval_r = REPEAT( ' ', 80 )
      CHARACTER ( len = 80 ) :: eval_j = REPEAT( ' ', 80 )
      CHARACTER ( len = 80 ) :: eval_h = REPEAT( ' ', 80 )
      CHARACTER ( len = 80 ) :: eval_p = REPEAT( ' ', 80 )
      LOGICAL :: opened, file_exists, initial_set = .FALSE.
      LOGICAL :: sparse_j
      LOGICAL :: debug = .FALSE.
!     LOGICAL :: debug = .TRUE.
      REAL( KIND = wp ) :: val
      CHARACTER ( len = 8 ) :: mode
      TYPE ( NLS_pointer_type ) :: NLS_pointer
      REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: IW

!  arguments for NLS

      TYPE ( NLPT_problem_type ) :: nlp
      TYPE ( GALAHAD_userdata_type ) :: userdata
      TYPE ( NLS_data_type ), SAVE :: data
      TYPE ( NLS_control_type ), SAVE :: control
      TYPE ( NLS_inform_type ) :: inform
      REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W

!  debugging

      IF ( out > 0 ) THEN
        WRITE( output_unit, "( I0 )" ) out
        filename = "debug." // TRIM( output_unit )
        OPEN( out, FILE = filename, FORM = 'FORMATTED',                        &
              STATUS = 'REPLACE', IOSTAT = iores )
      END IF

!  test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_nls requires at least 1 input argument' )

!  see if the first argument is a string or not. If it is, then we have a
!  sophisticated user

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN

          IF ( nrhs < 5 ) THEN
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_nls' )

!  arguments 2-5 are pattern, x0, eval_r and eval_j

          ELSE
            pat_arg = 2 ; x0_arg = 3 ; er_arg = 4 ; ej_arg = 5

!  pass through the remaining arguments, terminating when the next is control

            IF (  nrhs > 5 ) THEN
              IF ( mxIsStruct( prhs( 6 ) ) ) THEN
                con_arg = 6 ; eh_arg = - 1 ; ep_arg = - 1
              ELSE
                eh_arg = 6
                IF (  nrhs > 6 ) THEN
                  IF ( mxIsStruct( prhs( 7 ) ) ) THEN
                    con_arg = 7 ; ep_arg = - 1
                  ELSE
                    ep_arg = 7
                    IF (  nrhs > 7 ) THEN
                      IF ( mxIsStruct( prhs( 8 ) ) ) THEN
                        con_arg = 8
                      ELSE
                        CALL mexErrMsgTxt(                                     &
                          ' Too many input arguments to galahad_nls')
                      END IF
                    END IF
                  END IF
                END IF
              END IF
            END IF
          END IF
        END IF

!  simple (one-stop) user

      ELSE
        mode = 'all'
        IF ( nrhs < 4 ) THEN
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_nls' )

!  the first four arguments are pattern, x0, eval_r and eval_j

        ELSE
          pat_arg = 1 ; x0_arg = 2 ; er_arg = 3 ; ej_arg = 4

!  pass through the remaining arguments, terminating when the next is control

          IF (  nrhs > 4 ) THEN
            IF ( mxIsStruct( prhs( 5 ) ) ) THEN
              con_arg = 5 ; eh_arg = - 1 ; ep_arg = - 1
            ELSE
              eh_arg = 5
              IF (  nrhs > 5 ) THEN
                IF ( mxIsStruct( prhs( 6 ) ) ) THEN
                  con_arg = 6 ; ep_arg = - 1
                ELSE
                  ep_arg = 6
                  IF (  nrhs > 6 ) THEN
                    IF ( mxIsStruct( prhs( 7 ) ) ) THEN
                      con_arg = 7
                    ELSE
                      CALL mexErrMsgTxt(                                       &
                        ' Too many input arguments to galahad_nls')
                    END IF
                  END IF
                END IF
              END IF
            END IF
          END IF
        END IF
      END IF
      x_arg = 1 ; i_arg = 2

      IF ( nlhs > 2 )                                                          &
        CALL mexErrMsgTxt( ' galahad_nls provides at most 2 output arguments' )

!  initialize the internal structures for nls

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL NLS_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  if required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL NLS_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  check that NLS_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  find the name of the eval_r routine and ensure it exists

        i = mxGetString( prhs( er_arg ), eval_r, 80 )
        INQUIRE( file = TRIM( eval_r ) // '.m', exist = file_exists )
        IF ( .NOT. file_exists )                                               &
          CALL mexErrMsgTxt( TRIM( ' function evaluation file ' //             &
                              TRIM( eval_r ) // '.m does not exist' ) )

!  find the name of the eval_j routine and ensure it exists

        i = mxGetString( prhs( ej_arg ), eval_j, 80 )
        INQUIRE( file = TRIM( eval_j ) // '.m', exist = file_exists )
        IF ( .NOT. file_exists )                                               &
          CALL mexErrMsgTxt( TRIM( ' gradient evaluation file ' //             &
                              TRIM( eval_j ) // '.m does not exist' ) )


!  find the name of the eval_h routine and ensure it exists

        IF ( eh_arg > 0 ) THEN
          i = mxGetString( prhs( eh_arg ), eval_h, 80 )
          INQUIRE( file = TRIM( eval_h ) // '.m', exist = file_exists )
          IF ( .NOT. file_exists )                                             &
            CALL mexErrMsgTxt( TRIM( ' Hessian evaluation file ' //            &
                                TRIM( eval_h ) // '.m does not exist' ) )
        END IF

!  find the name of the eval_p routine and ensure it exists

        IF ( ep_arg > 0 ) THEN
          i = mxGetString( prhs( ep_arg ), eval_p, 80 )
          INQUIRE( file = TRIM( eval_p ) // '.m', exist = file_exists )
          IF ( .NOT. file_exists )                                             &
            CALL mexErrMsgTxt( TRIM( ' Hessian evaluation file ' //            &
            TRIM( eval_p ) // '.m does not exist' ) )
        END IF

!  if the control argument is present, extract the input control data

        s_len = slen
        IF ( con_arg > 0 ) THEN
          con_in = prhs( con_arg )
          IF ( .NOT. mxIsStruct( con_in ) )                                    &
            CALL mexErrMsgTxt( ' control input argument must be a structure' )
          CALL NLS_matlab_control_set( con_in, control, s_len )
        END IF

!  open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( output_unit, "( I0 )" ) control%error
          filename = "output_nls." // TRIM( output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( output_unit, "( I0 )" ) control%out
            filename = "output_nls." // TRIM( output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  create inform output structure

        CALL NLS_matlab_inform_create( plhs( i_arg ), NLS_pointer )

!  import the problem data

!  find the number of variables

        x0_in = prhs( x0_arg )
        n = INT( mxGetN( x0_in ), KIND = int4 ) ; nlp%n = n

!  allocate space for the solution

        ALLOCATE( nlp%X( n ), nlp%G( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )

!  set the starting point

        x0_pr = mxGetPr( x0_in )
        CALL MATLAB_copy_from_ptr( x0_pr, nlp%X, n )

!  find the number of residuals

        pat_in = prhs( pat_arg )
        m_in = mxGetField( pat_in, 1, 'm' )
        IF ( m_in /= 0 ) THEN
          val = mxGetScalar( m_in )
          m = INT( val, KIND = int4 ) ; nlp%m = m
        ELSE
          CALL mexErrMsgTxt( ' pattern.m not set on input to galahad_nls')
        END IF

!  allocate space for the residuals

        ALLOCATE( nlp%C( m ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )

!  see if the weights w are provided

        w_in = 0
        w_in = mxGetField( pat_in, 1, 'w' )
        IF ( w_in /= 0 ) THEN
          nz = INT( mxGetNumberOfElements( w_in ), KIND = int4 )
          IF ( nz < m ) CALL mexErrMsgTxt(                                     &
            ' length of pattern.w must be at least patter.m' //                &
            ' on input to galahad_nls' )

!  allocate space for the weights, and assign their values

          ALLOCATE( W( m ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )

          w_pr = mxGetPr( w_in )
          CALL MATLAB_copy_from_ptr( w_pr, W, m )
        ELSE
          nz = 0
        END IF

!  see if the Jacobian is dense or sparse

        row_in = mxGetField( pat_in, 1, 'j_row' )
        IF ( row_in /= 0 ) THEN
          nz = INT( mxGetNumberOfElements( row_in ), KIND = int4 )
        ELSE
          nz = 0
        END IF
        col_in = mxGetField( pat_in, 1, 'j_col' )
        IF ( col_in /= 0 ) THEN
          nz_col = INT( mxGetNumberOfElements( col_in ), KIND = int4 )
        ELSE
          nz_col = 0
        END IF
        IF ( nz /= nz_col ) CALL mexErrMsgTxt(                                 &
          ' lengths of pattern.j_row and pattern.j_col' //                     &
          ' must agree on input to galahad_nls' )

!  the Jacobian is sparse; obtain its structure

        sparse_j = nz > 0
        IF ( sparse_j ) THEN
          nlp%J%n = n ; nlp%J%m = m ; nlp%J%ne = nz
          CALL SMT_put( nlp%J%type, 'COORDINATE', alloc_stat )
          ALLOCATE( nlp%J%row( nlp%J%ne ), nlp%J%col( nlp%J%ne ),              &
                    IW( nz ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )

          row_pr = mxGetPr( row_in )
          CALL MATLAB_copy_from_ptr( row_pr, IW, nz )
          nlp%J%row( : nlp%J%ne ) = INT( IW( : nlp%J%ne ), KIND = int4 )

          col_pr = mxGetPr( col_in )
          CALL MATLAB_copy_from_ptr( col_pr, IW, nz )
          nlp%J%col( : nlp%J%ne ) = INT( IW( : nlp%J%ne ), KIND = int4 )

!  the Jacobian is dense

        ELSE
          nlp%J%ne = m * n
          CALL SMT_put( nlp%J%type, 'DENSE', alloc_stat )
        END IF
        ALLOCATE( nlp%J%val( nlp%J%ne ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
        control%jacobian_available = 2

!  see if the weighted Hessian is dense or sparse

        IF ( eh_arg > 0 ) THEN
          row_in = mxGetField( pat_in, 1, 'h_row' )
          IF ( row_in /= 0 ) THEN
            nz = INT( mxGetNumberOfElements( row_in ), KIND = int4 )
          ELSE
            nz = 0
          END IF
          col_in = mxGetField( pat_in, 1, 'h_col' )
          IF ( col_in /= 0 ) THEN
            nz_col = INT( mxGetNumberOfElements( col_in ), KIND = int4 )
          ELSE
            nz_col = 0
          END IF
          IF ( nz /= nz_col ) CALL mexErrMsgTxt(                               &
            ' lengths of pattern.h_row and pattern.h_col' //                   &
            ' must agree on input to galahad_nls' )

!  the Hessian is sparse; obtain its structure

          sparse_j = nz > 0
          IF ( sparse_j ) THEN
            nlp%H%n = n ; nlp%H%m = m ; nlp%H%ne = nz
            CALL SMT_put( nlp%H%type, 'COORDINATE', alloc_stat )
            IF ( ALLOCATED( IW ) ) DEALLOCATE( IW, STAT = alloc_stat )
            IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' deallocation failure ' )
            ALLOCATE( nlp%H%row( nlp%H%ne ), nlp%H%col( nlp%H%ne ),            &
                      IW( nz ), STAT = alloc_stat )
            IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )

            row_pr = mxGetPr( row_in )
            CALL MATLAB_copy_from_ptr( row_pr, IW, nz )
            nlp%H%row( : nlp%H%ne ) = INT( IW( : nlp%H%ne ), KIND = int4 )

            col_pr = mxGetPr( col_in )
            CALL MATLAB_copy_from_ptr( col_pr, IW, nz )
            nlp%H%col( : nlp%H%ne ) = INT( IW( : nlp%H%ne ), KIND = int4 )

!  the Hessian is dense

          ELSE
            nlp%H%ne = INT( ( n * ( n + 1 ) ) / 2, KIND = int4 )
            CALL SMT_put( nlp%H%type, 'DENSE', alloc_stat )
          END IF
          ALLOCATE( nlp%H%val( nlp%H%ne ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
          control%hessian_available = 2
        END IF

!  see if the residual-Hessian-vector product matrix is dense or sparse

        IF ( ep_arg > 0 ) THEN
          row_in = mxGetField( pat_in, 1, 'p_row' )
          IF ( row_in /= 0 ) THEN
            nz = INT( mxGetNumberOfElements( row_in ), KIND = int4 )
          ELSE
            nz = 0
          END IF
          col_in = mxGetField( pat_in, 1, 'p_col' )
          IF ( col_in /= 0 ) THEN
            nz_col = INT( mxGetNumberOfElements( col_in ), KIND = int4 )
          ELSE
            nz_col = 0
          END IF
          IF ( nz /= nz_col ) CALL mexErrMsgTxt(                               &
            ' lengths of pattern.p_row and pattern.p_col' //                   &
            ' must agree on input to galahad_nls' )

!  the product matrix is sparse; obtain its structure

          sparse_j = nz > 0
          IF ( sparse_j ) THEN
            nlp%P%n = m ; nlp%P%m = n ; nlp%P%ne = nz
            CALL SMT_put( nlp%P%type, 'SPARSE_BY_COLUMNS', alloc_stat )
            IF ( ALLOCATED( IW ) ) DEALLOCATE( IW, STAT = alloc_stat )
            IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' deallocation failure ' )
            ALLOCATE( nlp%P%ptr( m + 1 ), IW( nz ), STAT = alloc_stat )
            IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )

            col_pr = mxGetPr( col_in )
            CALL MATLAB_copy_from_ptr( col_pr, IW, nz )

            i = 1 ; nlp%P%ptr( 1 ) = 1
            DO l = 1, nlp%P%ne
              j = INT( IW( l ) )
              IF ( j > i ) THEN
                nlp%P%ptr( i + 1 : j ) = INT( l, KIND = int4 )
                i = j
              ELSE IF ( j < i ) THEN
                CALL mexErrMsgTxt( '  pattern.p_col not in increasing order' )
              END IF
            END DO
            nlp%P%ptr( i + 1 : m + 1 ) = INT( nz + 1, KIND = int4 )

            ALLOCATE( nlp%P%row( nlp%P%ne ), STAT = alloc_stat )
            IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
            row_pr = mxGetPr( row_in )
            CALL MATLAB_copy_from_ptr( row_pr, IW, nz )
            nlp%P%row( : nlp%P%ne ) = INT( IW( : nlp%P%ne ), KIND = int4 )

!  the product matrix is dense

          ELSE
            nlp%P%ne = n * m
            ALLOCATE( nlp%P%ptr( m + 1 ), nlp%P%row( nlp%P%ne ),               &
                      STAT = alloc_stat )
            IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
            l = 0
            DO j = 1, m
              nlp%P%ptr( j ) = INT( l + 1, KIND = int4 )
              nlp%P%row( l + 1 : l + n ) = INT( j, KIND = int4 )
              l = l + n
            END DO
            nlp%P%ptr( m + 1 ) = INT( l + 1, KIND = int4 )
            CALL SMT_put( nlp%P%type, 'DENSE_BY_COLUMNS', alloc_stat )
          END IF
          ALLOCATE( nlp%P%val( nlp%P%ne ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
        END IF

!  ensure that the model required is consistent with the evaluation
!  functions provided

        IF ( eh_arg <= 0 ) THEN
          IF ( control%model > 3 )  THEN
            control%model = 3
            CALL mexWarnMsgTxt( ' eval_h missing, control.model reset to 3' )
          END IF
        ELSE IF ( ep_arg <= 0 ) THEN
          IF ( control%model > 5 )  THEN
            control%model = 5
            CALL mexWarnMsgTxt( ' eval_p missing, control.model reset to 6' )
          END IF
        END IF

!  set for initial entry

        inform%status = 1
!       inform%status = -1
        m_mwsize = 1 ; n_mwsize = n
        input_x( 1 ) = mxCreateDoubleMatrix( m_mwsize, n_mwsize, 0 )
        n_mwsize = MAX( m, n )
        input_x( 2 ) = mxCreateDoubleMatrix( m_mwsize, n_mwsize, 0 )

!  loop to solve problem

        IF ( debug ) CALL mexWarnMsgTxt( ' start loop' )
        DO
          IF ( debug ) CALL mexWarnMsgTxt( ' enter solve' )
           IF ( w_in /= 0 ) THEN
!CALL mexWarnMsgTxt( ' w /= I' )
             CALL NLS_solve( nlp, control, inform, data, userdata, W = W )
           ELSE
            CALL NLS_solve( nlp, control, inform, data, userdata )
           END IF
          IF ( debug ) CALL mexWarnMsgTxt( ' end solve' )
!inform%status = 0
          SELECT CASE ( inform%status )

!  obtain the residuals

          CASE ( 2 )
            IF ( debug ) CALL mexWarnMsgTxt( ' 2' )
            x_pr = mxGetPr( input_x( 1 ) )
            CALL MATLAB_copy_to_ptr( nlp%X, x_pr, n )

!  evaluate r(x) in Matlab

            status = mexCallMATLABWithTrap( 2, output_r, 1, input_x, eval_r )

!  check to see that the evaluation succeeded

            s_in = output_r( 2 )
            s_pr = mxGetPr( s_in )
            CALL MATLAB_copy_from_ptr( s_pr, data%eval_status )

!  recover the residual value

            IF ( data%eval_status == 0 ) THEN
              r_in = output_r( 1 )
              r_pr = mxGetPr( r_in )
              CALL MATLAB_copy_from_ptr( r_pr, nlp%C, m )
            END IF

!  obtain the Jacobian

          CASE ( 3 )
            IF ( debug ) CALL mexWarnMsgTxt( ' 3' )
            x_pr = mxGetPr( input_x( 1 ) )
            CALL MATLAB_copy_to_ptr( nlp%X, x_pr, n )

!  evaluate J(x) in Matlab

            status = mexCallMATLABWithTrap( 2, output_j, 1, input_x, eval_j )

!  check to see that the evaluation succeeded

            s_in = output_j( 2 )
            s_pr = mxGetPr( s_in )
            CALL MATLAB_copy_from_ptr( s_pr, data%eval_status )

!  recover the Jacobian value

            IF ( data%eval_status == 0 ) THEN
              j_in = output_j( 1 )
              j_pr = mxGetPr( j_in )
              CALL MATLAB_copy_from_ptr( j_pr, nlp%J%val, nlp%J%ne )
            END IF

!  obtain the weighted Hessian

          CASE ( 4 )
            IF ( debug ) CALL mexWarnMsgTxt( ' 4' )
            x_pr = mxGetPr( input_x( 1 ) )
            CALL MATLAB_copy_to_ptr( nlp%X, x_pr, n )
            y_pr = mxGetPr( input_x( 2 ) )
            CALL MATLAB_copy_to_ptr( data%Y, y_pr, m )

!  evaluate H(x) in Matlab

            status = mexCallMATLABWithTrap( 2, output_h, 2, input_x, eval_h )

!  check to see that the evaluation succeeded

            s_in = output_h( 2 )
            s_pr = mxGetPr( s_in )
            CALL MATLAB_copy_from_ptr( s_pr, data%eval_status )

!  recover the Hessian value

            IF ( data%eval_status == 0 ) THEN
              h_in = output_h( 1 )
              h_pr = mxGetPr( h_in )
              CALL MATLAB_copy_from_ptr( h_pr, nlp%H%val, nlp%H%ne )
            END IF

!  obtain a Jacobian-vector product

!         CASE ( 5 )
!           data%U = data%U
!            data%eval_status = 0

!  obtain a Hessian-vector product

!         CASE ( 6 )
!           data%U = data%U + data%V
!            data%eval_status = 0

!  obtain the residual-Hessian-vector product matrix P(x,v)

          CASE ( 7 )
            IF ( debug ) CALL mexWarnMsgTxt( ' 7' )
            x_pr = mxGetPr( input_x( 1 ) )
            CALL MATLAB_copy_to_ptr( nlp%X, x_pr, n )
            v_pr = mxGetPr( input_x( 2 ) )
            CALL MATLAB_copy_to_ptr( data%V, v_pr, n )

!  evaluate P(x,v) in Matlab

            status = mexCallMATLABWithTrap( 2, output_p, 2, input_x, eval_p )

!  check to see that the evaluation succeeded

            s_in = output_p( 2 )
            s_pr = mxGetPr( s_in )
            CALL MATLAB_copy_from_ptr( s_pr, data%eval_status )

!  recover the product matrix value

            IF ( data%eval_status == 0 ) THEN
              p_in = output_p( 1 )
              p_pr = mxGetPr( p_in )
              CALL MATLAB_copy_from_ptr( p_pr, nlp%P%val, nlp%P%ne )
            END IF

!  terminal exit from loop

          CASE DEFAULT
            EXIT
          END SELECT
        END DO
        IF ( debug ) CALL mexWarnMsgTxt( ' end loop' )

!  print details to Matlab window

!       IF ( control%error > 0 ) CLOSE( control%error )
!       IF ( control%out > 0 .AND. control%error /= control%out )              &
!         CLOSE( control%out )

        IF ( control%out > 0 ) THEN
          REWIND( control%out, err = 500 )
          DO
            READ( control%out, "( A )", end = 500 ) str
            i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )
          END DO
        END IF
   500  CONTINUE
        IF ( debug ) CALL mexWarnMsgTxt( ' 500' )

!  output solution

        i4 = 1
        plhs( x_arg ) = MATLAB_create_real( n, i4 )
        x_pr = mxGetPr( plhs( x_arg ) )
        CALL MATLAB_copy_to_ptr( nlp%X, x_pr, n )
        IF ( debug ) CALL mexWarnMsgTxt( ' got x' )

!  record output information

        CALL NLS_matlab_inform_get( inform, NLS_pointer )
        IF ( debug ) CALL mexWarnMsgTxt( ' got inform' )

!  debug output

      IF ( .FALSE. ) THEN
!     IF ( out > 0 ) THEN
        REWIND( out, err = 600 )
        DO
          READ( out, "( A )", end = 600 ) str
          i = mexPrintf( TRIM( str ) // ACHAR( 10 ) )
        END DO
 600    CONTINUE
        CLOSE( out )
      END IF
      IF ( debug ) CALL mexWarnMsgTxt( ' write out' )

!  close any opened io units

      IF ( control%error > 0 ) THEN
        INQUIRE( control%error, OPENED = opened )
        IF ( opened ) CLOSE( control%error )
      END IF

      IF ( control%out > 0 ) THEN
        INQUIRE( control%out, OPENED = opened )
        IF ( opened ) CLOSE( control%out )
      END IF

!  check for errors

      IF ( inform%status < 0 )                                               &
        CALL mexWarnMsgTxt( ' Call to NLS_solve failed, check inform.status ' )
!       CALL mexErrMsgTxt( ' Call to NLS_solve failed ' )
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( nlp%J%row ) ) DEALLOCATE( nlp%J%row, STAT = info )
        IF ( ALLOCATED( nlp%J%col ) ) DEALLOCATE( nlp%J%col, STAT = info )
        IF ( ALLOCATED( nlp%J%val ) ) DEALLOCATE( nlp%J%val, STAT = info )
        IF ( ALLOCATED( nlp%H%row ) ) DEALLOCATE( nlp%H%row, STAT = info )
        IF ( ALLOCATED( nlp%H%col ) ) DEALLOCATE( nlp%H%col, STAT = info )
        IF ( ALLOCATED( nlp%H%val ) ) DEALLOCATE( nlp%H%val, STAT = info )
        IF ( ALLOCATED( nlp%P%row ) ) DEALLOCATE( nlp%P%row, STAT = info )
        IF ( ALLOCATED( nlp%P%col ) ) DEALLOCATE( nlp%P%col, STAT = info )
        IF ( ALLOCATED( nlp%P%val ) ) DEALLOCATE( nlp%P%val, STAT = info )
        IF ( ALLOCATED( nlp%C ) ) DEALLOCATE( nlp%C, STAT = info )
        IF ( ALLOCATED( nlp%G ) ) DEALLOCATE( nlp%G, STAT = info )
        IF ( ALLOCATED( nlp%X ) ) DEALLOCATE( nlp%X, STAT = info )
        IF ( ALLOCATED( W ) ) DEALLOCATE( W, STAT = alloc_stat )
        IF ( ALLOCATED( IW ) ) DEALLOCATE( IW, STAT = alloc_stat )
        CALL NLS_terminate( data, control, inform )
      END IF

      RETURN

      END SUBROUTINE mexFunction
