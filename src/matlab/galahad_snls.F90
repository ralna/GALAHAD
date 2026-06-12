#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.5 - 2026-03-26 AT 08:50 GMT.

! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!                 MEX INTERFACE TO GALAHAD_SNLS
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
!
!  find a local (unconstrained) minimizer of a differentiable, possibly
!  weighted, nonlinear least-squares objective function
!    f(x) = 1/2 sum_i=1^m_r w_i r_i^2(x)
!  where the n real variables x are required to lie within
!  the interection multiple non-overlapping regular simplices
!    sum_{C_j} x_i = 1, x_{C_j} >= 0 for j = 1,...,m_c
!  using a regularization method. Advantage may be taken of sparsity 
!  in the problem
!
!  Terminology -
!
!  r_i(x) is the ith residual, and the vector r(x) are the residuals.
!  Weights w_i>0 may be provided, but otherwise will be assumed to be 1.
!  The matrix Jr(x) for which Jr_i,j = d r_i(x) / dx_j is the Jacobian
!  of the residuals
!
!  Simple usage -
!
!  to find the minimizer
!   [ x, inform ]
!    = galahad_snls( pattern, x0, eval_r, eval_jr, cohort, w, control )
!
!  Sophisticated usage -
!
!  to initialize data and control structures prior to solution
!   [ control ]
!    = galahad_snls( 'initial' )
!
!  to solve the problem using existing data structures
!   [ x, inform ] = galahad_snls( 'existing', pattern, x0, ...
!                                  eval_r, eval_jr, cohort, w, control )
!
!  to remove data structures after solution
!   galahad_snls( 'final' )
!
!  Usual Input -
!     pattern: a structure that indicates the sparsity pattern of
!              the Jacobian matrix. Components are -
!                 m_r: number of residuals (compulsory)
!                 jr_row, jr_col: row and column indices of the nonzeros
!                 in the Jacobian of the residuals Jr(x) (optional).
!                 If absent, Jr(x) is assumed dense and stored by rows
!      x0: an initial estimate of the minimizer
!      eval_r: a user-provided subroutine named eval_r.m for which
!                [r,status] = eval_r(x)
!              returns the value of the vector of residual functions
!              r at x; r(i) contains r_i(x).
!              status should be set to 0 if the evaluation succeeds,
!              and a non-zero value if the evaluation fails.
!      eval_jr: a user-provided subroutine named eval_jr.m for which
!                [j_val,status] = eval_jr(x)
!              returns a vector of values of the Jacobian Jr(x) of the
!              residuals stored by rows. If Jr(x) is dense, the n*(i-1)+j-th
!              conponent of j_val should contain the derivative
!              dr_i(x)/dx_j dx_j at x, 1<=i<=m, 1<=j<=n. If Jr(x) is sparse,
!              the k-th component of j_val contains the derivative
!              dr_i(x)/dx_i dx_j for which i=jr_row(k) and j=jr_col(k),
!              as set in the structure pattern (see above).
!              status should be set to 0 if the evaluation succeeds,
!              and a non-zero value if the evaluation fails.
!
!  Optional Input -
!     cohort: the n-vector of cohorts, so that variable x_i is in cohort C_j
!             if cohort[i] = j, and x_i is not constrained if cohort[i] = 0.
!             If absent, a single cohort containing all variables will be used
!     w: the m_c-vector of weights w for which W=diag(w). If absent, 
!        weights of one will be used. ** N.B. If n=m_r and both w and cohort 
!        are provided, cohort must proceed w in the calling sequence
!     control: a structure containing control parameters.
!              The components are of the form control.value, where
!              value is the name of the corresponding component of
!              the derived type SNLS_control_type as described in
!              the manual for the fortran 90 package GALAHAD_SNLS.
!              See: http://galahad.rl.ac.uk/galahad-www/doc/snls.pdf
!
!  Usual Output -
!          x: a first-order criticl point that is usually a local minimizer.
!
!  Optional Output -
!     control: see 'initial' above. Returned values are the defaults
!     inform: a structure containing information parameters
!             The components are of the form inform.value, where
!             value is the name of the corresponding component of the
!             derived type SNLS_inform_type as described in the manual
!             for the fortran 90 package GALAHAD_SNLS. The components
!             of inform.time, inform.SLLS_inform and inform.SLLSB_inform
!             are themselves structures, holding the components of the 
!             derived types SNLS_time_type, SLLS_inform_type and
!             SLLSB_inform_type, respectively.
!             See: http://galahad.rl.ac.uk/galahad-www/doc/snls.pdf
!     aux: a structure containing Lagrange multipliers and constraint status
!      aux.y: Largrange multipliers y corresponding to the simplex constraints
!      aux.z: dual variables z corresponding to the non-negativity constraints
!           x_i >= 0
!      aux.r: values of the residuals r(x) = Ao x - b
!      aux.g: values of the gradients g(x) = Ao^T W r(x)
!      aux.x_status: vector indicating the status of the bound constraints
!              x_status(i) < 0 if (x)_i = 0
!              x_status(i) = 0 if (x)_i > 0
!
! *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 5.5. March 26th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
      USE GALAHAD_MATLAB
!     USE GALAHAD_TRANSFER_MATLAB
      USE GALAHAD_SNLS_MATLAB_TYPES
      USE GALAHAD_SNLS_double
!     USE GALAHAD_USERDATA_double
      IMPLICIT NONE
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

! ------------------------- Do not change -------------------------------

!  Keep the above subroutine, argument, and function declarations for use
!  in all your fortran mex files.
!
      INTEGER * 4 :: nlhs, nrhs
      mwPointer :: plhs( * ), prhs( * )

      INTEGER, PARAMETER :: slen = 30
      LOGICAL :: mxIsChar, mxIsStruct, mxIsClass
      mwSize :: mxGetString
      mwSize :: mxIsNumeric
      mwPointer :: mxCreateStructMatrix, mxGetPr, mxGetN, mxCreateDoubleMatrix
      mwPointer :: mxGetField, mxGetNumberOfElements
      INTEGER :: mexCallMATLABWithTrap
      REAL( KIND = wp ) :: mxGetScalar

      INTEGER ::  mexPrintf
      CHARACTER ( LEN = 200 ) :: str

! -----------------------------------------------------------------------

!  local variables

      INTEGER :: i, n_v, info
      INTEGER * 4 :: i4, m_r, n, nz, nz_col, alloc_stat
      INTEGER, PARAMETER :: int4 = KIND( i4 )

      mwSize :: status, m_mwsize, n_mwsize
      mwSize :: pat_arg, x0_arg, er_arg, ej_arg
      mwSize :: co_arg, w_arg, c_arg, optional_arg
      mwSize :: x_arg, i_arg, aux_arg
      mwSize :: s_len
      mwPointer :: pat_in, x0_in
      mwPointer :: m_r_in, row_in, col_in, co_in, w_in, c_in, i_in
      mwPointer :: s_in, j_in, r_in
      mwPointer :: row_pr, col_pr, x0_pr, co_pr, w_pr
      mwPointer :: s_pr, j_pr, x_pr, y_pr, z_pr, r_pr, g_pr, x_status_pr
      mwPointer :: aux_y_pr, aux_z_pr, aux_r_pr, aux_g_pr, aux_x_status_pr
      mwPointer input_x( 2 )
      mwPointer output_r( 2 ), output_j( 2 )

      CHARACTER ( LEN = 1 ) :: array
      CHARACTER ( LEN = 80 ) :: char_output_unit, filename
      CHARACTER ( LEN = 80 ) :: eval_r = REPEAT( ' ', 80 )
      CHARACTER ( LEN = 80 ) :: eval_jr = REPEAT( ' ', 80 )
      LOGICAL :: opened, file_exists, initial_set = .FALSE.
      LOGICAL :: sparse_j
      LOGICAL :: debug = .FALSE.
!     LOGICAL :: debug = .TRUE.
!     INTEGER :: iores, out = 11
      INTEGER :: iores, out = - 1
      REAL( KIND = wp ) :: val
      CHARACTER ( len = 8 ) :: mode
      TYPE ( SNLS_pointer_type ) :: SNLS_pointer
      REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: IW

      mwSize, PARAMETER :: naux = 5
      CHARACTER ( LEN = 8 ), PARAMETER :: faux( naux ) = (/                    &
           'y       ', 'z       ', 'r       ', 'g       ', 'x_status' /)

!  arguments for SNLS

      TYPE ( NLPT_problem_type ) :: nlp
      TYPE ( USERDATA_type ) :: userdata
      TYPE ( REVERSE_type ) :: reverse
      TYPE ( SNLS_data_type ), SAVE :: data
      TYPE ( SNLS_control_type ), SAVE :: control
      TYPE ( SNLS_inform_type ) :: inform
      REAL ( KIND = wp ), ALLOCATABLE :: real_cohort( : )

!  debugging

      IF ( out > 0 ) THEN
        WRITE( char_output_unit, "( I0 )" ) out
        filename = "debug." // TRIM( char_output_unit )
        OPEN( out, FILE = filename, FORM = 'FORMATTED',                        &
              STATUS = 'REPLACE', IOSTAT = iores )
      END IF


!  Test input/output arguments

      IF ( nrhs < 1 )                                                          &
        CALL mexErrMsgTxt( ' galahad_snls requires at least 1 input argument' )
      IF ( nlhs > 3 )                                                          &
        CALL mexErrMsgTxt( ' galahad_snls provides at most 3 output arguments' )

!  Check existence of compulsory arguments

!  sophisticated user

      IF ( mxIsChar( prhs( 1 ) ) ) THEN
        i = mxGetString( prhs( 1 ), mode, 8 )
        IF ( .NOT. ( TRIM( mode ) == 'initial' .OR.                            &
                     TRIM( mode ) == 'final' ) ) THEN
          IF ( nrhs < 5 )                                                      &
            CALL mexErrMsgTxt( ' Too few input arguments to galahad_snls' )
          pat_arg = 2 ; x0_arg = 3 ; er_arg = 4 ; ej_arg = 5 ; optional_arg = 6
          x_arg = 1 ; i_arg = 2 ; aux_arg = 3
          IF ( nrhs > 9 )                                                      &
            CALL mexErrMsgTxt( ' Too many input arguments to galahad_snls' )
        END IF

!  simple (one-stop) user

      ELSE
        mode = 'all'
        IF ( nrhs < 4 )                                                        &
          CALL mexErrMsgTxt( ' Too few input arguments to galahad_snls' )
        pat_arg = 1 ; x0_arg = 2 ; er_arg = 3 ; ej_arg = 4 ; optional_arg = 5
        x_arg = 1 ; i_arg = 2 ; aux_arg = 3
        IF ( nrhs > 8 )                                                        &
          CALL mexErrMsgTxt( ' Too many input arguments to galahad_snls' )
      END IF

!  initialize the internal structures for snls

      IF ( TRIM( mode ) == 'initial' .OR. TRIM( mode ) == 'all' ) THEN
        initial_set = .TRUE.
        CALL SNLS_initialize( data, control, inform )
        IF ( TRIM( mode ) == 'initial' ) THEN
          c_arg = 1
          IF ( nlhs > c_arg )                                                  &
            CALL mexErrMsgTxt( ' too many output arguments required' )

!  if required, return the default control parameters

          IF ( nlhs > 0 )                                                      &
            CALL SNLS_matlab_control_get( plhs( c_arg ), control )
          RETURN
        END IF
      END IF

      IF ( .NOT. TRIM( mode ) == 'final' ) THEN

!  check that SNLS_initialize has been called

        IF ( .NOT. initial_set )                                               &
          CALL mexErrMsgTxt( ' "initial" must be called first' )

!  Find the name of the eval_r routine and ensure it exists

        i = mxGetString( prhs( er_arg ), eval_r, 80 )
        INQUIRE( file = TRIM( eval_r ) // '.m', exist = file_exists )
        IF ( .NOT. file_exists )                                               &
          CALL mexErrMsgTxt( TRIM( ' function evaluation file ' //             &
                              TRIM( eval_r ) // '.m does not exist' ) )

!  Find the name of the eval_jr routine and ensure it exists

        i = mxGetString( prhs( ej_arg ), eval_jr, 80 )
        INQUIRE( file = TRIM( eval_jr ) // '.m', exist = file_exists )
        IF ( .NOT. file_exists )                                               &
          CALL mexErrMsgTxt( TRIM( ' Jacobian evaluation file ' //             &
                             TRIM( eval_jr ) // '.m does not exist' ) )

!  Find the number of variables

        x0_in = prhs( x0_arg )
        IF ( mxIsNumeric( x0_in ) == 0 )                                       &
           CALL mexErrMsgTxt( ' There must be a vector x_0' )
        n = INT( mxGetN( x0_in ), KIND = int4 ) ; nlp%n = n

!  Import the problem data, allocate space for the solution

        ALLOCATE( nlp%X( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' snls allocation failure ' )

!  Input x_0 as initial x

        x0_pr = mxGetPr( x0_in )
        CALL MATLAB_copy_from_ptr( x0_pr, nlp%X, n )

!  Find the number of residuals

        pat_in = prhs( pat_arg )
        m_r_in = mxGetField( pat_in, 1, 'm_r' )
        IF ( m_r_in /= 0 ) THEN
          val = mxGetScalar( m_r_in )
          m_r = INT( val, KIND = int4 ) ; nlp%m_r = m_r
        ELSE
          CALL mexErrMsgTxt( ' pattern.m not set on input to galahad_snls')
        END IF
        IF ( debug ) CALL mexWarnMsgTxt( ' r set' )

!  Allocate space for the residuals

        ALLOCATE( nlp%R( m_r ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )

!  See if the Jacobian is dense or sparse

        row_in = mxGetField( pat_in, 1, 'jr_row' )
        IF ( row_in /= 0 ) THEN
          nz = INT( mxGetNumberOfElements( row_in ), KIND = int4 )
        ELSE
          nz = 0
        END IF
        col_in = mxGetField( pat_in, 1, 'jr_col' )
        IF ( col_in /= 0 ) THEN
          nz_col = INT( mxGetNumberOfElements( col_in ), KIND = int4 )
        ELSE
          nz_col = 0
        END IF
        IF ( nz /= nz_col ) CALL mexErrMsgTxt(                                 &
          ' lengths of pattern.jr_row and pattern.jr_col' //                   &
          ' must agree on input to galahad_snls' )

!  The Jacobian is sparse; obtain its structure

        sparse_j = nz > 0
        IF ( sparse_j ) THEN
          nlp%Jr%n = n ; nlp%Jr%m = m_r ; nlp%Jr%ne = nz
          CALL SMT_put( nlp%Jr%type, 'COORDINATE', alloc_stat )
          ALLOCATE( nlp%Jr%row( nlp%Jr%ne ), nlp%Jr%col( nlp%Jr%ne ),          &
                    IW( nz ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )

          row_pr = mxGetPr( row_in )
          CALL MATLAB_copy_from_ptr( row_pr, IW, nz )
          nlp%Jr%row( : nlp%Jr%ne ) = INT( IW( : nlp%Jr%ne ), KIND = int4 )

          col_pr = mxGetPr( col_in )
          CALL MATLAB_copy_from_ptr( col_pr, IW, nz )
          nlp%Jr%col( : nlp%Jr%ne ) = INT( IW( : nlp%Jr%ne ), KIND = int4 )

!  The Jacobian is dense

        ELSE
          nlp%Jr%ne = m_r * n
          CALL SMT_put( nlp%Jr%type, 'DENSE', alloc_stat )
        END IF
        ALLOCATE( nlp%Jr%val( nlp%Jr%ne ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
        IF ( debug ) CALL mexWarnMsgTxt( ' Jr set' )

!  Sort through optional arguments

        co_arg = - 1  ; w_arg = - 1 ; c_arg = - 1
        DO i = optional_arg, nrhs
          i_in = prhs( i )

!  Input optional control data

          IF ( mxIsStruct( i_in ) ) THEN ! input control
            c_arg = i
            c_in = i_in
            s_len = slen
            CALL SNLS_matlab_control_set( c_in, control, s_len )
            IF ( debug ) CALL mexWarnMsgTxt( ' option control set' )

!  Input optional array argument

          ELSE IF ( mxIsClass( i_in, 'double') ) THEN
            IF ( co_arg > 0 .AND. w_arg > 0 ) CALL mexErrMsgTxt(               &
              ' array input arguments cohort and w already provided' )
            n_v = INT( mxGetN( i_in ) )
            IF ( nlp%n /= nlp%m_r ) THEN
              IF ( n_v == nlp%m_r ) THEN  ! argument is w
                array = 'w'
              ELSE IF ( n_v == nlp%n ) THEN ! argument is cohort
                array = 'c'
              ELSE ! argument not recognised
                CALL mexErrMsgTxt( ' array input argument not recognised' )
              END IF              
            ELSE
              IF ( n_v == nlp%n ) THEN  ! argument is x_s or w
                IF ( co_arg < 0 ) THEN
                  array = 'c'
                ELSE IF ( w_arg < 0 ) THEN
                  array = 'w'
                ELSE
                  CALL mexErrMsgTxt( ' too may optional array input arguments' )
                END IF
              ELSE ! argument not recognised
                CALL mexErrMsgTxt( ' arrays input argument incorrect size' )
              END IF              
            END IF

!  Input cohort

            IF ( array == 'c' ) THEN
              co_arg = i
              co_in = i_in
              co_pr = mxGetPr( co_in )

!  kludge to avoid Mex incorrect type error for supposed integers

              ALLOCATE( real_cohort( nlp%n ), STAT = info )
              CALL MATLAB_copy_from_ptr( co_pr, real_cohort, nlp%n )
              ALLOCATE( nlp%COHORT( nlp%n ), STAT = info )
              nlp%COHORT( : nlp%n ) = INT( real_cohort, KIND = KIND( nlp%n ) )
              DEALLOCATE( real_cohort, STAT = info )
              nlp%m_c = MAXVAL( nlp%COHORT( : nlp%n ) )
              IF ( debug ) CALL mexWarnMsgTxt( ' option cohort set' )

!  Input w

            ELSE
              w_arg = i
              w_in = i_in
              w_pr = mxGetPr( w_in )
              ALLOCATE( nlp%W( nlp%m_r ), STAT = info )
              CALL MATLAB_copy_from_ptr( w_pr, nlp%W, nlp%m_r )
              IF ( debug ) CALL mexWarnMsgTxt( ' option w set' )
            END IF
          ELSE
            CALL mexErrMsgTxt( ' Unknown optional argument' )
          END IF
        END DO
        IF ( debug ) CALL mexWarnMsgTxt( ' options set' )

!  Open i/o units

        IF ( control%error > 0 ) THEN
          WRITE( char_output_unit, "( I0 )" ) control%error
          filename = "output_snls." // TRIM( char_output_unit )
          OPEN( control%error, FILE = filename, FORM = 'FORMATTED',            &
                STATUS = 'REPLACE', IOSTAT = iores )
        END IF

        IF ( control%out > 0 ) THEN
          INQUIRE( control%out, OPENED = opened )
          IF ( .NOT. opened ) THEN
            WRITE( char_output_unit, "( I0 )" ) control%out
            filename = "output_snls." // TRIM( char_output_unit )
            OPEN( control%out, FILE = filename, FORM = 'FORMATTED',            &
                  STATUS = 'REPLACE', IOSTAT = iores )
          END IF
        END IF

!  Create inform output structure

        control%jacobian_available = 2
        CALL SNLS_matlab_inform_create( plhs( i_arg ), SNLS_pointer )

!  Prepare for initial entry

        inform%status = 1
!       inform%status = -1
        m_mwsize = 1 ; n_mwsize = n
        input_x( 1 ) = mxCreateDoubleMatrix( m_mwsize, n_mwsize, 0 )
        n_mwsize = MAX( m_r, n )
        input_x( 2 ) = mxCreateDoubleMatrix( m_mwsize, n_mwsize, 0 )

!  Loop to solve problem

        IF ( debug ) CALL mexWarnMsgTxt( ' start loop' )
        DO

!  Call the snls solver 

          IF ( debug ) CALL mexWarnMsgTxt( ' enter solve' )
          CALL SNLS_solve( nlp, control, inform, data, userdata,               &
                           reverse = reverse )
          IF ( debug ) CALL mexWarnMsgTxt( ' end solve' )
!inform%status = 0
          SELECT CASE ( inform%status )

!  Obtain the residuals

          CASE ( 2 )
            IF ( debug ) CALL mexWarnMsgTxt( ' 2' )
            x_pr = mxGetPr( input_x( 1 ) )
            CALL MATLAB_copy_to_ptr( nlp%X, x_pr, n )

!  Evaluate r(x) in Matlab

            status = mexCallMATLABWithTrap( 2, output_r, 1, input_x, eval_r )

!  Check to see that the evaluation succeeded

            s_in = output_r( 2 )
            s_pr = mxGetPr( s_in )
            CALL MATLAB_copy_from_ptr( s_pr, reverse%eval_status )

!  Recover the residual value

            IF ( reverse%eval_status == 0 ) THEN
              r_in = output_r( 1 )
              r_pr = mxGetPr( r_in )
              CALL MATLAB_copy_from_ptr( r_pr, nlp%R, m_r )
            END IF

!  Obtain the Jacobian

          CASE ( 3 )
            IF ( debug ) CALL mexWarnMsgTxt( ' 3' )
            x_pr = mxGetPr( input_x( 1 ) )
            CALL MATLAB_copy_to_ptr( nlp%X, x_pr, n )

!  Evaluate Jr(x) in Matlab

            status = mexCallMATLABWithTrap( 2, output_j, 1, input_x, eval_jr )

!  Check to see that the evaluation succeeded

            s_in = output_j( 2 )
            s_pr = mxGetPr( s_in )
            CALL MATLAB_copy_from_ptr( s_pr, reverse%eval_status )

!  Recover the Jacobian value

            IF ( reverse%eval_status == 0 ) THEN
              j_in = output_j( 1 )
              j_pr = mxGetPr( j_in )
              CALL MATLAB_copy_from_ptr( j_pr, nlp%Jr%val, nlp%Jr%ne )
            END IF

!  Terminal exit from loop

          CASE DEFAULT
            EXIT
          END SELECT
        END DO
        IF ( debug ) CALL mexWarnMsgTxt( ' end loop' )

!  Print details to Matlab window

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

!  Output solution

        i4 = 1
        plhs( x_arg ) = MATLAB_create_real( n, i4 )
        x_pr = mxGetPr( plhs( x_arg ) )
        CALL MATLAB_copy_to_ptr( nlp%X, x_pr, n )
        IF ( debug ) CALL mexWarnMsgTxt( ' got x' )

!  Record output information

        CALL SNLS_matlab_inform_get( inform, SNLS_pointer )
        IF ( debug ) CALL mexWarnMsgTxt( ' got inform' )

!  If required, set auxiliary output containing Lagrange multipliesr and
!  constraint bound status

        IF ( nlhs == aux_arg ) THEN

!  set up space for the auxiliary arrays

          plhs( aux_arg ) = mxCreateStructMatrix( 1, 1, naux, faux )

          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'y', nlp%m_c, aux_y_pr )
          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'z', nlp%n, aux_z_pr )
          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'r', nlp%m_r, aux_r_pr )
          CALL MATLAB_create_real_component( plhs( aux_arg ),                  &
            'g', nlp%n, aux_g_pr )
          CALL MATLAB_create_integer_component( plhs( aux_arg ),               &
            'x_status', nlp%n, aux_x_status_pr )

!  copy the values

          y_pr = mxGetPr( aux_y_pr )
          CALL MATLAB_copy_to_ptr( nlp%Y, y_pr, nlp%m_c )
          z_pr = mxGetPr( aux_z_pr )
          CALL MATLAB_copy_to_ptr( nlp%Z, z_pr, nlp%n )
          r_pr = mxGetPr( aux_r_pr )
          CALL MATLAB_copy_to_ptr( nlp%R, r_pr, nlp%m_r )
          g_pr = mxGetPr( aux_g_pr )
          CALL MATLAB_copy_to_ptr( nlp%G, g_pr, nlp%n )
          x_status_pr = mxGetPr( aux_x_status_pr )
          CALL MATLAB_copy_to_ptr( nlp%X_status, x_status_pr, nlp%n )
        END IF

!  debug output

        IF ( .FALSE. ) THEN
!       IF ( out > 0 ) THEN
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

      IF ( inform%status < 0 )                                                 &
        CALL mexWarnMsgTxt( ' Call to SNLS_solve failed, check inform.status ' )
!       CALL mexErrMsgTxt( ' Call to SNLS_solve failed ' )
      END IF

!  all components now set

      IF ( TRIM( mode ) == 'final' .OR. TRIM( mode ) == 'all' ) THEN
        IF ( ALLOCATED( nlp%Jr%type ) ) DEALLOCATE( nlp%Jr%type, STAT = info )
        IF ( ALLOCATED( nlp%Jr%row ) ) DEALLOCATE( nlp%Jr%row, STAT = info )
        IF ( ALLOCATED( nlp%Jr%col ) ) DEALLOCATE( nlp%Jr%col, STAT = info )
        IF ( ALLOCATED( nlp%Jr%val ) ) DEALLOCATE( nlp%Jr%val, STAT = info )
        IF ( ALLOCATED( nlp%COHORT ) ) DEALLOCATE( nlp%COHORT, STAT = info )
        IF ( ALLOCATED( nlp%W ) ) DEALLOCATE( nlp%W, STAT = info )
        IF ( ALLOCATED( nlp%X ) ) DEALLOCATE( nlp%X, STAT = info )
        IF ( ALLOCATED( nlp%Y ) ) DEALLOCATE( nlp%Y, STAT = info )
        IF ( ALLOCATED( nlp%Z ) ) DEALLOCATE( nlp%Z, STAT = info )
        IF ( ALLOCATED( nlp%R ) ) DEALLOCATE( nlp%R, STAT = info )
        IF ( ALLOCATED( nlp%G ) ) DEALLOCATE( nlp%G, STAT = info )
        IF ( ALLOCATED( nlp%X_status ) ) DEALLOCATE( nlp%X_status, STAT = info )
        IF ( ALLOCATED( IW ) ) DEALLOCATE( IW, STAT = alloc_stat )
        CALL SNLS_terminate( data, control, inform )
      END IF

      RETURN

      END SUBROUTINE mexFunction
