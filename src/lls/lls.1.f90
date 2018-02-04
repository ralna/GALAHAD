! THIS VERSION: GALAHAD 2.1 - 20/10/2007 AT 17:00 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ L L S   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started October 20th 2007
!   originally released GALAHAD Version 2.1. October 20th 2007

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_LLS_double

!      ----------------------------------------
!     |                                        |
!     | Solve the linear least-squares problem |
!     |                                        |
!     |   minimize     || A x + c ||_2         |
!     |   subject to     || x ||_2 <= Delta    |
!     |                                        |
!     | using a preconditined CG method        |
!     |                                        |
!      ----------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_SPACE_double
!     USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SBLS_double
      USE GALAHAD_GLTR_double
      USE GALAHAD_SPECFILE_double
   
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LLS_initialize, LLS_read_specfile, LLS_solve, LLS_terminate,   &
                QPT_problem_type, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: LLS_time_type
        REAL :: total, factorize, solve
      END TYPE

      TYPE, PUBLIC :: LLS_control_type
        INTEGER :: error, out, print_level
        INTEGER :: factorization, max_col, indmin, valmin, len_glsmin, itref_max
        INTEGER :: cg_maxit, preconditioner, new_a
        REAL ( KIND = wp ) :: pivot_tol, pivot_tol_for_basis, zero_pivot
        REAL ( KIND = wp ) :: inner_fraction_opt, radius
        REAL ( KIND = wp ) :: max_infeasibility_relative
        REAL ( KIND = wp ) :: max_infeasibility_absolute
        REAL ( KIND = wp ) :: inner_stop_relative, inner_stop_absolute
        LOGICAL :: remove_dependencies, find_basis_by_transpose
        LOGICAL :: space_critical, deallocate_error_fatal
        CHARACTER ( LEN = 30 ) :: prefix
        TYPE ( SBLS_control_type ) :: SBLS_control
        TYPE ( GLTR_control_type ) :: GLTR_control
      END TYPE

      TYPE, PUBLIC :: LLS_data_type
        LOGICAL :: new_h, new_c
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ATc
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VECTOR
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WORK
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Ax
        TYPE ( SBLS_data_type ) :: SBLS_data
        TYPE ( SMT_type ) :: H, C
        TYPE ( GLTR_data_type ) :: GLTR_data
      END TYPE

      TYPE, PUBLIC :: LLS_inform_type
        INTEGER :: status, alloc_status, cg_iter
        INTEGER :: factorization_integer, factorization_real
        REAL ( KIND = wp ) :: obj, norm_x
        TYPE ( LLS_time_type ) :: time
        TYPE ( SBLS_inform_type ) :: SBLS_inform
        TYPE ( GLTR_info_type ) :: GLTR_inform
        CHARACTER ( LEN = 80 ) :: bad_alloc
      END TYPE

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

   CONTAINS

!-*-*-*-*-*-   L L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LLS_initialize( data, control )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LLS. This routine should be called before
!  LLS_solve
! 
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. Components are -
!
!  INTEGER control parameters:
!
!   error. Error and warning diagnostics occur on stream error 
!   
!   out. General output occurs on stream out
!   
!   print_level. The level of output required is specified by print_level
!   
!   factorization. The factorization to be used.
!    Possible values are
!
!      0  automatic 
!      1  Schur-complement factorization
!      2  augmented-system factorization
!
!   max_col. The maximum number of nonzeros in a column of A which is permitted
!    with the Schur-complement factorization
!
!   indmin. An initial guess as to the integer workspace required by SILS
!
!   valmin. An initial guess as to the real workspace required by SILS
! 
!   len_glsmin. An initial guess as to the workspace required by GLS
!
!   itref_max. The maximum number of iterative refinements allowed
!
!   cg_maxit. The maximum number of CG iterations allowed. If cg_maxit < 0,
!     this number will be reset to the dimension of the system + 1
!
!   preconditioner. The preconditioner to be used for the CG is defined by 
!    preconditioner. Possible values are
!
!    variable:
!
!      0  no preconditioner
!
!    explicit factorization:
!
!      1  G = I
!
!    implicit factorization:
!
!      -1  G_11 = 0, G_21 = 0, G_22 = I
!
!   new_a. How much of A has changed since the previous factorization.
!    Possible values are
!
!      0  unchanged
!      1  values but not indices have changed
!      2  values and indices have changed 
!
!  REAL control parameters:
!
!   pivot_tol. The threshold pivot used by the matrix factorization.
!    See the documentation for SILS for details
!
!   pivot_tol_for_basis. The threshold pivot used by the matrix 
!    factorization when finding the basis.
!    See the documentation for GLS for details
!
!   zero_pivot. Any pivots smaller than zero_pivot in absolute value will 
!    be regarded to be zero when attempting to detect linearly dependent 
!    constraints
!
!   inner_fraction_opt. a search direction which gives at least 
!    inner_fraction_opt times the optimal model decrease will be found
!
!   inner_stop_relative and inner_stop_absolute. The search direction is
!    considered as an acceptable approximation to the minimizer of the
!    model if the gradient of the model in the preconditioning(inverse) 
!    norm is less than 
!     max( inner_stop_relative * initial preconditioning(inverse)
!                                 gradient norm, inner_stop_absolute )
!
!   max_infeasibility_relative and max_infeasibility_absolute. If the 
!     constraints are believed to be rank defficient and the residual
!     at a "typical" feasiblke point is larger than
!      max( max_infeasibility_relative * norm A, max_infeasibility_absolute )
!     the problem will be marked as infeasible
!
!   radius. An upper bound on the permitted step
!
!  LOGICAL control parameters:
!
!   find_basis_by_transpose. If true, implicit factorization preconditioners
!    will be based on a basis of A found by examining A's transpose
!
!   remove_dependencies. If true, the equality constraints will be preprocessed
!    to remove any linear dependencies
!
!   space_critical. If true, every effort will be made to use as little
!    space as possible. This may result in longer computation times
!
!   deallocate_error_fatal. If true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue
!
!  CHARACTER control parameters:
!
!  prefix (len=30). All output lines will be prefixed by 
!    %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( LLS_data_type ), INTENT( OUT ) :: data
      TYPE ( LLS_control_type ), INTENT( OUT ) :: control        

!  Set control parameters

!  Integer parameters

      control%error  = 6
      control%out  = 6
      control%print_level = 0
      control%factorization = 0
      control%preconditioner = 0
      control%max_col = 35
      control%indmin = 1000
      control%valmin = 1000
      control%len_glsmin = 1000
      control%itref_max = 1
!     control%cg_maxit = - 1
      control%cg_maxit = 200
      control%new_a = 2

!  Real parameters

      control%radius = SQRT( point1 * HUGE( one ) )
      control%pivot_tol = 0.01_wp
!     control%pivot_tol = epsmch ** 0.75
      control%pivot_tol_for_basis = half
      control%zero_pivot = epsmch ** 0.75
      control%inner_fraction_opt = point1
!     control%inner_stop_relative = zero
      control%inner_stop_relative = point01
      control%inner_stop_absolute = SQRT( epsmch )
      control%max_infeasibility_relative = epsmch ** 0.75
      control%max_infeasibility_absolute = epsmch ** 0.75

!  Logical parameters

      control%remove_dependencies = .TRUE.
      control%find_basis_by_transpose = .TRUE.
      control%space_critical = .FALSE.
      control%deallocate_error_fatal = .FALSE.

!  Character parameters

      control%prefix = '""                            '

!  Ensure that the private data arrays have the correct initial status

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control )
      CALL GLTR_initialize( data%GLTR_data, control%GLTR_control )

!  Reset GLTR and SBLS data for this package

      control%GLTR_control%stop_relative = control%inner_stop_relative
      control%GLTR_control%stop_absolute = control%inner_stop_absolute
      control%GLTR_control%fraction_opt = control%inner_fraction_opt
      control%GLTR_control%unitm = .FALSE.
      control%GLTR_control%error = control%error
      control%GLTR_control%out = control%out
      control%GLTR_control%print_level = control%print_level - 1
      control%GLTR_control%itmax = control%cg_maxit
      control%GLTR_control%boundary = .FALSE.
      control%GLTR_control%space_critical = control%space_critical
      control%GLTR_control%deallocate_error_fatal                              &
        = control%deallocate_error_fatal
      control%GLTR_control%prefix = '" - GLTR:"                     '

      control%SBLS_control%error = control%error
      control%SBLS_control%out = control%out
      control%SBLS_control%print_level = control%print_level - 1
      control%SBLS_control%indmin = control%indmin
      control%SBLS_control%valmin = control%valmin
      control%SBLS_control%len_glsmin = control%len_glsmin
      control%SBLS_control%itref_max = control%itref_max
      control%SBLS_control%factorization = control%factorization
      control%SBLS_control%preconditioner = control%preconditioner
      control%SBLS_control%new_a = control%new_a
      control%SBLS_control%max_col = control%max_col
      control%SBLS_control%pivot_tol = control%pivot_tol
      control%SBLS_control%pivot_tol_for_basis = control%pivot_tol_for_basis
      control%SBLS_control%zero_pivot = control%zero_pivot
      control%SBLS_control%remove_dependencies = control%remove_dependencies
      control%SBLS_control%find_basis_by_transpose =                           &
        control%find_basis_by_transpose
      control%SBLS_control%deallocate_error_fatal =                            &
        control%deallocate_error_fatal
      control%SBLS_control%prefix = '" - SBLS:"                     '

      data%new_h = .TRUE.
      data%new_c = .TRUE.
      RETURN  

!  End of LLS_initialize

      END SUBROUTINE LLS_initialize

!-*-*-*-*-   L L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LLS_initialize could (roughly) 
!  have been set as:

!  BEGIN LLS SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   initial-workspace-for-unsymmetric-solver        1000
!   initial-integer-workspace                       1000
!   initial-real-workspace                          1000
!   preconditioner-used                             0
!   factorization-used                              0
!   maximum-column-nonzeros-in-schur-complement     35
!   maximum-refinements                             1
!   maximum-number-of-cg-iterations                 200
!   truat-region-radius                             1.0D+19
!   pivot-tolerance-used                            1.0D-12
!   pivot-tolerance-used-for-basis                  0.5
!   zero-pivot-tolerance                            1.0D-12
!   inner-iteration-fraction-optimality-required    0.1
!   inner-iteration-relative-accuracy-required      0.01
!   inner-iteration-absolute-accuracy-required      1.0E-8
!   max-relative-infeasibility-allowed              1.0E-12
!   max-absolute-infeasibility-allowed              1.0E-12
!   find-basis-by-transpose                         T
!   remove-linear-dependencies                      T
!   space-critical                                  F
!   deallocate-error-fatal                          F
!   output-line-prefix                              ""
!  END LLS SPECIFICATIONS

!  Dummy arguments

      TYPE ( LLS_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 35
      CHARACTER( LEN = 16 ), PARAMETER :: specname = 'LLS             '
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level' 
      spec(  7 )%keyword = 'factorization-used'
      spec(  8 )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec(  4 )%keyword = 'initial-workspace-for-unsymmetric-solver'
      spec(  9 )%keyword = 'initial-integer-workspace'
      spec( 10 )%keyword = 'initial-real-workspace'
      spec( 11 )%keyword = 'maximum-refinements'
      spec( 13 )%keyword = 'maximum-number-of-cg-iterations'
      spec( 14 )%keyword = 'preconditioner-used'

!  Real key-words

      spec( 19 )%keyword = 'trust-region-radius'
      spec( 23 )%keyword = 'max-relative-infeasibility-allowed'
      spec( 24 )%keyword = 'max-absolute-infeasibility-allowed'
      spec( 26 )%keyword = 'pivot-tolerance-used'
      spec( 27 )%keyword = 'pivot-tolerance-used-for-basis'
      spec( 28 )%keyword = 'zero-pivot-tolerance'
      spec( 30 )%keyword = 'inner-iteration-fraction-optimality-required'
      spec( 31 )%keyword = 'inner-iteration-relative-accuracy-required'
      spec( 32 )%keyword = 'inner-iteration-absolute-accuracy-required'

!  Logical key-words

      spec( 33 )%keyword = 'find-basis-by-transpose'
      spec( 34 )%keyword = 'remove-linear-dependencies'
      spec( 16 )%keyword = 'space-critical'
      spec( 35 )%keyword = 'deallocate-error-fatal'

!  Character key-words

!     spec( 36 )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_integer( spec( 1 ), control%error,                  &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 2 ), control%out,                    &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 3 ), control%print_level,            &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 7 ), control%factorization,          &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 8 ), control%max_col,                &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 4 ), control%len_glsmin,             &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 9 ), control%indmin,                 &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 10 ), control%valmin,                &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 11 ), control%itref_max,             &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 13 ), control%cg_maxit,              &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 14 ), control%preconditioner,        &
                                    control%error )

!  Set real values

      CALL SPECFILE_assign_real( spec( 19 ), control%radius,                   &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 23 ),                                   &
                                 control%max_infeasibility_relative,           &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 24 ),                                   &
                                 control%max_infeasibility_absolute,           &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 26 ), control%pivot_tol,                &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 27 ),                                   &
                                 control%pivot_tol_for_basis,                  &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 28 ), control%zero_pivot,               &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 30 ), control%inner_fraction_opt,       &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 31 ), control%inner_stop_relative,      &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 32 ), control%inner_stop_absolute,      &
                                 control%error )

!  Set logical values

      CALL SPECFILE_assign_logical( spec( 33 ),                                &
                                    control%find_basis_by_transpose,           &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 34 ), control%remove_dependencies,   &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 16 ),                                &
                                    control%space_critical,                    &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 35 ),                                &
                                    control%deallocate_error_fatal,            &
                                    control%error )

!  Set charcter values

!     CALL SPECFILE_assign_string( spec( 36 ), control%prefix,                 &
!                                  control%error )

!  Reset GLTR and SBLS data for this package

      control%GLTR_control%stop_relative = control%inner_stop_relative
      control%GLTR_control%stop_absolute = control%inner_stop_absolute
      control%GLTR_control%error = control%error
      control%GLTR_control%out = control%out
      control%GLTR_control%print_level = control%print_level - 1
      control%GLTR_control%itmax = control%cg_maxit

      control%SBLS_control%error = control%error
      control%SBLS_control%out = control%out
      control%SBLS_control%print_level = control%print_level - 1
      control%SBLS_control%indmin = control%indmin
      control%SBLS_control%valmin = control%valmin
      control%SBLS_control%itref_max = control%itref_max
      control%SBLS_control%factorization = control%factorization
      control%SBLS_control%preconditioner = control%preconditioner
      control%SBLS_control%new_a = control%new_a
      control%SBLS_control%max_col = control%max_col
      control%SBLS_control%pivot_tol = control%pivot_tol
      control%SBLS_control%pivot_tol_for_basis = control%pivot_tol_for_basis
      control%SBLS_control%zero_pivot = control%zero_pivot
      control%SBLS_control%remove_dependencies = control%remove_dependencies
      control%SBLS_control%find_basis_by_transpose =                           &
        control%find_basis_by_transpose
      control%SBLS_control%space_critical = control%space_critical
      control%SBLS_control%deallocate_error_fatal =                            &
        control%deallocate_error_fatal

!  Read the controls for the preconditioner and iterative solver

      CALL SBLS_read_specfile( control%SBLS_control, device )
      CALL GLTR_read_specfile( control%GLTR_control, device )

      RETURN

      END SUBROUTINE LLS_read_specfile

!-*-*-*-*-*-*-*-*-   L L S _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE LLS_solve( prob, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 || A x + c ||_2^2
!
!     subject to    || x ||_2 <= Delta
!
!  where x is a vector of n components ( x_1, .... , x_n ), 
!  A is an m by n matrix, c is an m-vector and Delta is a constant, 
!  using a preconditioned conjugate-gradient method.
!  The subroutine is particularly appropriate when A is sparse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold 
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %new_problem_structure is a LOGICAL variable, which must be set to 
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to LLS_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!                 
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!                 
!   %A is a structure of type SMT_type used to hold the matrix A. 
!    Three storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row
!       A%col( : )   the column indices of the components of A
!       A%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 5 ) = TRANSFER( 'DENSE', A%type )
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of 
!                    increasing column indicies.
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output A%row will contain
!    the row numbers corresponding to the values in A%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).   
! 
!   %X is a REAL array of length %n, which must be set by the user
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
!
!   %C is a REAL array of length %m, which must be set by the user
!    to the values of the array c of constant terms in || Ax + c ||
!   
!  data is a structure of type LLS_data_type which holds private internal data
!
!  control is a structure of type LLS_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to LLS_initialize. See LLS_initialize 
!   for details
!
!  inform is a structure of type LLS_inform_type that provides 
!    information on exit from LLS_solve. The component status 
!    has possible values:
!  
!     0 Normal termination with a locally optimal solution.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!   - 3 one of the restrictions 
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -4 an error has occured in SILS_analyse; the status as returned by 
!       AINFO%FLAG is given in the component sils_analyse_status
!
!    -5 an error has occured in SILS_factorize; the status as returned by 
!       FINFO%FLAG is given in the component sils_factorize_status
!
!    -6 an error has occured in SILS_solve; the status as returned by 
!       SINFO%FLAG is given in the component sils_solve_status
!
!    -7 an error has occured in GLS_analyse; the status as returned by 
!       AINFO%FLAG is given in the component gls_analyse_status
!
!    -8 an error has occured in GLS_solve; the status as returned by 
!       SINFO%FLAG is given in the component gls_solve_status
!
!    -9 the computed precondition is insufficient. Try another
!
!   -11 the residuals are large; the factorization may be unsatisfactory
!
!  On exit from LLS_solve, other components of inform give the 
!  following:
!
!     alloc_status = the status of the last attempted allocation/deallocation 
!     bad_alloc = the name of the last array for which (de)allocation failed
!     cg_iter = the total number of conjugate gradient iterations required.
!     factorization_integer = the total integer workspace required for the 
!       factorization.
!     factorization_real = the total real workspace required for the 
!       factorization.
!     obj = the value of the objective function 1/2||Ax+c||^2 at the best 
!       estimate of the solution determined by LLS_solve.
!     time%total = the total time spent in the package.
!     time%factorize = the time spent factorizing the required matrices.
!     time%solve = the time spent computing the search direction.
!     SBLS_inform = inform components from SBLS
!     GLTR_inform = inform components from GLTR
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLS_control_type ), INTENT( INOUT ) :: control
      TYPE ( LLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j
      REAL :: time_end

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Set initial values for inform 

      inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%cg_iter = - 1
      inform%obj = zero
      inform%time%factorize = 0.0 ; inform%time%solve = 0.0
      CALL CPU_TIME( inform%time%total )

      IF ( prob%n <= 0 .OR. prob%m < 0 .OR.                                    &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = - 3
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
            "( ' ', /, A, ' **  Error return ', I0,' from LLS ' )" )           &
            prefix, inform%status 
        RETURN
      END IF 

!  If required, write out problem 

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', 2I8 )" ) prob%n, prob%m
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )          &
            prob%A%val( : prob%n * prob%m )
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( ' A (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, prob%A%col( j ), prob%A%val( j ),                           &
                j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( ' A (co-ordinate) = ' )" )
          WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
          ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne )
        END IF
        WRITE( control%out, "( ' C = ', /, ( 5ES12.4 ) )" )                    &
          prob%C( : prob%m )
      END IF

!  Call the solver

      CALL LLS_solve_main( prob%n, prob%m, prob%A, prob%C, prob%q, prob%X,     &
                           data, control, inform )

      CALL CPU_TIME( time_end )
      inform%time%total = time_end - inform%time%total
      RETURN

!  End of LLS_solve

      END SUBROUTINE LLS_solve

!-*-*-*-*-   L L S _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LLS_solve_main( n, m, A, C, q, X, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 || A x + c ||_2^2
!
!     subject to    || x ||_2 <= Delta
!
!  where x is a vector of n components ( x_1, .... , x_n ), 
!  A is an m by n matrix, c is an m-vector and Delta is a constant, 
!  using a preconditioned conjugate-gradient method.
!  The subroutine is particularly appropriate when A is sparse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( OUT ) :: q
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C
      TYPE ( SMT_type ), INTENT( IN ) :: A
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      TYPE ( LLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLS_control_type ), INTENT( INOUT ) :: control
      TYPE ( LLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: out, i, j, l
      LOGICAL :: printt, printw
      REAL :: time_end
      REAL ( KIND = wp ) :: radius
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  ===========================
!  Control the output printing
!  ===========================

      out = control%out

!  Single line of output per iteration
!  but with additional timings for various operations

      printt = out > 0 .AND. control%print_level >= 2 

!  As per printt, but with checking of residuals, etc, and also with an 
!  indication of where in the code we are

      printw = out > 0 .AND. control%print_level >= 4

      inform%GLTR_inform%status = 1 
      inform%GLTR_inform%negative_curvature = .TRUE.

      IF ( control%preconditioner /= 0 ) THEN

!  Set C appropriately

        IF ( data%new_c ) THEN
          data%new_c = .FALSE.
          control%SBLS_control%new_c = 2
          data%C%ne = 0

          array_name = 'lls: data%C%row'
          CALL SPACE_resize_array( data%C%ne, data%C%row, inform%status,       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN

          array_name = 'lls: data%C%col'
          CALL SPACE_resize_array( data%C%ne, data%C%col, inform%status,       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN

          array_name = 'lls: data%C%val'
          CALL SPACE_resize_array( data%C%ne, data%C%val, inform%status,       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN

          array_name = 'lls: data%C%type'
          CALL SPACE_dealloc_array( data%C%type, inform%status,                &
             inform%alloc_status, array_name = array_name, out = control%error )
          CALL SMT_put( data%C%type, 'COORDINATE', inform%alloc_status )
        ELSE
          control%SBLS_control%new_c = 0
        END IF

!  Set C appropriately

        IF ( data%new_h ) THEN
          data%new_c = .FALSE.
          control%SBLS_control%new_h = 2
          data%H%ne = n
          array_name = 'lls: data%H%val'
          CALL SPACE_resize_array( data%H%ne, data%H%val, inform%status,       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN
          data%H%val = one

          array_name = 'lls: data%H%type'
          CALL SPACE_dealloc_array( data%H%type, inform%status,                &
             inform%alloc_status, array_name = array_name, out = control%error )
          CALL SMT_put( data%H%type, 'DIAGONAL', inform%alloc_status )
        ELSE
          control%SBLS_control%new_h = 0
        END IF

!  ------------------------------------------
!   1. Form and factorize the preconditioner
!  ------------------------------------------

        CALL CPU_TIME( inform%time%factorize )

        control%SBLS_control%error = control%error
        control%SBLS_control%out = control%out
        control%SBLS_control%print_level = control%print_level - 1
        control%SBLS_control%indmin = control%indmin
        control%SBLS_control%valmin = control%valmin
        control%SBLS_control%len_glsmin = control%len_glsmin
        control%SBLS_control%itref_max = control%itref_max
        control%SBLS_control%factorization = control%factorization
        control%SBLS_control%preconditioner = control%preconditioner
        control%SBLS_control%new_a = control%new_a
        control%SBLS_control%max_col = control%max_col
        control%SBLS_control%pivot_tol = control%pivot_tol
        control%SBLS_control%pivot_tol_for_basis = control%pivot_tol_for_basis
        control%SBLS_control%zero_pivot = control%zero_pivot
        control%SBLS_control%remove_dependencies = control%remove_dependencies
        control%SBLS_control%find_basis_by_transpose =                         &
          control%find_basis_by_transpose
        control%SBLS_control%deallocate_error_fatal =                          &
          control%deallocate_error_fatal
        control%SBLS_control%prefix = '" - SBLS:"                     '

        CALL SBLS_form_and_factorize( n, m, data%H, A, data%C, data%SBLS_data, &
                                      control%SBLS_control, inform%SBLS_inform )

        CALL CPU_TIME( time_end )
        inform%time%factorize = time_end - inform%time%factorize

        IF ( inform%SBLS_inform%status < 0 ) THEN
          inform%status = inform%SBLS_inform%status
          RETURN
        END IF

        IF ( printt ) WRITE( out,                                              &
           "(  A, ' on exit from SBLS: status = ', I0, ', time = ', F0.2 )" )  &
             prefix, inform%SBLS_inform%status, inform%time%factorize

!  ---------------------
!   2. Solve the problem
!  ---------------------

!  Allocate workspace

        array_name = 'lls: data%VECTOR'
        CALL SPACE_resize_array( n + m, data%VECTOR, inform%status,            &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN

     ELSE

        array_name = 'lls: data%VECTOR'
        CALL SPACE_resize_array( n, data%VECTOR, inform%status,                &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN

      END IF

      array_name = 'lls: data%Ax'
      CALL SPACE_resize_array( m, data%Ax, inform%status,                      &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'lls: data%ATc'
      CALL SPACE_resize_array( n, data%ATc, inform%status,                     &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'lls: data%R'
      CALL SPACE_resize_array( n + m, data%R, inform%status,                   &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'lls: data%S'
      CALL SPACE_resize_array( n, data%S, inform%status,                       &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

!  ----------------------------------------------------------------
!   Use GLTR to minimize the objective
!  ----------------------------------------------------------------

      CALL CPU_TIME( inform%time%solve )

!  Compute the gradient A^T c

      data%ATc( : n ) = zero
      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' ) 
        l = 0
        DO i = 1, n
          data%ATc( i )                                                     &
            = data%ATc( i ) + DOT_PRODUCT( A%val( l + 1 : l + m ), C )
          l = l + m
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            j = A%col( l )
            data%ATc( j ) = data%ATc( j ) + A%val( l ) * C( A%col( i ) )
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          j = A%col( l )
          data%ATc( j ) = data%ATc( j ) + A%val( l ) * C( A%row( l ) )
        END DO
      END SELECT

!  Set initial data
     
      IF ( control%radius > zero ) THEN
        radius = control%radius
      ELSE
        radius = SQRT( point1 * HUGE( one ) )
      END IF 

      control%GLTR_control%f_0 = half * DOT_PRODUCT( C, C )
!     control%GLTR_control%f_0 = zero

      data%R( : n ) = data%ATc( : n )
      inform%GLTR_inform%status = 1
      inform%cg_iter = 0

      control%GLTR_control%stop_relative = control%inner_stop_relative
      control%GLTR_control%stop_absolute = control%inner_stop_absolute
      control%GLTR_control%fraction_opt = control%inner_fraction_opt
      control%GLTR_control%rminvr_zero = hundred * epsmch ** 2
      control%GLTR_control%unitm = control%preconditioner == 0
      control%GLTR_control%error = control%error
      control%GLTR_control%out = control%out
      control%GLTR_control%print_level = control%print_level - 1
      control%GLTR_control%itmax = control%cg_maxit
      control%GLTR_control%boundary = .FALSE.
      control%GLTR_control%space_critical = control%space_critical
      control%GLTR_control%deallocate_error_fatal                             &
        = control%deallocate_error_fatal
      control%GLTR_control%prefix = '" - GLTR:"                     '

      DO
        CALL GLTR_solve( n, radius, q, X( : n ), data%R( : n ),               &
                         data%VECTOR( : n ), data%GLTR_data,                  &
                         control%GLTR_control, inform%GLTR_inform )

!  Check for error returns

!       WRITE(6,"( ' case ', i3  )" ) inform%GLTR_inform%status
        SELECT CASE( inform%GLTR_inform%status )

!  Successful return

        CASE ( 0 )
          EXIT

!  Warnings

        CASE ( - 2, - 1 )
          IF ( printt ) WRITE( out, "( /,                                      &
         &  A, ' Warning return from GLTR, status = ', I6 )" ) prefix,         &
              inform%GLTR_inform%status
          EXIT
          
!  Allocation errors

           CASE ( - 6 )
             inform%status = - 1
             inform%alloc_status = inform%gltr_inform%alloc_status
             inform%bad_alloc = inform%gltr_inform%bad_alloc
             GO TO 900

!  Deallocation errors

           CASE ( - 7 )
             inform%status = - 2
             inform%alloc_status = inform%gltr_inform%alloc_status
             inform%bad_alloc = inform%gltr_inform%bad_alloc
             GO TO 900

!  Error return

        CASE DEFAULT
          IF ( printt ) WRITE( out, "( /,                                      &
         &  A, ' Error return from GLTR, status = ', I6 )" ) prefix,           &
              inform%GLTR_inform%status
          EXIT

!  Find the preconditioned gradient

        CASE ( 2, 6 )
          IF ( printw ) WRITE( out,                                            &
             "( A, ' ............... precondition  ............... ' )" ) prefix

!         control%SBLS_control%out = 6
!         control%SBLS_control%print_level = 2
          control%SBLS_control%affine = .TRUE.
          CALL SBLS_solve( n, m, A, data%C, data%SBLS_data,                    &
             control%SBLS_control, inform%SBLS_inform, data%VECTOR )

          IF ( inform%SBLS_inform%status < 0 ) THEN
            inform%status = inform%SBLS_inform%status
            GO TO 900
          END IF

!  Form the product of VECTOR with A^T A

        CASE ( 3, 7 )

          IF ( printw ) WRITE( out,                                            &
            "( A, ' ............ matrix-vector product ..........' )" ) prefix

          data%Ax( : m ) = zero
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' ) 
            l = 0
            DO i = 1, m
              data%Ax( i ) = data%Ax( i )                                      &
                + DOT_PRODUCT( A%val( l + 1 : l + n ), data%VECTOR )
              l = l + n
            END DO
            l = 0
            data%VECTOR = zero
            DO i = 1, n
              data%VECTOR( i ) = data%VECTOR( i )                              &
                + DOT_PRODUCT( A%val( l + 1 : l + m ), data%Ax )
              l = l + m
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                data%Ax( i ) = data%Ax( i )                                    &
                  + A%val( l ) * data%VECTOR( A%col( l ) )
              END DO
            END DO
            data%VECTOR = zero
            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l )
                data%VECTOR( j ) = data%VECTOR( j )                            &
                  + A%val( l ) * data%Ax( A%col( i ) )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = A%row( l )
              data%Ax( i ) = data%Ax( i )                                      &
                + A%val( l ) * data%VECTOR( A%col( l ) )
            END DO
            data%VECTOR = zero
            DO l = 1, A%ne
              j = A%col( l )
              data%VECTOR( j ) = data%VECTOR( j )                              &
                + A%val( l ) * data%Ax( A%row( l ) )
            END DO
          END SELECT

!  Reform the initial residual

        CASE ( 5 )
          
          IF ( printw ) WRITE( out,                                            &
            "( A, ' ................. restarting ................ ' )" ) prefix

          data%R( : n ) = data%ATc

        END SELECT

      END DO

      inform%obj = ABS( q )
      inform%cg_iter = inform%GLTR_inform%iter
      inform%norm_x = inform%GLTR_inform%mnormx
!     write(6,*) inform%GLTR_inform%iter_pass2

      IF ( printw ) THEN
        data%Ax( : m ) = C( : m )
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            data%Ax( i ) = data%Ax( i ) + DOT_PRODUCT( A%val( l + 1 : l + n ),X )
            l = l + n
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              data%Ax( i ) = data%Ax( i ) + A%val( l ) * X( A%col( l ) )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l )
            data%Ax( i ) = data%Ax( i ) + A%val( l ) * X( A%col( l ) )
          END DO
        END SELECT
        WRITE( out, "(  A, ' computed objective ', ES12.4 )" )                 &
          half * DOT_PRODUCT( data%Ax( : m ), data%Ax( : m ) )
      END IF

      CALL CPU_TIME( time_end )
      inform%time%solve = time_end - inform%time%solve

      IF ( printt ) WRITE( out,                                                &
         "(  A, ' on exit from GLTR: status = ', I0, ', CG iterations = ', I0, &
        &   ', time = ', F0.2 )" ) prefix,                                     &
            inform%GLTR_inform%status, inform%cg_iter, inform%time%solve

      IF ( printt ) THEN
        SELECT CASE( control%preconditioner )
        CASE( 0 )
          WRITE( out, "( A, ' No preconditioner' )" ) prefix
        CASE( 1 )
          WRITE( out, "( A, ' Preconditioner G = I' )" ) prefix
        CASE( - 1 )
          WRITE( out, "( A, ' Preconditioner G_22 = I' )" ) prefix
        END SELECT
      END IF

      RETURN
 
!  Error returns

  900 CONTINUE
      CALL CPU_TIME( time_end )
      inform%time%solve = time_end - inform%time%solve

      RETURN

!  End of LLS_solve_main

      END SUBROUTINE LLS_solve_main

!-*-*-*-*-*-*-   L L S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE LLS_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine LLS_initialize
!   control see Subroutine LLS_initialize
!   inform  see Subroutine LLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLS_control_type ), INTENT( IN ) :: control        
      TYPE ( LLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by SBLS and GLTR

      CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
      CALL GLTR_terminate( data%GLTR_data, control%GLTR_control,               &
                           inform%GLTR_inform )

!  Deallocate all remaining allocated arrays

      array_name = 'lls: data%C%row'
      CALL SPACE_dealloc_array( data%C%row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%C%col'
      CALL SPACE_dealloc_array( data%C%col,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%C%val'
      CALL SPACE_dealloc_array( data%C%val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%C%type'
      CALL SPACE_dealloc_array( data%C%type,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%G_f'
      CALL SPACE_dealloc_array( data%ATc,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%R'
      CALL SPACE_dealloc_array( data%R,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%VECTOR'
      CALL SPACE_dealloc_array( data%VECTOR,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%Ax'
      CALL SPACE_dealloc_array( data%Ax,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  End of subroutine LLS_terminate

      END SUBROUTINE LLS_terminate

!  End of module LLS

   END MODULE GALAHAD_LLS_double
