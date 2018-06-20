! THIS VERSION: GALAHAD 2.6 - 15/10/2014 AT 13:20 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ B Q P   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.4. January 1st 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BQP_double

!        ----------------------------------------------------------------
!        |                                                              |
!        | Solve the convex bound-constrained quadratic program         |
!        |                                                              |
!        |    minimize     1/2 x(T) H x + g(T) x + f                    |
!        |    subject to     x_l <=  x  <= x_u                          |
!        |                                                              |
!        | using a preconditioned projected conjugate-gradient approach |
!        |                                                              |
!        ----------------------------------------------------------------

     USE GALAHAD_SYMBOLS
     USE GALAHAD_STRING_double, ONLY: STRING_integer_6, STRING_real_7
     USE GALAHAD_SPACE_double
     USE GALAHAD_SORT_double, ONLY: SORT_heapsort_build, SORT_heapsort_smallest
     USE GALAHAD_SBLS_double
     USE GALAHAD_QPT_double
     USE GALAHAD_QPP_double
     USE GALAHAD_QPD_double, ONLY: QPD_SIF
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: BQP_initialize, BQP_read_specfile, BQP_solve, BQP_terminate,    &
               BQP_reverse_type, BQP_data_type, NLPT_userdata_type,            &
               QPT_problem_type, SMT_type, SMT_put, SMT_get,                   &
               BQP_arcsearch_data_type

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: g_zero = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: h_zero = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
     REAL ( KIND = wp ), PARAMETER :: t_max = infinity
     REAL ( KIND = wp ), PARAMETER :: alpha_search = one
     REAL ( KIND = wp ), PARAMETER :: beta_search = half
     REAL ( KIND = wp ), PARAMETER :: mu_search = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: fixed_tol = epsmch

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BQP_control_type

!  unit number for error and warning diagnostics

       INTEGER :: error = 6

!  general output unit number

       INTEGER :: out  = 6

!  the level of output required

       INTEGER :: print_level = 0

!  on which iteration to start printing

       INTEGER :: start_print = - 1

!  on which iteration to stop printing

       INTEGER :: stop_print = - 1

!  how many iterations between printing

       INTEGER :: print_gap = 1

!  how many iterations to perform (-ve reverts to HUGE(1)-1)

       INTEGER :: maxit = 1000

!  cold_start should be set to 0 if a warm start is required (with variables
!   assigned according to B_stat, see below), and to any other value if the
!   values given in prob%X suffice

       INTEGER :: cold_start = 1

!  the ratio of how many iterations use CG rather steepest descent

       INTEGER :: ratio_cg_vs_sd = 1

!  how many CG iterations to perform per BQP iteration (-ve reverts to n+1)

       INTEGER :: cg_maxit = 1000

!  the unit number to write generated SIF file describing the current problem

       INTEGER :: sif_file_device = 52

!  any bound larger than infinity in modulus will be regarded as infinite

       REAL ( KIND = wp ) :: infinity = ten ** 19

!  the required accuracy for the primal infeasibility

       REAL ( KIND = wp ) :: stop_p = ten ** ( - 6 )

!  the required accuracy for the dual infeasibility

       REAL ( KIND = wp ) :: stop_d = ten ** ( - 6 )

!  the required accuracy for the complementary slackness

       REAL ( KIND = wp ) :: stop_c = ten ** ( - 6 )

!  any pair of constraint bounds (x_l,x_u) that are closer than i
!   dentical_bounds_tol will be reset to the average of their values
!
       REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!  the CG iteration will be stopped as soon as the current norm of the
!  preconditioned gradient is smaller than
!    max( stop_cg_relative * initial preconditioned gradient, stop_cg_absolute )

       REAL ( KIND = wp ) :: stop_cg_relative = ten ** ( - 2 )
       REAL ( KIND = wp ) :: stop_cg_absolute = epsmch

!  threshold below which curvature is regarded as zero

       REAL ( KIND = wp ) :: zero_curvature = ten * epsmch

!  the maximum CPU time allowed (-ve = no limit)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!  exact_arcsearch is true if an exact arcsearch is required, and false if an
!   approximation suffices

       LOGICAL :: exact_arcsearch = .TRUE.

!  if space_critical is true, every effort will be made to use as little
!   space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!  if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!  if generate_sif_file is true, a SIF file describing the current problem
!  will be generated

       LOGICAL :: generate_sif_file = .FALSE.

!  name (max 30 characters) of generated SIF file containing input problem

       CHARACTER ( LEN = 30 ) :: sif_file_name =                               &
         "BQPPROB.SIF"  // REPEAT( ' ', 19 )

!  all output lines will be prefixed by a string (max 30 characters)
!    prefix(2:LEN(TRIM(%prefix))-1)
!   where prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'
!
       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SBLS

       TYPE ( SBLS_control_type ) :: SBLS_control
     END TYPE BQP_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BQP_time_type

!  total time

       REAL :: total = 0.0

!  time for the analysis phase

       REAL :: analyse = 0.0

!  time for the factorization phase

       REAL :: factorize = 0.0

!  time for the linear solution phase

       REAL :: solve = 0.0
     END TYPE BQP_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BQP_inform_type

!  reported return status:
!     0  success
!    -1  allocation error
!    -2  deallocation error
!    -3  matrix data faulty (%n < 1, %ne < 0)
!   -20  alegedly +ve definite matrix is not

       INTEGER :: status = 1

!  STAT value after allocate failure

       INTEGER :: alloc_status = 0

!  status return from factorization

       INTEGER :: factorization_status = 0

!  number of iterations required

       INTEGER :: iter = - 1

!  number of CG iterations required

       INTEGER :: cg_iter = 0

!  current value of the objective function

       REAL ( KIND = wp ) :: obj = infinity

!  current value of the projected gradient

       REAL ( KIND = wp ) :: norm_pg = infinity

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  times for various stages

       TYPE ( BQP_time_type ) :: time

!  inform values from SBLS

       TYPE ( SBLS_inform_type ) :: SBLS_inform
     END TYPE BQP_inform_type

!  - - - - - - - - - - - - - - -
!   arcsearch data derived type
!  - - - - - - - - - - - - - - -

     TYPE :: BQP_arcsearch_data_type
       INTEGER :: iterca, iter, itmax, n_freed, nbreak, nzero, branch
       INTEGER :: arcsearch_iter
       REAL ( KIND = wp ) :: tk, gxt, hxt, epstl2, tpttp, tcauch
       REAL ( KIND = wp ) :: tbreak, deltat, epsqrt, gxtold, g0tp
       REAL ( KIND = wp ) :: t, tamax , ptp, gtp, flxt, t_new
       LOGICAL :: prnter, pronel, recomp, explicit_h, use_hprod
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: USED
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BREAKP, GRAD
     END TYPE BQP_arcsearch_data_type

!  - - - - - - - - - - -
!   reverse derived type
!  - - - - - - - - - - -

     TYPE :: BQP_reverse_type
       INTEGER :: nz_v_start, nz_v_end, nz_prod_end
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_v, NZ_prod
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V, PROD
     END TYPE BQP_reverse_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE :: BQP_data_type
       INTEGER :: out, error, print_level, start_print, stop_print, print_gap
       INTEGER :: arcsearch_status, n_free, branch, cg_iter, change_status
       INTEGER :: nz_v_start, nz_v_end, nz_prod_end, maxit, cg_maxit
       REAL :: time_start
       REAL ( KIND = wp ) :: q_t, norm_step, step, stop_cg, old_gnrmsq, pnrmsq
       REAL ( KIND = wp ) :: curvature
       LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
       LOGICAL :: set_printm, printt, printi, printm, printw, printd, printe
       LOGICAL :: reverse, explicit_h, use_hprod, header
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: VARIABLE_status, OLD_status
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_v, NZ_prod
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_new, G, V, PROD
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_free, P_free
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_free, PG_free
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HP_free
       TYPE ( SMT_type ) :: H
       TYPE ( BQP_arcsearch_data_type ) :: arcsearch_data
       TYPE ( SBLS_data_type ) :: SBLS_data
     END TYPE BQP_data_type

   CONTAINS

!-*-*-*-*-*-   B Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE BQP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for BQP. This routine should be called before
!  BQP_solve
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

     TYPE ( BQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( BQP_control_type ), INTENT( OUT ) :: control
     TYPE ( BQP_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!  initialize control parameters for SBLS (see GALAHAD_SBLS for details)

     CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
     control%SBLS_control%prefix = '" - SBLS:"                    '

!  added here to prevent for compiler bugs

     control%stop_p = epsmch ** 0.33_wp
     control%stop_d = epsmch ** 0.33_wp
     control%stop_c = epsmch ** 0.33_wp
     control%stop_cg_absolute = SQRT( epsmch )

     RETURN

!  End of BQP_initialize

     END SUBROUTINE BQP_initialize

!-*-*-*-*-   B Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE BQP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by BQP_initialize could (roughly)
!  have been set as:

! BEGIN BQP SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  start-print                                       -1
!  stop-print                                        -1
!  iterations-between-printing                       1
!  maximum-number-of-iterations                      1000
!  cold-start                                        1
!  ratio-of-cg-iterations-to-steepest-descent        1
!  maximum-number-of-cg-iterations-per-iteration     1000
!  sif-file-device                                   52
!  infinity-value                                    1.0D+19
!  primal-accuracy-required                          1.0D-5
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  identical-bounds-tolerance                        1.0D-15
!  cg-relative-accuracy-required                     0.01
!  cg-absolute-accuracy-required                     1.0D-8
!  zero-curvature-threshold                          1.0D-15
!  maximum-cpu-time-limit                            -1.0
!  exact-arcsearch-used                              T
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  sif-file-name                                     BQPPROB.SIF
!  output-line-prefix                                ""
! END BQP SPECIFICATIONS

!  Dummy arguments

     TYPE ( BQP_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: start_print = print_level + 1
     INTEGER, PARAMETER :: stop_print = start_print + 1
     INTEGER, PARAMETER :: print_gap = stop_print + 1
     INTEGER, PARAMETER :: sif_file_device = print_gap + 1
     INTEGER, PARAMETER :: maxit = sif_file_device + 1
     INTEGER, PARAMETER :: cold_start = maxit + 1
     INTEGER, PARAMETER :: ratio_cg_vs_sd = cold_start + 1
     INTEGER, PARAMETER :: cg_maxit = ratio_cg_vs_sd + 1
     INTEGER, PARAMETER :: infinity = cg_maxit + 1
     INTEGER, PARAMETER :: stop_p = infinity + 1
     INTEGER, PARAMETER :: stop_d = stop_p + 1
     INTEGER, PARAMETER :: stop_c = stop_d + 1
     INTEGER, PARAMETER :: identical_bounds_tol = stop_c + 1
     INTEGER, PARAMETER :: stop_cg_relative = identical_bounds_tol + 1
     INTEGER, PARAMETER :: stop_cg_absolute = stop_cg_relative + 1
     INTEGER, PARAMETER :: zero_curvature = stop_cg_absolute + 1
     INTEGER, PARAMETER :: cpu_time_limit = zero_curvature + 1
     INTEGER, PARAMETER :: exact_arcsearch = cpu_time_limit + 1
     INTEGER, PARAMETER :: space_critical = exact_arcsearch + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: sif_file_name = generate_sif_file + 1
     INTEGER, PARAMETER :: prefix = sif_file_name + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 3 ), PARAMETER :: specname = 'BQP'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( cold_start )%keyword = 'cold-start'
     spec( ratio_cg_vs_sd )%keyword =                                          &
       'ratio-of-cg-iterations-to-steepest-descent'
     spec( cg_maxit )%keyword = 'maximum-number-of-cg-iterations-per-iteration'
     spec( sif_file_device )%keyword = 'sif-file-device'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( stop_p )%keyword = 'primal-accuracy-required'
     spec( stop_d )%keyword = 'dual-accuracy-required'
     spec( stop_c )%keyword = 'complementary-slackness-accuracy-required'
     spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
     spec( stop_cg_relative )%keyword = 'cg-relative-accuracy-required'
     spec( stop_cg_absolute )%keyword = 'cg-absolute-accuracy-required'
     spec( zero_curvature )%keyword = 'zero-curvature-threshold'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'

!  Logical key-words

     spec( exact_arcsearch )%keyword = 'exact-arcsearch-used'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
     spec( generate_sif_file )%keyword = 'generate-sif-file'

!  Character key-words

     spec( sif_file_name )%keyword = 'sif-file-name'
     spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

    IF ( PRESENT( alt_specname ) ) THEN
      CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
    ELSE
      CALL SPECFILE_read( device, specname, spec, lspec, control%error )
    END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( start_print ),                          &
                                 control%start_print,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_print ),                           &
                                 control%stop_print,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_gap ),                            &
                                 control%print_gap,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cold_start ),                           &
                                 control%cold_start,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( ratio_cg_vs_sd ),                       &
                                 control%ratio_cg_vs_sd,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cg_maxit ),                             &
                                 control%cg_maxit,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )

!  Set real value

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_p ),                               &
                                 control%stop_p,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_d ),                               &
                                 control%stop_d,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_c ),                               &
                                 control%stop_c,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                 &
                                 control%identical_bounds_tol,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_cg_relative ),                     &
                                 control%stop_cg_relative,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_cg_absolute ),                     &
                                 control%stop_cg_absolute,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( zero_curvature ),                       &
                                 control%zero_curvature,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( exact_arcsearch ),                      &
                                 control%exact_arcsearch,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_sif_file ),                    &
                                 control%generate_sif_file,                    &
                                 control%error )

!  Set character value

     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Read the specfiles for SBLS

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SBLS_read_specfile( control%SBLS_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-SBLS' )
     ELSE
       CALL SBLS_read_specfile( control%SBLS_control, device )
     END IF

     RETURN

     END SUBROUTINE BQP_read_specfile

!-*-*-*-*-*-*-*-*-   B Q P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE BQP_solve( prob, B_stat, data, control, inform,               &
                           userdata, reverse, eval_HPROD )
!                          userdata, reverse, eval_HPROD, eval_PREC )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to   (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), const is a
!  constant, g is an n-vector, H is a symmetric, positive definite matrix,
!  and any of the bounds (x_l)_i, (x_u)_i may be infinite, using a
!  preconditioned projected CG method.
!
!  The subroutine is particularly appropriate when H is sparse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H. Four storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       H%type( 1 : 10 ) = TRANSFER( 'COORDINATE', H%type )
!       H%val( : )   the values of the components of H
!       H%row( : )   the row indices of the components of H
!       H%col( : )   the column indices of the components of H
!       H%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of H
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', H%type )
!       H%val( : )   the values of the components of H, stored row by row
!       H%col( : )   the column indices of the components of H
!       H%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = TRANSFER( 'DENSE', H%type )
!       H%val( : )   the values of the components of H, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', H%type )
!       H%val( : )   the values of the diagonals of H, stored in order
!
!    If H is not available explicitly, matrix-vector products must be
!      provided by the user using either reverse communication
!      (see reverse below) or a provided subroutine (see eval_HPROD below).
!
!   %G is a REAL array of length %n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ...., n,
!    should contain the value of g_i.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
!
!   %X_l, %X_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %X_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %X_u to a value larger than
!    control%infinity. On exit, %X_l and %X_u will most likely have been
!    reordered.
!
!   %Z is a REAL array of length %n, which must be set by the user to
!    appropriate estimates of the values of the dual variables,
!    i.e., Lagrange multipliers corresponding to the simple bound
!    constraints x_l <= x <= x_u. On successful exit, it will contain
!   the required vector of dual variables.
!
!  B_stat is a INTEGER array of length n, which may be set by the user
!   on entry to BQP_solve to indicate which of the simple bound constraints
!   are to be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; B_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   B_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   B_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  data is a structure of type BQP_data_type which holds private internal data
!
!  control is a structure of type BQP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to BQP_initialize. See BQP_initialize
!   for details
!
!  inform is a structure of type BQP_inform_type that provides
!    information on exit from BQP_solve. The component %status
!    must be set to 1 on initial entry, and on exit has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!     2 The product H * v of the Hessian H with a given output vector v
!       is required from the user. The vector v will be stored in reverse%V
!       and the product H * v must be returned in reverse%PROD, and
!       BQP_solve re-entered with all other arguments unchanged.
!
!     3 The product H * v of the Hessian H with a given output vector v
!       is required from the user. Only components
!         reverse%NZ_v( reverse%nz_v_start : reverse%nz_v_end )
!       of the vector v stored in reverse%V are nonzero. The resulting
!       product H * v must be placed in reverse%PROD, and BQP_solve re-entered
!       with all other arguments unchanged.
!
!     4 The product H * v of the Hessian H with a given output vector v
!       is required from the user. Only components
!         reverse%NZ_v( reverse%nz_v_start : reverse%nz_v_end )
!       of the vector v stored in reverse%V are nonzero. The resulting
!       NONZEROS in the product H * v must be placed in their appropriate
!       comnpinents of reverse%PROD, while a list of indices of the nonzeos
!       placed in reverse%NZ_prod( 1 : reverse%nz_prod_end ). BQP_solve should
!       then be re-entered with all other arguments unchanged. Typically
!       v will be very sparse (i.e., reverse%nz_p_end-reverse%nz_p_start
!       will be small).
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!   - 3 one of the restrictions
!        prob%n     >=  1
!        prob%m     >=  0
!        prob%H%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE',
!                         'DIAGONAL', 'REVERSE' }
!       has been violated.
!
!    -4 The bound constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -9 The factorization failed; the return status from the factorization
!       package is given in the component factorization_status.
!
!    -13 The problem is so ill-conditoned that further progress is impossible.
!
!    -16 The step is too small to make further impact.
!
!    -17 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!    -18 Too much CPU time has passed. This may happen if
!        control%cpu_time_limit is too small, but may also be
!        symptomatic of a badly scaled problem.
!
!    -23 an entry from the strict upper triangle of H has been input.
!
!  On exit from BQP_solve, other components of inform give the
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation
!     factorization_integer = The total integer workspace required for the
!       factorization.
!     factorization_real = The total real workspace required for the
!       factorization.
!     nfacts = The total number of factorizations performed.
!     nmods = The total number of factorizations which were modified to
!       ensure that the matrix was an appropriate preconditioner.
!     factorization_status = the return status from the matrix factorization
!       package.
!     obj = the value of the objective function at the best estimate of the
!       solution determined by BQP_solve.
!     non_negligible_pivot = the smallest pivot which was not judged to be
!       zero when detecting linearly dependent constraints
!     bad_alloc = the name of the array for which an allocation/deallocation
!       error ocurred
!     time%total = the total time spent in the package.
!     time%analyse = the time spent analysing the required matrices prior to
!       factorization.
!     time%factorize = the time spent factorizing the required matrices.
!     time%solve = the time spent computing the search direction.
!
!  userdata is a scalar variable of type NLPT_userdata_type which may be used
!   to pass user data to and from the eval_* subroutines (see below)
!   Available coomponents which may be allocated as required are:
!
!    integer is a rank-one allocatable array of type default integer.
!    real is a rank-one allocatable array of type default real
!    complex is a rank-one allocatable array of type default comple.
!    character is a rank-one allocatable array of type default character.
!    logical is a rank-one allocatable array of type default logical.
!    integer_pointer is a rank-one pointer array of type default integer.
!    real_pointer is a rank-one pointer array of type default  real
!    complex_pointer is a rank-one pointer array of type default complex.
!    character_pointer is a rank-one pointer array of type default character.
!    logical_pointer is a rank-one pointer array of type default logical.
!
!  reverse is an OPTIONAL structure of type BQP_reverse_type which is used to
!   pass intermediate data to and from BQP_solve. This will only be necessary
!   if reverse-communication is to be used to form matrix-vector products
!   of the form H * v or preconditioning steps of the form P * v. If
!   reverse is present (and eval_HPROD is absent), reverse communication
!   will be used and the user must monitor the value of inform%status
!   (see above) to await instructions about required matrix-vector products.
!
!  eval_HPROD is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product H * v of the given
!   matrix H and vector v stored in V must be returned in PROD; only the
!   components NZ_v( nz_v_start : nz_v_end ) of V are nonzero. If either of
!   the optional argeuments NZ_prod or nz_prod_end are absent, the WHOLE of H v
!   including zeros should be returned in PROD. If NZ_prod and nz_prod_end are
!   present, the NONZEROS in the product H * v must be placed in their
!   appropriate comnponents of reverse%PROD, while a list of indices of the
!   nonzeos placed in NZ_prod( 1 : nz_prod_end ). In both cases, the status
!   variable should be set to 0 unless the product is impossible in which
!   case status should be set to a nonzero value. If eval_HPROD is not
!   present, BQP_solve will either return to the user each time an evaluation
!   is required (see reverse above) or form the product directly from
!   user-provided %H.
!
!  eval_PREC is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product P * v of the given
!   preconditioner P and vector v stored in V must be returned in PV.
!   The status variable should be set to 0 unless the product is impossible
!   in which case status should be set to a nonzero value. If eval_PREC
!   is not present, BQP_solve will return to the user each time an evaluation
!   is required.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     INTEGER, INTENT( INOUT ), DIMENSION( prob%n ) :: B_stat
     TYPE ( BQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( BQP_control_type ), INTENT( IN ) :: control
     TYPE ( BQP_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     TYPE ( BQP_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse
     OPTIONAL :: eval_HPROD
!    OPTIONAL :: eval_HPROD, eval_PREC

!  interface blocks

     INTERFACE
       SUBROUTINE eval_HPROD( status, userdata, V, PROD, NZ_v, nz_v_start,     &
                              nz_v_end, NZ_prod, nz_prod_end )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: PROD
       INTEGER, OPTIONAL, INTENT( IN ) :: nz_v_start, nz_v_end
       INTEGER, OPTIONAL, INTENT( INOUT ) :: nz_prod_end
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_v
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_prod
       END SUBROUTINE eval_HPROD
     END INTERFACE

!    INTERFACE
!      SUBROUTINE eval_PREC( status, userdata, V, PV )
!      USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
!      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
!      INTEGER, INTENT( OUT ) :: status
!      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
!      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
!      REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: PV
!      END SUBROUTINE eval_PREC
!    END INTERFACE

!  Local variables

     INTEGER :: i, j, k, l, nnz
     REAL :: time
     REAL ( KIND = wp ) :: val, av_bnd, x_i, p_i, curvature
     REAL ( KIND = wp ) :: gnrmsq, beta
     LOGICAL :: reset_bnd
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Enter or re-enter the package and jump to appropriate re-entry point

     IF ( inform%status == 1 ) data%branch = 100
     IF ( inform%status == 10 ) data%branch = 150

     SELECT CASE ( data%branch )
     CASE ( 100 ) ; GO TO 100
     CASE ( 150 ) ; GO TO 150
     CASE ( 200 ) ; GO TO 200
     CASE ( 300 ) ; GO TO 300
     CASE ( 400 ) ; GO TO 400
     CASE ( 500 ) ; GO TO 500
     END SELECT

 100 CONTINUE

     IF ( control%out > 0 .AND. control%print_level >= 5 )                     &
       WRITE( control%out, 2000 ) prefix, ' entering '

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

     IF ( control%generate_sif_file ) THEN
       CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,     &
                     control%infinity, .TRUE., no_linear = .TRUE. )
     END IF

!  SIF file generated
! -------------------------------------------------------------------

!  Initialize time

     CALL CPU_TIME( data%time_start )

!  Set initial timing breakdowns

     inform%time%total = 0.0 ; inform%time%analyse = 0.0
     inform%time%factorize = 0.0 ; inform%time%solve = 0.0

     data%use_hprod = PRESENT( eval_HPROD )
     data%reverse = PRESENT( reverse ) .AND. .NOT. data%use_hprod
     data%explicit_h = .NOT. ( data%use_hprod .OR. data%reverse )

     IF ( control%maxit < 0 ) THEN
       data%maxit = HUGE( 1 ) - 1
     ELSE
       data%maxit = control%maxit
     END IF

     IF ( control%cg_maxit < 0 ) THEN
       data%cg_maxit = prob%n + 1
     ELSE
       data%cg_maxit = control%cg_maxit
     END IF

!  ===========================
!  Control the output printing
!  ===========================

     data%out = control%out ; data%error = control%error
     data%print_level = 0
     IF ( control%start_print <= 0 ) THEN
       data%start_print = 0
     ELSE
       data%start_print = control%start_print
     END IF

     IF ( control%stop_print < 0 ) THEN
       data%stop_print = data%maxit + 1
     ELSE
       data%stop_print = control%stop_print
     END IF

     IF ( control%print_gap < 2 ) THEN
       data%print_gap = 1
     ELSE
       data%print_gap = control%print_gap
     END IF

!  error output

     data%set_printe = data%error > 0 .AND. control%print_level >= 1

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. control%print_level >= 2

!  as per printt, but with checking of residuals, etc

     data%set_printm = data%out > 0 .AND. control%print_level >= 3

!  as per printm but also with an indication of where in the code we are

     data%set_printw = data%out > 0 .AND. control%print_level >= 4

!  full debugging printing with significant arrays printed

     data%set_printd = data%out > 0 .AND. control%print_level >= 5

!  start setting control parameters

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
       data%print_level = control%print_level
       data%printe = data%set_printe ; data%printi = data%set_printi
       data%printt = data%set_printt ; data%printm = data%set_printm
       data%printw = data%set_printw ; data%printd = data%set_printd
     ELSE
       data%print_level = 0
       data%printe = .FALSE. ; data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
     END IF

!  Ensure that input parameters are within allowed ranges

     IF ( prob%n <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 910
     ELSE IF ( data%explicit_h ) THEN
       IF ( .NOT. QPT_keyword_H( prob%H%type ) ) THEN
         inform%status = GALAHAD_error_restrictions
         GO TO 910
       END IF
     END IF

!  If required, write out problem

     IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
       WRITE( control%out, "( ' n, m = ', I0, 1X, I0 )" ) prob%n, prob%m
       WRITE( control%out, "( ' f = ', ES12.4 )" ) prob%f
       WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
       IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
         WRITE( control%out, "( ' H (diagonal) = ', /, ( 5ES12.4 ) )" )        &
           prob%H%val( : prob%n )
       ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
         WRITE( control%out, "( ' H (dense) = ', /, ( 5ES12.4 ) )" )           &
           prob%H%val( : prob%n * ( prob%n + 1 ) / 2 )
       ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
         WRITE( control%out, "( ' H (row-wise) = ' )" )
         DO i = 1, prob%m
           WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                    &
             ( i, prob%H%col( j ), prob%H%val( j ),                            &
               j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1 )
         END DO
       ELSE
         WRITE( control%out, "( ' H (co-ordinate) = ' )" )
         WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                      &
        ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, prob%H%ne  )
       END IF
       WRITE( control%out, "( ' X_l = ', /, ( 5ES12.4 ) )" )                   &
         prob%X_l( : prob%n )
       WRITE( control%out, "( ' X_u = ', /, ( 5ES12.4 ) )" )                   &
         prob%X_u( : prob%n )
     END IF

!  Check that problem bounds are consistent; reassign any pair of bounds
!  that are "essentially" the same

     reset_bnd = .FALSE.
     DO i = 1, prob%n
       IF ( prob%X_l( i ) - prob%X_u( i ) > control%identical_bounds_tol ) THEN
         inform%status = GALAHAD_error_bad_bounds
         GO TO 910
       ELSE IF ( prob%X_u( i ) == prob%X_l( i ) ) THEN
       ELSE IF ( prob%X_u( i ) - prob%X_l( i )                                 &
                 <= control%identical_bounds_tol ) THEN
         av_bnd = half * ( prob%X_l( i ) + prob%X_u( i ) )
         prob%X_l( i ) = av_bnd ; prob%X_u( i ) = av_bnd
         reset_bnd = .TRUE.
       ELSE IF ( control%cold_start == 0 ) THEN
         IF ( B_stat( i ) < 0 ) THEN
            prob%X_l( i ) =  prob%X_l( i )
           reset_bnd = .TRUE.
         ELSE IF ( B_stat( i ) > 0 ) THEN
            prob%X_l( i ) =  prob%X_u( i )
           reset_bnd = .TRUE.
         END IF
       END IF
     END DO
     IF ( reset_bnd .AND. data%printi ) WRITE( control%out,                    &
       "( ' ', /, A, '   **  Warning: one or more variable bounds reset ' )" ) &
         prefix

!  allocate workspace arrays

     array_name = 'bqp: data%G'
     CALL SPACE_resize_array( prob%n, data%G, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqp: data%S_free'
     CALL SPACE_resize_array( prob%n, data%S_free, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqp: data%G_free'
     CALL SPACE_resize_array( prob%n, data%G_free, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqp: data%PG_free'
     CALL SPACE_resize_array( prob%n, data%PG_free, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqp: data%P_free'
     CALL SPACE_resize_array( prob%n, data%P_free, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqp: data%HP_free'
     CALL SPACE_resize_array( prob%n, data%HP_free, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqp: data%X_new'
     CALL SPACE_resize_array( prob%n, data%X_new, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqp: data%VARIABLE_status'
     CALL SPACE_resize_array( prob%n, data%VARIABLE_status, inform%status,     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqp: data%OLD_status'
     CALL SPACE_resize_array( prob%n, data%OLD_status, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     IF ( data%reverse ) THEN
       array_name = 'bqp: reverse%NZ_v'
       CALL SPACE_resize_array( prob%n, reverse%NZ_v, inform%status,           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqp: reverse%NZ_prod'
       CALL SPACE_resize_array( prob%n, reverse%NZ_prod, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqp: reverse%V'
       CALL SPACE_resize_array( prob%n, reverse%V, inform%status,              &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqp: reverse%PROD'
       CALL SPACE_resize_array( prob%n, reverse%PROD, inform%status,           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     ELSE
       array_name = 'bqp: data%NZ_v'
       CALL SPACE_resize_array( prob%n, data%NZ_v, inform%status,              &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqp: data%NZ_prod'
       CALL SPACE_resize_array( prob%n, data%NZ_prod, inform%status,           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqp: data%V'
       CALL SPACE_resize_array( prob%n, data%V, inform%status,                 &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqp: data%PROD'
       CALL SPACE_resize_array( prob%n, data%PROD, inform%status,              &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

     array_name = 'bqp: data%arcsearch_data%BREAKP'
     CALL SPACE_resize_array( prob%n, data%arcsearch_data%BREAKP,              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     IF ( .NOT. control%exact_arcsearch ) THEN
       array_name = 'bqp: data%arcsearch_data%GRAD'
       CALL SPACE_resize_array( prob%n, data%arcsearch_data%GRAD,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

     IF ( data%explicit_h ) THEN
       array_name = 'bqp: data%arcsearch_data%USED'
       CALL SPACE_resize_array( prob%n, data%arcsearch_data%USED,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

!  Build a copy of H stored by rows (both lower and upper triangles)

 150 CONTINUE
     data%header = .TRUE.
     IF ( data%explicit_h ) THEN

!  allocate space to record row lengths

       array_name = 'bqp: data%H%ptr'
       CALL SPACE_resize_array( prob%n + 1, data%H%ptr, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  compute the number of nonzeros in each row

       SELECT CASE ( SMT_get( prob%H%type ) )
       CASE ( 'COORDINATE' )
         data%H%ptr = 0
         DO l = 1, prob%H%ne
           i = prob%H%row( l ) ; j = prob%H%col( l )
           data%H%ptr( i ) = data%H%ptr( i ) + 1
           IF ( i /= j ) data%H%ptr( j ) = data%H%ptr( j ) + 1
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         data%H%ptr = 0
         DO i = 1, prob%n
           DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
             j = prob%H%col( l )
             data%H%ptr( i ) = data%H%ptr( i ) + 1
             IF ( i /= j ) data%H%ptr( j ) = data%H%ptr( j ) + 1
           END DO
         END DO
       CASE ( 'DENSE' )
         data%H%ptr = prob%n
       CASE ( 'DIAGONAL' )
         data%H%ptr = 1
       END SELECT

!  set starting addresses for each row in the row-wise scheme

       nnz = 1
       DO i = 1, prob%n
         l = data%H%ptr( i )
         data%H%ptr( i ) = nnz
         nnz = nnz + l
       END DO
       data%H%ptr( prob%n + 1 ) = nnz

!  allocate space to hold the column indices and values in the row-wise scheme

       array_name = 'bqp: data%H%col'
       CALL SPACE_resize_array( nnz, data%H%col, inform%status,                &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqp: data%H%val'
       CALL SPACE_resize_array( nnz, data%H%val, inform%status,                &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  now copy H into the row-wise scheme

       SELECT CASE ( SMT_get( prob%H%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, prob%H%ne
           i = prob%H%row( l ) ; j = prob%H%col( l ) ; val = prob%H%val( l )
           data%H%col( data%H%ptr( i ) ) = j
           data%H%val( data%H%ptr( i ) ) = val
           data%H%ptr( i ) = data%H%ptr( i ) + 1
           IF ( i /= j ) THEN
             data%H%col( data%H%ptr( j ) ) = i
             data%H%val( data%H%ptr( j ) ) = val
             data%H%ptr( j ) = data%H%ptr( j ) + 1
           END IF
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, prob%n
           DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
             j = prob%H%col( l ) ; val = prob%H%val( l )
             data%H%col( data%H%ptr( i ) ) = j
             data%H%val( data%H%ptr( i ) ) = val
             data%H%ptr( i ) = data%H%ptr( i ) + 1
             IF ( i /= j ) THEN
               data%H%col( data%H%ptr( j ) ) = i
               data%H%val( data%H%ptr( j ) ) = val
               data%H%ptr( j ) = data%H%ptr( j ) + 1
             END IF
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, prob%n
           DO j = 1, i
             l = l + 1
             val = prob%H%val( l )
             data%H%col( data%H%ptr( i ) ) = j
             data%H%val( data%H%ptr( i ) ) = val
             data%H%ptr( i ) = data%H%ptr( i ) + 1
             IF ( i /= j ) THEN
               data%H%col( data%H%ptr( j ) ) = i
               data%H%val( data%H%ptr( j ) ) = val
               data%H%ptr( j ) = data%H%ptr( j ) + 1
             END IF
           END DO
         END DO
       CASE ( 'DIAGONAL' )
         DO i = 1, prob%n
           data%H%col( data%H%ptr( i ) ) = i
           data%H%val( data%H%ptr( i ) ) =  prob%H%val( i )
           data%H%ptr( i ) = data%H%ptr( i ) + 1
         END DO
       END SELECT

!  finally, reset the starting addresses for each row

       DO i = prob%n, 1, - 1
         data%H%ptr( i + 1 ) = data%H%ptr( i )
       END DO
       data%H%ptr( 1 ) = 1
     END IF

     data%change_status = prob%n

!  ------------------------
!  Start the main iteration
!  ------------------------

 110 CONTINUE
       CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start

!  set the print levels for the iteration

       inform%iter = inform%iter + 1
       IF ( ( inform%iter >= data%start_print .AND.                            &
              inform%iter < data%stop_print ) .AND.                            &
            MOD( inform%iter - data%start_print, data%print_gap ) == 0 ) THEN
         data%print_level = control%print_level
         data%printe = data%set_printe ; data%printi = data%set_printi
         data%printt = data%set_printt ; data%printm = data%set_printm
         data%printw = data%set_printw ; data%printd = data%set_printd
       ELSE
         data%print_level = 0
         data%printe = .FALSE. ; data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
       END IF

!  obtain the matrix-vector product H * x

       IF ( data%reverse ) THEN
         reverse%V = prob%X
         data%branch = 200 ; inform%status = 2 ; RETURN
       ELSE IF ( data%use_hprod ) THEN
         CALL eval_HPROD( i, userdata, prob%X, data%PROD )
       END IF

!  compute the gradient

 200   CONTINUE
       IF ( data%explicit_h ) THEN
         data%G = prob%G
         DO i = 1, prob%n
           x_i = prob%X( i )
           DO k = data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
             data%G( data%H%col( k ) )                                         &
               = data%G( data%H%col( k ) ) + data%H%val( k ) * x_i
           END DO
         END DO
       ELSE IF ( data%use_hprod ) THEN
         data%G( : prob%n ) = prob%G( : prob%n ) + data%PROD( : prob%n )
       ELSE
         data%G( : prob%n ) = prob%G( : prob%n ) + reverse%PROD( : prob%n )
       END IF
       prob%Z( : prob%n ) = data%G( : prob%n )

!  compute the objective function

       inform%obj = half * ( DOT_PRODUCT( data%G, prob%X ) +                   &
                             DOT_PRODUCT( prob%G, prob%X ) ) + prob%f

!  compute the norm of the projected gradient

       inform%norm_pg = MAXVAL( ABS( MAX( MIN( prob%X - data%G, prob%X_u ),    &
                                          prob%X_l ) - prob%X ) )

!  print details of the current iteration

       IF ( ( data%printi .AND. data%header ) .OR. data%printt )               &
         WRITE( data%out, "( /, A, '  #its   #cg            f          ',      &
        &       ' proj gr     step   #free chnge   time' )" ) prefix
       data%header = .FALSE.
       IF ( data%printi ) THEN
         IF ( inform%iter > 1 ) THEN
           WRITE( data%out, "( A, 2A6, ES22.14, 2ES10.3, 2A6, A7 )" )          &
             prefix, STRING_integer_6( inform%iter ),                          &
             STRING_integer_6( data%cg_iter ),                                 &
             inform%obj, inform%norm_pg, data%norm_step,                       &
             STRING_integer_6( data%n_free ),                                  &
             STRING_integer_6( data%change_status ),                           &
             STRING_real_7( inform%time%total )
         ELSE IF ( inform%iter == 1 ) THEN
           WRITE( data%out, "( A, 2A6, ES22.14, 2ES10.3, A6, 5X, '-',  A7 )" ) &
             prefix, STRING_integer_6( inform%iter ),                          &
             STRING_integer_6( data%cg_iter ),                                 &
             inform%obj, inform%norm_pg, data%norm_step,                       &
             STRING_integer_6( data%n_free ),                                  &
             STRING_real_7( inform%time%total )
         ELSE
           WRITE( data%out, "( A, A6, '     -', ES22.14, ES10.3,               &
          & '      -        -     -', A7 )" )                                  &
             prefix, STRING_integer_6( inform%iter ),                          &
             inform%obj, inform%norm_pg,                                       &
             STRING_real_7( inform%time%total )
         END IF
       END IF

!  test for an approximate first-order critical point

       IF ( inform%norm_pg <= control%stop_d ) THEN
         inform%status = GALAHAD_ok ; GO TO 910
       END IF

!  test to see if more than maxit iterations have been performed

       IF ( inform%iter > data%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  check that the CPU time limit has not been reached

       IF ( control%cpu_time_limit >= zero .AND.                               &
            inform%time%total > control%cpu_time_limit ) THEN
         inform%status = GALAHAD_error_cpu_limit ; GO TO 900
       END IF

!  ----------------------------------------------------------------------------
!                  compute the search direction
!  ----------------------------------------------------------------------------

       data%cg_iter = 0

!  compute the search direction as the steepest-descent direction

       IF ( MOD( inform%iter, control%ratio_cg_vs_sd + 1 ) == 0                &
            .AND. data%change_status /= 0 ) THEN
         IF ( data% reverse ) THEN
           reverse%V = - data%G
         ELSE
           data%V = - data%G
         END IF

!  initialize the status of the variables

         DO i = 1, prob%n
           IF ( prob%X_l( i ) == prob%X_u( i ) ) THEN
             data%VARIABLE_status( i ) = 3
           ELSE
             data%VARIABLE_status( i ) = 0
           END IF
         END DO
         GO TO 490
       END IF

!  compute the search direction as the CG direction

!      ELSE

!  perform CG from x in the space of free variables: variables
!  reverse%nz_v(:data%n_free) or data%nz_v(:data%n_free) are free

!  compute g_free, the gradient in the free variables

         IF ( data% reverse ) THEN
           data%G_free( : data%n_free ) = data%G( reverse%nz_v( : data%n_free) )
         ELSE
           data%G_free( : data%n_free ) = data%G( data%nz_v( : data%n_free ) )
         END IF

!  start from s_free = 0

         data%S_free( : data%n_free ) = zero
         data%pnrmsq = zero
         data%q_t = inform%obj

!  - - - - - - - - - -
!  Start the CG loop
!  - - - - - - - - - -

 210     CONTINUE

!  obtain the preconditioned gradient pg_free = P(inverse) g_free

           data%PG_free( : data%n_free ) = data%G_free( : data%n_free )
 300       CONTINUE
           gnrmsq =  DOT_PRODUCT( data%PG_free( : data%n_free ),               &
                                  data%G_free( : data%n_free ) )

!  compute the CG stopping tolerance

           IF (  data%cg_iter == 0 ) THEN
             IF ( data%change_status == 0 ) THEN
               data%stop_cg =                                                  &
                 MAX( SQRT( ABS( gnrmsq ) ) * SQRT( epsmch ),                  &
                                              control%stop_cg_absolute )
             ELSE
               data%stop_cg =                                                  &
                 MAX( SQRT( ABS( gnrmsq ) ) * control%stop_cg_relative,        &
                                              control%stop_cg_absolute )
             END IF
           END IF

!  print details of the current iteration

           IF ( data%printm ) THEN
             IF ( data%cg_iter == 0 ) THEN
               WRITE( control%out, "( /, A, ' ** CG entered ** ',              &
              &    /, A, '    required gradient =', ES8.1, /, A,               &
              &    '    iter     model    proj grad    curvature     step')" ) &
               prefix, prefix, data%stop_cg, prefix
               WRITE( control%out,                                             &
                 "( A, 1X, I7, 2ES12.4, '      -            -     ' )" )       &
                 prefix, data%cg_iter, data%q_t, SQRT( ABS( gnrmsq ) )
             ELSE
               WRITE( control%out, "( A, 1X, I7, 4ES12.4 )" )                  &
                prefix, data%cg_iter, data%q_t, SQRT( ABS( gnrmsq ) ),         &
                data%curvature, data%step
             END IF
           END IF

!  if the gradient of the model is sufficiently small or if the CG iteration
!  limit is exceeded, exit; record the CG direction

           IF ( SQRT( ABS( gnrmsq ) ) <= data%stop_cg .OR.                     &
                data%cg_iter + 1 > data%cg_maxit ) THEN
             IF ( data% reverse ) THEN
               reverse%V = zero
               reverse%V( reverse%nz_v( : data%n_free) ) =                     &
                 data%S_free( : data%n_free )
             ELSE
               data%V = zero
               data%V( data%nz_v( : data%n_free ) ) =                          &
                 data%S_free( : data%n_free )
             END IF
             GO TO 410
           END IF

!  compute the search direction, p_free, and the square of its length

           data%cg_iter = data%cg_iter + 1
           IF ( data%cg_iter > 1 ) THEN
             beta = gnrmsq / data%old_gnrmsq
             data%P_free( : data%n_free ) = - data%PG_free( : data%n_free )    &
                + beta * data%P_free( : data%n_free )
             data%pnrmsq = gnrmsq + data%pnrmsq * beta ** 2
           ELSE
             data%P_free( : data%n_free ) = - data%PG_free( : data%n_free )
             data%pnrmsq = gnrmsq
           END IF

!  save the norm of the preconditioned gradient

           data%old_gnrmsq = gnrmsq

!  compute PROD = H * p ...

           IF ( data%explicit_h ) THEN
             data%PROD = zero
             DO l = 1, data%n_free
               i = data%NZ_v( l ) ; p_i = data%P_free( l )
               DO k = data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
                 data%PROD( data%H%col( k ) )                                  &
                   = data%PROD( data%H%col( k ) ) + data%H%val( k ) * p_i
               END DO
             END DO

!  ... or obtain the product from a user-provided subroutine ...

           ELSE IF ( data%use_hprod ) THEN
             data%V( data%nz_v( : data%n_free) )                               &
               = data%P_free( :  data%n_free )
             CALL eval_HPROD( i, userdata, data%V, data%PROD, NZ_v = data%NZ_v,&
                        nz_v_start = data%nz_v_start, nz_v_end = data%nz_v_end )

!  ... or return to the calling program to calculate PROD = H * p

           ELSE
             reverse%V( reverse%nz_v( : data%n_free) )                         &
               = data%P_free( :  data%n_free )
             data%branch = 400 ; inform%status = 3 ; RETURN
           END IF

!  record the free components of H * p

 400       CONTINUE
           IF ( data% reverse ) THEN
             data%HP_free( : data%n_free )                                     &
               = reverse%PROD( reverse%nz_v( : data%n_free) )
           ELSE
             data%HP_free( : data%n_free ) =                                   &
               data%PROD( data%nz_v( : data%n_free) )
           END IF

!  compute the curvature p^T H p along the search direction

           curvature = DOT_PRODUCT( data%HP_free( : data%n_free ),             &
                                    data%P_free( : data%n_free ) )
           data%curvature = curvature / data%pnrmsq

!  if the curvature is positive, compute the step to the minimizer of
!  the objective along the search direction

           IF ( curvature > control%zero_curvature * data%pnrmsq ) THEN
             data%step = data%old_gnrmsq / curvature

!  otherwise, the objective is unbounded ....

           ELSE IF ( curvature >= - control%zero_curvature * data%pnrmsq ) THEN
do i = 1, data%n_free
!write(6,"( ' p, hp ', 2ES12.4 )" ) data%P_free( i ), data%HP_free( i )
end do
!stop
!write(6,*) ' curvature ', data%curvature
             IF ( data% reverse ) THEN
               reverse%V = zero
               reverse%V( reverse%nz_v( : data%n_free) ) =                     &
                 data%P_free( : data%n_free )
             ELSE
               data%V = zero
               data%V( data%nz_v( : data%n_free ) ) =                          &
                 data%P_free( : data%n_free )
             END IF
             GO TO 410
           ELSE
do i = 1, data%n_free
!write(6,"( ' p, hp ', 2ES12.4 )" ) data%P_free( i ), data%HP_free( i )
end do
!write(6,*) ' curvature ', data%curvature
!stop
           inform%status = GALAHAD_error_inertia
             GO TO 900
           END IF

!  update the objective value

           data%q_t = data%q_t + data%step                                     &
             * ( - data%old_gnrmsq + half * data%step * curvature )

!  update the estimate of the solution

           data%S_free( : data%n_free ) = data%S_free( : data%n_free )         &
             + data%step * data%P_free( : data%n_free )

!  update the gradient at the estimate of the solution

           data%G_free( : data%n_free ) = data%G_free( : data%n_free )         &
             + data%step * data%HP_free( : data%n_free )
           GO TO 210

!  - - - - - - - - -
!  End the CG loop
!  - - - - - - - - -

 410     CONTINUE
         inform%cg_iter = inform%cg_iter + data%cg_iter

!         DO i = 1, prob%n
!           IF ( prob%X_l( i ) == prob%X( i ) .OR.                             &
!                prob%X_u( i ) == prob%X( i ) ) data%VARIABLE_status( i ) = 4
!         END DO

!      END IF

!  - - - - - - - - - -
!  Start stepsize loop
!  - - - - - - - - - -

 490     CONTINUE
       data%arcsearch_status = 1
       IF ( data%explicit_h ) THEN
         data%arcsearch_data%arcsearch_iter = 0 ; data%arcsearch_data%USED = 0
       END IF
 500   CONTINUE

!  find an improved point, X_new, by arcsearch when H is explicit ...

         IF ( data%explicit_h ) THEN

!  perform an exact arcsearch ...

           IF ( control%exact_arcsearch ) THEN
             CALL BQP_exact_arcsearch( prob%n, prob%X, data%G, inform%obj,  &
                                          prob%X_l, prob%X_u, t_max,           &
                                          data%X_new, data%q_t,                &
                                          data%VARIABLE_status, fixed_tol,     &
                                          data%V, data%NZ_v,                   &
                                          data%nz_v_start,                     &
                                          data%nz_v_end,                       &
                                          data%PROD, data%NZ_prod,             &
                                          data%nz_prod_end, data%out,          &
                                          data%print_level, prefix,            &
                                          data%arcsearch_status, data%n_free,  &
                                          data%arcsearch_data, userdata,       &
                                          H = data%H )

!  ... or an approximation

           ELSE
             CALL BQP_inexact_arcsearch( prob%n, prob%X, data%G, inform%obj,&
                                            prob%X_l, prob%X_u, t_max,         &
                                            data%X_new, data%q_t,              &
                                            data%VARIABLE_status, fixed_tol,   &
                                            mu_search,                         &
                                            data%V, data%NZ_v,                 &
                                            data%nz_v_start,                   &
                                            data%nz_v_end,                     &
                                            data%PROD, data%out,               &
                                            data%print_level, prefix,          &
                                            data%arcsearch_status, data%n_free,&
                                            data%arcsearch_data, userdata,     &
                                            H = data%H )
           END IF

!  ... or matrix-vector products are available via the user's subroutine ...

         ELSE IF ( data%use_hprod ) THEN

!  perform an exact arcsearch ...

           IF ( control%exact_arcsearch ) THEN
             CALL BQP_exact_arcsearch( prob%n, prob%X, data%G, inform%obj,     &
                                          prob%X_l, prob%X_u, t_max,           &
                                          data%X_new, data%q_t,                &
                                          data%VARIABLE_status, fixed_tol,     &
                                          data%V, data%NZ_v,                   &
                                          data%nz_v_start,                     &
                                          data%nz_v_end,                       &
                                          data%PROD, data%NZ_prod,             &
                                          data%nz_prod_end, data%out,          &
                                          data%print_level, prefix,            &
                                          data%arcsearch_status, data%n_free,  &
                                          data%arcsearch_data, userdata,       &
                                          eval_HPROD = eval_HPROD )

!  ... or an approximation

           ELSE
             CALL BQP_inexact_arcsearch( prob%n, prob%X, data%G, inform%obj,&
                                            prob%X_l, prob%X_u, t_max,         &
                                            data%X_new, data%q_t,              &
                                            data%VARIABLE_status, fixed_tol,   &
                                            mu_search,                         &
                                            data%V, data%NZ_v,                 &
                                            data%nz_v_start,                   &
                                            data%nz_v_end,                     &
                                            data%PROD, data%out,               &
                                            data%print_level, prefix,          &
                                            data%arcsearch_status, data%n_free,&
                                            data%arcsearch_data, userdata,     &
                                            eval_HPROD = eval_HPROD )
           END IF

!  ... or matrix-vector products are available via reverse communication

         ELSE

!  perform an exact arcsearch ...

           IF ( control%exact_arcsearch ) THEN
             CALL BQP_exact_arcsearch( prob%n, prob%X, data%G, inform%obj,     &
                                          prob%X_l, prob%X_u, t_max,           &
                                          data%X_new, data%q_t,                &
                                          data%VARIABLE_status, fixed_tol,     &
                                          reverse%V, reverse%NZ_v,             &
                                          reverse%nz_v_start,                  &
                                          reverse%nz_v_end,                    &
                                          reverse%PROD, reverse%NZ_prod,       &
                                          reverse%nz_prod_end, data%out,       &
                                          data%print_level, prefix,            &
                                          data%arcsearch_status, data%n_free,  &
                                          data%arcsearch_data, userdata )

!  ... or an approximation

           ELSE
             CALL BQP_inexact_arcsearch( prob%n, prob%X, data%G, inform%obj,   &
                                            prob%X_l, prob%X_u, t_max,         &
                                            data%X_new, data%q_t,              &
                                            data%VARIABLE_status, fixed_tol,   &
                                            mu_search,                         &
                                            reverse%V, reverse%NZ_v,           &
                                            reverse%nz_v_start,                &
                                            reverse%nz_v_end,                  &
                                            reverse%PROD, data%out,            &
                                            data%print_level, prefix,          &
                                            data%arcsearch_status, data%n_free,&
                                            data%arcsearch_data, userdata )
           END IF
         END IF

         SELECT CASE ( data%arcsearch_status )

!  successful exit with the new point

         CASE ( 0 )
           IF ( data% reverse ) THEN
             data%norm_step = MAXVAL( ABS( reverse%V ) )
           ELSE
             data%norm_step = MAXVAL( ABS( data%V ) )
           END IF
           GO TO 510

!  error exit without the new point

         CASE ( : - 1 )
           IF ( data%printe ) WRITE( control%error, 2010 )                     &
             prefix, data%arcsearch_status, 'BQP_(in)exact_arcsearch'
           GO TO 900

!  form the matrix-vector product H * v

         CASE ( 2 )
           data%branch = 500 ; inform%status = 3 ; RETURN

!  form the sparse matrix-vector product H * v

         CASE ( 3 )
           data%branch = 500 ; inform%status = 4 ; RETURN
         END SELECT
         GO TO 500

!  - - - - - - - - - - -
!  End the stepsize loop
!  - - - - - - - - - - -

 510   CONTINUE

!  record the new point in x

       prob%X = data%X_new
       inform%obj = data%q_t

!  record the number of variables that have changed status

       IF ( inform%iter > 0 ) data%change_status                               &
         = COUNT( data%VARIABLE_status /= data%OLD_status )
       data%OLD_status = data%VARIABLE_status
       GO TO 110

!  ----------------------
!  End the main iteration
!  ----------------------

!  Successful return

 900 CONTINUE
     CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN

!  Error returns

 910 CONTINUE
     CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start
     IF ( data%printi ) THEN
       SELECT CASE ( inform%status )
       CASE ( GALAHAD_ok )
       CASE ( GALAHAD_error_allocate )
         WRITE( control%out, 2020 ) prefix, 'allocation error'
         WRITE( control%out, 2030 ) prefix, inform%alloc_status,               &
                                            inform%bad_alloc
       CASE ( GALAHAD_error_deallocate )
         WRITE( control%out, 2020 ) prefix, 'de-allocation error'
         WRITE( control%out, 2030 ) prefix, inform%alloc_status,               &
                                            inform%bad_alloc
       CASE ( GALAHAD_error_restrictions )
         WRITE( control%out, 2020 ) prefix, 'input restriction violated'
       CASE ( GALAHAD_error_dual_infeasible )
         WRITE( control%out, 2020 ) prefix, 'no feasible point'
       CASE ( GALAHAD_error_unbounded )
         WRITE( control%out, 2020 ) prefix, 'problem unbounded'
       CASE ( GALAHAD_error_max_iterations )
         WRITE( control%out, 2020 ) prefix, 'iteration limit exceeded'
       CASE ( GALAHAD_error_cpu_limit )
         WRITE( control%out, 2020 ) prefix, 'CPU time limit exceeded'
       CASE ( GALAHAD_error_inertia )
         WRITE( control%out, 2020 ) prefix, 'problem is not strictly convex'
       CASE DEFAULT
         WRITE( control%out, 2020 ) prefix, 'undefined error'
       END SELECT
     ELSE IF ( data%printe ) THEN
       WRITE( control%error, 2010 ) prefix, inform%status, 'BQP_solve'
     END IF
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN

!  Non-executable statements

2000 FORMAT( /, A, ' --', A, ' BQP_solve' )
2010 FORMAT( A, '   **  Error return ', I0, ' from ', A )
2020 FORMAT( /, A, ' BQP error exit: ', A )
2030 FORMAT( /, A, ' allocation error status ', I0, ' for ', A )

!  End of BQP_solve

      END SUBROUTINE BQP_solve

!-*-*-*-*-*-*-   B Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

     SUBROUTINE BQP_terminate( data, control, inform, reverse )

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
!   data    see Subroutine BQP_initialize
!   control see Subroutine BQP_initialize
!   inform  see Subroutine BQP_solve
!   reverse see Subroutine BQP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( BQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( BQP_control_type ), INTENT( IN ) :: control
     TYPE ( BQP_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( BQP_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by SBLS

     CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,                &
                          inform%SBLS_inform )
     IF ( inform%SBLS_inform%status /= GALAHAD_ok ) THEN
       inform%status = GALAHAD_error_deallocate
       inform%alloc_status = inform%SBLS_inform%alloc_status
!      inform%bad_alloc = inform%SBLS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all remaining allocated arrays

     array_name = 'bqp: data%H%ptr'
     CALL SPACE_dealloc_array( data%H%ptr,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%H%col'
     CALL SPACE_dealloc_array( data%H%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%H%val'
     CALL SPACE_dealloc_array( data%H%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%G'
     CALL SPACE_dealloc_array( data%G,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%S_free'
     CALL SPACE_dealloc_array( data%S_free,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%G_free'
     CALL SPACE_dealloc_array( data%G_free,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%PG_free'
     CALL SPACE_dealloc_array( data%PG_free,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%P_free'
     CALL SPACE_dealloc_array( data%P_free,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%HP_free'
     CALL SPACE_dealloc_array( data%HP_free,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%X_new'
     CALL SPACE_dealloc_array( data%X_new,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%VARIABLE_status'
     CALL SPACE_dealloc_array( data%VARIABLE_status,                           &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%OLD_status'
     CALL SPACE_dealloc_array( data%OLD_status,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%arcsearch_data%USED'
     CALL SPACE_dealloc_array( data%arcsearch_data%USED,                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     IF ( PRESENT( reverse ) ) THEN
       array_name = 'bqp: reverse%V'
       CALL SPACE_dealloc_array( reverse%V,                                    &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqp: reverse%PROD'
       CALL SPACE_dealloc_array( reverse%PROD,                                 &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqp: reverse%NZ_v'
       CALL SPACE_dealloc_array( reverse%NZ_v,                                 &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqp: reverse%NZ_prod'
       CALL SPACE_dealloc_array( reverse%NZ_prod,                              &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN
     END IF

     array_name = 'bqp: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%PROD'
     CALL SPACE_dealloc_array( data%PROD,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%NZ_v'
     CALL SPACE_dealloc_array( data%NZ_v,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%NZ_prod'
     CALL SPACE_dealloc_array( data%NZ_prod,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%arcsearch_data%BREAKP'
     CALL SPACE_dealloc_array( data%arcsearch_data%BREAKP,                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqp: data%arcsearch_data%GRAD'
     CALL SPACE_dealloc_array( data%arcsearch_data%GRAD,                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine BQP_terminate

     END SUBROUTINE BQP_terminate

!-*-*-*-   B Q P _ E X A C T _ A R C S E A R C H   S U B R O U T I N E   -*-*-*-

     SUBROUTINE BQP_exact_arcsearch( n, X_0, G, f, X_l, X_u, t_max,         &
                                        X_t, q_t, VARIABLE_status, fixed_tol,  &
                                        P, NZ_p, nz_p_start, nz_p_end,         &
                                        HP, NZ_hp, nz_hp_end, out,             &
                                        print_level, prefix, status, n_free,   &
                                        data, userdata, H, eval_HPROD )

!  If we define the 'search arc' x(t) = projection of x_0 + t * p into the box
!  region x_l(*) <= x(*) <= x_u(*), find the global minimizer of the quadratic
!  function

!     1/2 (x-x_0)^T H (x-x_0) + g^T (x-x_0) + f

!  for points lying on x(t), with 0 <= t <= t_max.

!  Control is passed from the routine whenever a product of the vector P with
!  H is required, and the user is responsible for forming the product in the
!  vector HP

!  Progress through the routine is controlled by the parameter status

!  If status = 0, the minimizer has been found
!  If status = 1, an initial entry has been made
!  If status = 2, 3 the vector HP = H * P is required

!  The value of the array VARIABLE_status gives the status of the variables

!  IF VARIABLE_status( i ) = 0, the i-th variable is free
!  IF VARIABLE_status( i ) = 1, the i-th variable is fixed on its lower bound
!  IF VARIABLE_status( i ) = 2, the i-th variable is fixed on its upper bound
!  IF VARIABLE_status( i ) = 3, the i-th variable is permanently fixed
!  IF VARIABLE_status( i ) = 4, the i-th variable is fixed at some other value

!  The addresses of the free variables are given in the first n_free entries
!  of the array NZ_p

!  If the product H * P is required (status = 2,3), the nonzero components
!  of P occur in positions NZ_p(i) for i lying between nz_p_start and nz_p_end

!  At the initial point, variables within fixed_tol of their bounds and
!  for which the search direction P points out of the box will be fixed

!  ------------------------- dummy arguments --------------------------

!  n      (INTEGER) the number of independent variables.
!          ** this variable is not altered by the subroutine
!  X_0    (REAL array of length at least n) the point x_0 from which the search
!          arc commences. ** this variable is not altered by the subroutine
!  G      (REAL array of length at least n) the coefficients of
!          the linear term in the quadratic function
!          ** this variable is not altered by the subroutine
!  f      (REAL) the value of the quadratic at x_0, see above.
!          ** this variable is not altered by the subroutine
!  X_l    (REAL array of length at least n) the lower bounds on the variables
!  X_u    (REAL array of length at least n) the upper bounds on the variables
!  t_max  (REAL) the largest allowable value of t
!  X_t    (REAL array of length at least n) the current estimate of the
!          minimizer
!  q_t    (REAL) the value of the piecewise quadratic function at the current
!          estimate of the minimizer
!  VARIABLE_status (INTEGER array of length at least n) specifies which
!          of the variables are to be fixed at the start of the minimization.
!          VARIABLE_status should be set as follows:
!          If VARIABLE_status( i ) = 0, the i-th variable is free
!          If VARIABLE_status( i ) = 1, the i-th variable is on its lower bound
!          If VARIABLE_status( i ) = 2, the i-th variable is on its upper bound
!          If VARIABLE_status( i ) = 3, 4, the i-th variable is fixed at X_t(i)
!  fixed_tol (REAL) a tolerance on feasibility of X_0, see above.
!          ** this variable is not altered by the subroutine.
!  P      (REAL array of length at least n) contains the values of the
!          components of the vector P. On initial (status=1) entry, P must
!          contain the initial direction of the 'search arc'. On a non optimal
!          exit, (status=2,3), P is the vector for which the product H * P
!          is required before the next re-entry. On a terminal exit (status=0),
!          P contains the step X_t - X_0. The components NZ_p(i) = nz_p_start,
!           ... , nz_p_end of P contain the values of the nonzero components
!          of P (see, NZ_p, nz_p_start, nz_p_end)
!  NZ_p   (INTEGER array of length at least n) on all normal exits
!         (status=0,2,3), NZ_p(i), i = nz_p_start, ..., nz_p_end, gives the
!          indices of the nonzero components of P
!  nz_p_start  (INTEGER) see NZ_p, above
!  nz_p_end  (INTEGER) see NZ_p, above
!  HP     (REAL array of length at least n) on a non initial entry
!         (status=2,3), HP must contain the vector H * P. Only the
!          components NZ_p(i), i=1,...,n_free, of HP need be set (the other
!          components are not used)
!  NZ_hp (INTEGER array of length at least nz_hp_end) on status = 3 entries,
!          NZ_hp(i), i = 1,....,nz_hp_end, must give the indices of the
!          nonzero components of HP. On other entries, NZ_hp need not be set
!  nz_hp_end (INTEGER) the number of nonzero components of HP on a status=3
!          entry. nz_hp_end need not be set on other entries
!  out   (INTEGER) the fortran output channel number to be used
!  print_level (INTEGER) allows detailed printing. If print_level is larger
!          than 4, detailed output from the routine will be given. Otherwise,
!          no output occurs
!  status (INTEGER) controls flow through the subroutine.
!          If status = 0, the minimizer has been found
!          If status = 1, an initial entry has been made
!          If status = 2 or 3, the vector HP = H * P is required
!  n_free (INTEGER) the number of free variables at the initial point
!  data   (BQP_arcsearch_data_type) private data that must be preserved between
!          calls
!  userdata (NLPT_userdata_type) user provided data for use in eval_HPROD
!  H      (SMT_type) optionaly, the whole of H stored by rows
!  H_PROD subroutine, optionally, compute H * vector products

!  ------------------ end of dummy arguments --------------------------

!  Based on CAUCHY_get_exact_gcp from LANCELOT B

!  Dummy arguments

     INTEGER, INTENT( IN ):: n, out, print_level
     INTEGER, INTENT( INOUT ):: n_free, nz_p_start, nz_p_end, nz_hp_end, status
     REAL ( KIND = wp ), INTENT( IN ):: t_max
     REAL ( KIND = wp ), INTENT( IN ):: fixed_tol
     REAL ( KIND = wp ), INTENT( INOUT ):: f, q_t
     CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
     INTEGER, DIMENSION( n ), INTENT( INOUT ) :: VARIABLE_status
     INTEGER, DIMENSION( n ), INTENT( INOUT ) :: NZ_p, NZ_hp
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_0, G
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X_t
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: P, HP
     TYPE ( BQP_arcsearch_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     TYPE ( SMT_type ), OPTIONAL, INTENT( IN ) :: H
     OPTIONAL :: eval_HPROD

!  interface blocks

     INTERFACE
       SUBROUTINE eval_HPROD( status, userdata, V, PROD, NZ_v, nz_v_start,     &
                              nz_v_end, NZ_prod, nz_prod_end )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: PROD
       INTEGER, OPTIONAL, INTENT( IN ) :: nz_v_start, nz_v_end
       INTEGER, OPTIONAL, INTENT( INOUT ) :: nz_prod_end
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_v
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_prod
       END SUBROUTINE eval_HPROD
     END INTERFACE

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      n, X_0, G, f, X_l, X_u, t_max, fixed_tol, P, out, print_level, status

!  status must have the value 1. In addition, if the i-th variable is required
!  to be fixed at its initial value, X_0(i), VARIABLE_status(i) must be set to
!  3 or 4.

!  RE-ENTRY:

!  If the variable status has the value 2 or 3 on exit, the
!  subroutine must be re-entered with the vector HP containing
!  the product of the second derivative matrix H and the output
!  vector P. All other parameters MUST NOT BE ALTERED

!  Local variables

     INTEGER :: i, j, k, l, ibreak, insort, n_freed
     REAL ( KIND = wp ) :: t, tstar, gp, pbp, feasep, p_i, qipi
     LOGICAL :: xlower, xupper

!  Enter or re-enter the package and jump to appropriate re-entry point

     IF ( status <= 1 ) data%branch = 100

     SELECT CASE ( data%branch )
     CASE ( 100 ) ; GO TO 100
     CASE ( 200 ) ; GO TO 200
     CASE ( 300 ) ; GO TO 300
     CASE ( 400 ) ; GO TO 400
     END SELECT

!  On initial entry, set constants

 100 CONTINUE
     data%use_hprod = PRESENT( eval_HPROD )
     data%explicit_h = PRESENT( H )
     data%prnter = print_level >= 4 .AND. out > 0
     data%pronel = print_level == 3 .AND. out > 0
!    data%prnter = .TRUE. ; data%pronel = .FALSE.
     IF ( data%prnter .OR. data%pronel ) WRITE( out,                           &
       "( / A, ' ** exact arcsearch entered ** ' )" ) prefix
     data%nbreak = 0 ; n_freed = 0 ; data%nzero = n + 1
     data%epstl2 = ten * epsmch ; data%epsqrt = SQRT( epsmch )
     data%tbreak = zero

     IF ( print_level >= 100 .AND. out > 0 ) THEN
       DO i = 1, n
         WRITE( out, "( A, ' Var low X up P ', I6, 4ES12.4 )" )                &
           prefix, i, X_l( i ), X_0( i ), X_u( i ), P( i )
       END DO
     END IF

!  Find the status of the variables

!DIR$ IVDEP
     DO i = 1, n

!  Check to see whether the variable is fixed

       IF ( VARIABLE_status( i ) <= 2 ) THEN
         VARIABLE_status( i ) = 0
         xupper = X_u( i ) - X_0( i ) <= fixed_tol
         xlower = X_0( i ) - X_l( i ) <= fixed_tol
         IF ( .NOT. ( xupper .OR. xlower ) ) THEN

!  The variable lies between its bounds. Check to see if the search
!  direction is zero

           IF ( ABS( P( i ) ) > epsmch ) GO TO 110
           data%nzero = data%nzero - 1
           NZ_p( data%nzero ) = i
         ELSE
           IF ( xlower ) THEN

!  The variable lies close to its lower bound

             IF ( P( i ) > epsmch ) THEN
               n_freed = n_freed + 1
               GO TO 110
             END IF
             VARIABLE_status( i ) = 1
           ELSE

!  The variable lies close to its upper bound

             IF ( P( i ) < - epsmch ) THEN
               n_freed = n_freed + 1
               GO TO 110
             END IF
             VARIABLE_status( i ) = 2
           END IF
         END IF
       END IF

!  Set the search direction to zero

       X_t( i ) = X_0( i )
       P( i ) = zero
       CYCLE
 110   CONTINUE

!  If the variable is free, set up the pointers to the nonzeros in the vector
!  P ready for calculating HP = H * P

       data%nbreak = data%nbreak + 1
       NZ_p( data%nbreak ) = i
     END DO

!  Record the number of free variables at the starting point

     n_free = data%nbreak ; nz_p_end = n_free ; q_t = f

!  If all of the variables are fixed, exit

     IF ( data%pronel ) WRITE( out, "( A, '  ', I0,' vars. freed, ', I0,       &
    &  ' vars. remain fixed' )" ) prefix, n_freed, n - data%nbreak
     IF ( data%prnter ) WRITE( out, "( A, I0, ' variables freed from',         &
    &  ' their bounds ', /, A, I0, ' variables remain fixed ',/ )" )           &
          prefix, n_freed, prefix, n - data%nbreak
     IF ( data%nbreak == 0 ) GO TO 600
     data%iter = 0
     IF ( data%pronel ) WRITE( out,                                            &
       "( A, ' segment    model      gradient   curvature     step' )" )       &
         prefix

!  Find the breakpoints for the piecewise linear arc (the distances
!  to the boundary)

     DO j = 1, data%nbreak
       i = NZ_p( j )
       IF ( P( i ) > epsmch ) THEN
         t = ( X_u( i ) - X_0( i ) ) / P( i )
       ELSE
         t = ( X_l( i ) - X_0( i ) ) / P( i )
       END IF
       data%BREAKP( j ) = t
     END DO

!  Order the breakpoints in increasing size using a heapsort. Build the heap

     CALL SORT_heapsort_build( data%nbreak, data%BREAKP, insort, INDA = NZ_p )

!  Calculate HP = H * p ...

     nz_p_start = 1 ; nz_p_end = n_free
     IF ( data%explicit_h ) THEN
       HP = zero
       DO l = nz_p_start, nz_p_end
         i = NZ_p( l ) ; p_i = P( i )
         DO k = H%ptr( i ), H%ptr( i + 1 ) - 1
           HP( H%col( k ) ) = HP( H%col( k ) ) + H%val( k ) * p_i
         END DO
       END DO

!  ... or obtain the product from a user-provided subroutine ...

     ELSE IF ( data%use_hprod ) THEN
       CALL eval_HPROD( i, userdata, P, HP, NZ_v = NZ_p,                       &
                        nz_v_start = nz_p_start, nz_v_end = nz_p_end )

!  ... or return to the calling program to calculate HP = H * p

     ELSE
       data%branch = 200 ; status = 2 ; RETURN
     END IF

!  Calculate the function value (q_t), first derivative (gxt) and
!  second derivative (hxt) of the univariate piecewise quadratic
!  function at the start of the piecewise linear arc

 200 CONTINUE
     data%gxt = zero ; data%hxt = zero
     DO i = 1, n_free
       data%gxt = data%gxt + G( NZ_p( i ) ) * P( NZ_p( i ) )
       data%hxt = data%hxt + HP( NZ_p( i ) ) * P( NZ_p( i ) )
     END DO

     IF ( data%explicit_h ) THEN
       data%arcsearch_iter = 0 ; data%USED = 0
     END IF

!  Start the main loop to find the first local minimizer of the piecewise
!  quadratic function. Consider the problem over successive pieces

 210 CONTINUE

!  Print details of the piecewise quadratic in the next interval

       data%iter = data%iter + 1
       IF ( data%prnter ) WRITE( out,                                          &
         "( /, A, ' Piece', I5, ' - f, G and H at start point ', 4ES12.4 )" )  &
           prefix, data%iter, q_t, data%gxt, data%hxt, data%tbreak
       IF ( data%pronel ) WRITE( out, "( A, 1X, I7, 4ES12.4 )" )               &
         prefix, data%iter, q_t, data%gxt, data%hxt, data%tbreak

!  If the gradient of the univariate function increases, exit

       IF ( data%gxt > g_zero ) GO TO 600

!  Record the value of the last breakpoint

       data%tk = data%tbreak

!  Find the next breakpoint ( end of the piece )

       data%tbreak = data%BREAKP(  1  )
       CALL SORT_heapsort_smallest( data%nbreak, data%BREAKP, insort,          &
                                    INDA = NZ_p )

!  Compute the length of the current piece

       data%deltat = MIN( data%tbreak, t_max ) - data%tk

!  Print details of the breakpoint

       IF ( data%prnter ) THEN
         WRITE( out, "( /, A, ' Next break point = ', ES12.4, /, A,            &
        & ' Maximum step     = ', ES12.4 )" ) prefix, data%tbreak, prefix, t_max
       END IF

!  If the gradient of the univariate function is small and its curvature
!  is positive, exit

       IF ( ABS( data%gxt ) <= g_zero ) THEN
         IF ( data%hxt > - h_zero .OR. data%deltat >= infinity ) THEN
           data%tcauch = data%tk
           GO TO 600
         ELSE
           data%tcauch = data%tbreak
         END IF
       ELSE

!  If the gradient of the univariate function is nonzero and its curvature is
!  positive, compute the line minimum

         IF ( data%hxt > zero ) THEN
           tstar = - data%gxt / data%hxt
           IF ( data%prnter )                                                  &
             WRITE( out, "( A, ' Stationary point = ', ES12.4 )" ) prefix, tstar

!  If the line minimum occurs before the breakpoint, the line minimum gives
!  the required minimizer. Exit

           data%tcauch = MIN( data%tk + tstar, data%tbreak )
           IF ( tstar < data%deltat ) THEN
             data%deltat = tstar
             GO TO 500
           END IF
         ELSE
           data%tcauch = data%tbreak
         END IF
       END IF

!  If the required minimizer occurs at t_max, exit.

       IF ( t_max < data%tcauch ) THEN
         data%tcauch = t_max
         data%deltat = t_max - data%tk
         GO TO 500
       END IF

!  Update the univariate function and gradient values

       q_t = q_t + data%deltat * ( data%gxt + half * data%deltat * data%hxt )
       data%gxtold = data%gxt ; data%gxt = data%gxt + data%deltat * data%hxt

!  Record the new breakpoint and the amount by which other breakpoints
!  are allowed to vary from this one and still be considered to be
!  within the same cluster

       feasep = data%tbreak + data%epstl2

!  Move the appropriate variable(s) to their bound(s)

 220   CONTINUE
       ibreak = NZ_p( data%nbreak )
       data%nbreak = data%nbreak - 1
       IF ( data%prnter ) WRITE( out, 2000 ) prefix, ibreak, data%tbreak

!  Indicate the status of the newly fixed variable - the value is negated to
!  indicate that the variable has just been fixed

       IF ( P( ibreak ) < zero ) THEN
         VARIABLE_status( ibreak ) = - 1
       ELSE
         VARIABLE_status( ibreak ) = - 2
       END IF

!  If all of the remaining search direction is zero, return

       IF ( data%nbreak == 0 ) THEN
!DIR$ IVDEP
         DO j = 1, nz_p_end
           i = NZ_p( j )

!  Restore VARIABLE_status to its correct sign

           VARIABLE_status( i ) = - VARIABLE_status( i )

!  Move the variable onto its bound

           IF ( VARIABLE_status( i ) == 1 ) THEN
             X_t( i ) = X_l( i )
           ELSE
             X_t( i ) = X_u( i )
           END IF

!  Store the step from the initial point to the minimizer in P

           P( i ) = X_t( i ) - X_0( i )
         END DO
         nz_p_end = 0
         GO TO 600
       END IF

!  Determine if other variables hit their bounds at the breakpoint

       IF (  data%BREAKP(  1  ) < feasep  ) THEN
         CALL SORT_heapsort_smallest( data%nbreak, data%BREAKP, insort,        &
                                      INDA = NZ_p )
         GO TO 220
       END IF

!  Calculate HP = H * P ...

       nz_p_start = data%nbreak + 1
       IF ( data%explicit_h ) THEN

!  record the nonzero indices of H * P

         data%arcsearch_iter = data%arcsearch_iter + 1
         nz_hp_end = 0
         DO l = nz_p_start, nz_p_end
           i = NZ_p( l )
           DO k = H%ptr( i ), H%ptr( i + 1 ) - 1
             j = H%col( k )
             IF ( data%USED( j ) < data%arcsearch_iter ) THEN
               data%USED( j ) = data%arcsearch_iter
               HP( j ) = zero
               nz_hp_end = nz_hp_end + 1
               NZ_hp( nz_hp_end ) = j
             END IF
           END DO
         END DO

!  assign the nonzeros to HP

         DO l = nz_p_start, nz_p_end
           i = NZ_p( l ) ; p_i = P( i )
           DO k = H%ptr( i ), H%ptr( i + 1 ) - 1
             HP( H%col( k ) ) = HP( H%col( k ) ) + H%val( k ) * p_i
           END DO
         END DO

!  ... or obtain the product from a user-provided subroutine ...

       ELSE IF ( data%use_hprod ) THEN
         CALL eval_HPROD( i, userdata, P, HP, NZ_v = NZ_p,                     &
                          nz_v_start = nz_p_start, nz_v_end = nz_p_end,        &
                          NZ_prod = NZ_hp, nz_prod_end = nz_hp_end )

!  ... or return to the calling program to calculate HP = H * P

       ELSE
         data%branch = 300 ; status = 3 ; RETURN
       END IF

!  Update the first and second derivatives of the univariate function

 300   CONTINUE

!  Start with the second-order terms. Only process nonzero components of HP

!DIR$ IVDEP
       DO j = 1, nz_hp_end
         i = NZ_hp( j )
         qipi = HP( i ) * P( i )
         IF ( VARIABLE_status( i ) == 0 ) THEN

!  Include contributions from the free components of HP

           data%gxt = data%gxt - qipi * data%tbreak
           data%hxt = data%hxt - qipi * two
         ELSE
           IF ( VARIABLE_status( i ) < 0 ) THEN

!  Include contributions from the components of HP which were just fixed

             data%gxt = data%gxt - qipi * data%tbreak
             data%hxt = data%hxt - qipi
           ELSE

!  Include contributions from the components of HP which were previously fixed

             data%gxt = data%gxt - qipi
           END IF
         END IF
       END DO

!  Now include the contributions from the variables which have just been fixed

!DIR$ IVDEP
       DO j = nz_p_start, nz_p_end
         i = NZ_p( j )
         data%gxt = data%gxt - P( i ) * G( i )

!  Restore VARIABLE_status to its correct sign

         VARIABLE_status( i ) = - VARIABLE_status( i )

!  Move the variable onto its bound

         IF ( VARIABLE_status( i ) == 1 ) THEN
           X_t( i ) = X_l( i )
         ELSE
           X_t( i ) = X_u( i )
         END IF

!  Store the step from the initial point to the minimizer in P

         P( i ) = X_t( i ) - X_0( i )
       END DO

!  Reset the number of free variables

       nz_p_end = data%nbreak

!  Check that the size of the line gradient has not shrunk significantly in
!  the current segment of the piecewise arc. If it has, there may be a loss
!  of accuracy, so the line derivatives will be recomputed

       data%recomp = ABS( data%gxt ) < - data%epsqrt * data%gxtold

!  If required, compute the true line gradient and curvature.

       IF ( data%recomp .OR. data%prnter ) THEN

!  Calculate HP = H * P ...

         nz_p_start = 1
         IF ( data%explicit_h ) THEN
           HP = zero ;
           DO l = nz_p_start, nz_p_end
             i = NZ_p( l ) ; p_i = P( i )
             DO k = H%ptr( i ), H%ptr( i + 1 ) - 1
               HP( H%col( k ) ) = HP( H%col( k ) ) + H%val( k ) * p_i
             END DO
           END DO

!  ... or obtain the product from a user-provided subroutine ...

         ELSE IF ( data%use_hprod ) THEN
           CALL eval_HPROD( i, userdata, P, HP, NZ_v = NZ_p,                   &
                            nz_v_start = nz_p_start, nz_v_end = nz_p_end )

!  ... or return to the calling program to calculate HP = H * P

         ELSE
           data%branch = 400 ; status = 2 ; RETURN
         END IF
       END IF

!  Calculate the line gradient and curvature

 400   CONTINUE
       IF ( data%recomp .OR. data%prnter ) THEN
         pbp = zero ; gp = zero
         IF ( print_level > 100 .AND. out > 0 ) WRITE( out,                    &
           "( A, ' Current search direction ', /, ( 4( I6, ES12.4 ) ) )" )     &
            prefix, ( NZ_p( i ), P( NZ_p( i ) ), i = 1, nz_p_end )
         DO j = 1, nz_p_end
           i = NZ_p( j )
           qipi = P( i ) * HP( i )
           pbp = pbp + qipi ; gp = gp + P( i ) * G( i ) + data%tbreak * qipi
         END DO
         DO i = nz_p_end + 1, n_free
           gp = gp + P( NZ_p( i ) ) * HP( NZ_p( i ) )
         END DO
         IF ( data%prnter )                                                    &
           WRITE( out, "( /, A, ' Calculated gxt and hxt = ', 2ES12.4, /,      &
          &   A, ' Recurred   gxt and hxt = ', 2ES12.4 )" )                    &
            prefix, gp, pbp, prefix, data%gxt, data%hxt
         IF ( data%recomp ) THEN
           data%gxt = gp ; data%hxt = pbp
         END IF
       END IF

!  End of the main loop. Jump back to calculate the next breakpoint

     GO TO 210

!  Step to the minimizer

 500 CONTINUE

!  Calculate the function value for the piecewise quadratic

     q_t = q_t + data%deltat * ( data%gxt + half * data%deltat * data%hxt )
     IF ( data%pronel ) WRITE( out, "( A, 1X, I7, ES12.4, 24X, ES12.4 )" )     &
       prefix, data%iter + 1, q_t, data%tcauch
     IF ( data%prnter ) WRITE( out,                                            &
       "( /, A, ' Function value at the arc minimizer ', ES12.4 )" ) prefix, q_t

!  The arc minimizer has been found. Set the array P to the step from the
!  initial point to the minimizer

 600 CONTINUE
     P( NZ_p( : nz_p_end ) ) = data%tcauch * P( NZ_p( : nz_p_end ) )
     X_t( NZ_p( : nz_p_end ) )                                                 &
       = X_0( NZ_p( : nz_p_end ) ) + P( NZ_p( : nz_p_end ) )

!  Record that variables whose gradients were zero at the initial
!  point are free variables

     DO j = data%nzero, n
       data%nbreak = data%nbreak + 1
       NZ_p( data%nbreak ) = NZ_p( j )
     END DO

!  Set return conditions

     status = 0 ; nz_p_start = 1 ; nz_p_end = data%nbreak
     n_free = data%nbreak

     RETURN

!  Non-executable statement

2000 FORMAT( A, ' Variable ', I4, ' is fixed, step = ', ES12.4 )

!  End of subroutine BQP_exact_arcsearch

     END SUBROUTINE BQP_exact_arcsearch

!-*-*-  B Q P _ I N E X A C T _ A R C S E A R C H   S U B R O U T I N E  -*-*-

     SUBROUTINE BQP_inexact_arcsearch( n, X_0, G, f, X_l, X_u, t_max, X_t,  &
                                          q_t, VARIABLE_status, fixed_tol, mu, &
                                          P, NZ_p, nz_p_start, nz_p_end, HP,   &
                                          out, print_level, prefix, status,    &
                                          n_free, data, userdata, H,           &
                                          eval_HPROD )

!  If we define the 'search arc' x(t) = projection of x_0 + t * p into the box
!  region x_l(*) <= x(*) <= x_u(*), find a suitable approximation to the global
!  minimizer of the quadratic function

!     1/2 (x-x_0)^T H (x-x_0) + g^T (x-x_0) + f

!  for points lying on x(t), with 0 <= t <= t_max. A suitable inexact
!  arc search is defined as follows:

!  1) If the minimizer of q(x) along x_0 + t * p lies on the search arc,
!     this is the required point. Otherwise,

!  2) Starting from some specified t_0, construct a decreasing sequence
!     of values t_1, t_2, t_3, .... . Given 0 < mu < 1, pick the first
!     t_i (I = 0, 1, ...) for which the Armijo condition

!        q(x(t_i)) <= linear(x(t_i),mu) = f + mu * g^T (x(t_i) - x_0)

!     is satisfied. x_0 + t_i * p is then the required point

!  Progress through the routine is controlled by the parameter status

!  If status = 0, the approximate minimizer has been found
!  If status = 1, an initial entry has been made
!  If status = 2 the vector HP = H * P is required

!  The value of the array VARIABLE_status gives the status of the variables

!  IF VARIABLE_status( i ) = 0, the i-th variable is free
!  IF VARIABLE_status( i ) = 1, the i-th variable is fixed on its lower bound
!  IF VARIABLE_status( i ) = 2, the i-th variable is fixed on its upper bound
!  IF VARIABLE_status( i ) = 3, the i-th variable is permanently fixed
!  IF VARIABLE_status( i ) = 4, the i-th variable is fixed at some other value

!  The addresses of the free variables are given in the first n_free entries
!  of the array NZ_p

!  If the product H * P is required (status = 2,3,4), the nonzero components
!  of P occur in positions NZ_p(i) for i lying between nz_p_start and nz_p_end

!  At the initial point, variables within fixed_tol of their bounds and
!  for which the search direction P points out of the box will be fixed

!  ------------------------- dummy arguments --------------------------

!  n      (INTEGER) the number of independent variables.
!          ** this variable is not altered by the subroutine
!  X_0     (REAL array of length at least n) the point x0 from which the search
!          arc commences. ** this variable is not altered by the subroutine
!  G      (REAL array of length at least n) the coefficients of
!          the linear term in the quadratic function
!          ** this variable is not altered by the subroutine
!  f      (REAL) the value of the quadratic at X_0, see above.
!          ** this variable is not altered by the subroutine
!  X_l    (REAL array of length at least n) the lower bounds on the variables
!  X_u    (REAL array of length at least n) the upper bounds on the variables
!  t_max  (REAL) the largest allowable value of t
!  X_t    (REAL array of length at least n) the current estimate of the
!          minimizer
!  q_t    (REAL) the value of the piecewise quadratic function at the current
!          estimate of the minimizer
!  VARIABLE_status (INTEGER array of length at least N) specifies which
!          of the variables are to be fixed at the start of the minimization.
!          VARIABLE_status should be set as follows:
!          If VARIABLE_status( i ) = 0, the i-th variable is free
!          If VARIABLE_status( i ) = 1, the i-th variable is on its lower bound
!          If VARIABLE_status( i ) = 2, the i-th variable is on its upper bound
!          If VARIABLE_status( i ) = 3, 4, the i-th variable is fixed at X_t(i)
!  fixed_tol (REAL) a tolerance on feasibility of X_0, see above.
!          ** this variable is not altered by the subroutine.
!  mu     (REAL) the slope of the majorizing linear model linear(x,mu)
!  P      (REAL array of length at least n) contains the values of the
!          components of the vector P. On initial (status=1) entry, P must
!          contain the initial direction of the 'search arc'. On a non optimal
!          exit, (status=2,3,4), P is the vector for which the product H * P
!          is required before the next re-entry. On a terminal exit (status=0),
!          P contains the step X_t - X_0. The components NZ_p(i) = nz_p_start,
!          ... , nz_p_end of P contain the values of the nonzero components of
!          P (see, NZ_p, nz_p_start, nz_p_end)
!  NZ_p   (INTEGER array of length at least n) on all normal exits
!         (status=0,2), NZ_p(i), i = nz_p_start, ..., nz_p_end, gives
!          the indices of the nonzero components of P
!  nz_p_start  (INTEGER) see NZ_p, above
!  nz_p_end  (INTEGER) see NZ_p, above
!  HP     (REAL array of length at least n) on a non initial entry (status=2),
!          HP must contain the vector H * P. Only the components NZ_p(i),
!          i=1,...,n_free, of HP need be set (the other components are not used)
!  out    (INTEGER) the fortran output channel number to be used
!  print_level (INTEGER) allows detailed printing. If print_level is larger
!          than 4, detailed output from the routine will be given. Otherwise,
!          no output occurs
!  status (INTEGER) controls flow through the subroutine.
!          If status = 0, the minimizer has been found
!          If status = 1, an initial entry has been made
!          If status = 2, the vector HP = H * P is required
!  n_free  (INTEGER) the number of free variables at the initial point
!  data   (BQP_arcsearch_data_type) private data that must be preserved between
!          calls
!  userdata (NLPT_userdata_type) user provided data for use in eval_HPROD
!  H      (SMT_type) optionaly, the whole of H stored by rows
!  H_PROD subroutine, optionally, compute H * vector products

!  ------------------ end of dummy arguments --------------------------

!  Based on CAUCHY_get_approx_gcp from LANCELOT B

!  Dummy arguments

     INTEGER, INTENT( IN    ):: n, out, print_level
     INTEGER, INTENT( INOUT ):: n_free, nz_p_start, nz_p_end, status
     REAL ( KIND = wp ), INTENT( IN ):: t_max, mu
     REAL ( KIND = wp ), INTENT( IN ):: fixed_tol
     REAL ( KIND = wp ), INTENT( INOUT ):: f, q_t
     CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
     INTEGER, DIMENSION( n ), INTENT( INOUT ) :: VARIABLE_status
     INTEGER, DIMENSION( n ), INTENT( INOUT ) :: NZ_p
     REAL ( KIND = wp ), INTENT( IN    ), DIMENSION( n ) :: X_l, X_u
     REAL ( KIND = wp ), INTENT( IN    ), DIMENSION( n ) :: X_0, G
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X_t
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: P, HP
     TYPE ( BQP_arcsearch_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     TYPE ( SMT_type ), OPTIONAL, INTENT( IN ) :: H
     OPTIONAL :: eval_HPROD

!  interface blocks

     INTERFACE
       SUBROUTINE eval_HPROD( status, userdata, V, PROD, NZ_v, nz_v_start,     &
                              nz_v_end, NZ_prod, nz_prod_end )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: PROD
       INTEGER, OPTIONAL, INTENT( IN ) :: nz_v_start, nz_v_end
       INTEGER, OPTIONAL, INTENT( INOUT ) :: nz_prod_end
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_v
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_prod
       END SUBROUTINE eval_HPROD
     END INTERFACE

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!   n, X_0, G, f, X_l, X_u, t_max, fixed_tol, mu, P, out, print_level, status

!  status must have the value 1. In addition, if the i-th variable is required
!  to be fixed at its initial value, X_0(i), VARIABLE_status(i) must be set to
!  3 or 4.

!  RE-ENTRY:

!  If the variable status has the value 2 or 3 on exit, the
!  subroutine must be re-entered with the vector HP containing
!  the product of the second derivative matrix H and then output
!  vector P. All other parameters MUST NOT BE ALTERED

!  Local variables

     INTEGER :: i, j, k, l, nbreak, n3
     REAL ( KIND = wp ) :: ptbp, p_i, tbmax
     LOGICAL :: xlower, xupper

!  Enter or re-enter the package and jump to appropriate re-entry point

     IF ( status <= 1 ) data%branch = 100

     SELECT CASE ( data%branch )
     CASE ( 100 ) ; GO TO 100
     CASE ( 200 ) ; GO TO 200
     CASE ( 300 ) ; GO TO 300
     CASE ( 400 ) ; GO TO 400
     END SELECT

!  On initial entry, set constants

 100 CONTINUE
     data%use_hprod = PRESENT( eval_HPROD )
     data%explicit_h = PRESENT( H )
     data%prnter = print_level >= 3 .AND. out > 0
     data%pronel = print_level == 2 .AND. out > 0
     data%iterca = 0 ; data%itmax = 100
     data%n_freed = 0 ; nbreak = 0 ; data%nzero = n + 1
     data%t = zero

!  Find the status of the variables

!DIR$ IVDEP
     DO i = 1, n

!  Check to see whether the variable is fixed

       IF ( VARIABLE_status( i ) <= 2 ) THEN
         VARIABLE_status( i ) = 0
         xupper = X_u( i ) - X_0( i ) <= fixed_tol
         xlower = X_0( i ) - X_l( i ) <= fixed_tol
         IF ( .NOT. ( xupper .OR. xlower ) ) THEN

!  The variable lies between its bounds. Check to see if the search direction
!  is zero

           IF ( ABS( P( i ) ) > epsmch ) GO TO 110
           data%nzero = data%nzero - 1
           NZ_p( data%nzero ) = i
         ELSE
           IF ( xlower ) THEN
!
!  The variable lies close to its lower bound
!
             IF ( P( i ) > epsmch ) THEN
               data%n_freed = data%n_freed + 1
               GO TO 110
             END IF
             VARIABLE_status( i ) = 1
           ELSE

!  The variable lies close to its upper bound

             IF ( P( i ) < - epsmch ) THEN
               data%n_freed = data%n_freed + 1
               GO TO 110
             END IF
             VARIABLE_status( i ) = 2
           END IF
         END IF
       END IF

!  Set the search direction to zero

       P( i ) = zero
       X_t( i ) = X_0( i )
       CYCLE
  110  CONTINUE

!  If the variable is free, set up the pointers to the nonzeros in the vector
!  P ready for calculating HP = H * P. The vector X_t is used temporarily to
!  store the original input direction P

       nbreak = nbreak + 1
       NZ_p( nbreak ) = i
       X_t( i ) = P( i )
     END DO

!  If all of the variables are fixed, exit

     IF ( data%prnter ) WRITE( out,                                            &
       "( /, A, ' ----------- inexact arcsearch -------------', //,            &
      &  A, I8, ' variables freed from their bounds ', /, A, I8,               &
      &  ' variables remain fixed ', / )" ) prefix, prefix, data%n_freed,      &
                                            prefix, n - nbreak
     IF ( data%pronel .OR. data%prnter )                                       &
        WRITE( out, "( /, A, 3X, ' ** arcsearch entered  iter     step     ',  &
       &  '  HP( step )   L( step,mu )', /, A, 21X, I6, 2ES12.4 )" )           &
     prefix, prefix, data%iterca, zero, f
     data%iterca = data%iterca + 1
     n_free = nbreak
     IF ( nbreak == 0 ) GO TO 600

!  Calculate HP = H * P ...

     nz_p_start = 1 ; nz_p_end = nbreak
     IF ( data%explicit_h ) THEN
       HP = zero
       DO l = nz_p_start, nz_p_end
         i = NZ_p( l ) ; p_i = P( i )
         DO k = H%ptr( i ), H%ptr( i + 1 ) - 1
           HP( H%col( k ) ) = HP( H%col( k ) ) + H%val( k ) * p_i
         END DO
       END DO

!  ... or obtain the product from a user-provided subroutine ...

     ELSE IF ( data%use_hprod ) THEN
       CALL eval_HPROD( i, userdata, P, HP, NZ_v = NZ_p,                       &
                        nz_v_start = nz_p_start, nz_v_end = nz_p_end )

!  ... or return to the calling program to calculate HP = H * P

     ELSE
       data%branch = 200 ; status = 2 ; RETURN
     END IF

 200 CONTINUE

!  Compute the slope and curvature of quad( X ) in the direction P

     data%gtp = zero ; ptbp = zero
     data%tbreak = infinity ; tbmax = zero
!DIR$ IVDEP
     DO j = 1, n_free
       i = NZ_p( j )
       data%gtp = data%gtp + P( i ) * G( i )
       ptbp = ptbp + P( i ) * HP( i )

!  Find the breakpoints for the piecewise linear arc (the distances to the
!  boundary)

       IF ( P( i ) > zero ) THEN
         data%BREAKP( i ) = ( X_u( i ) - X_0( i ) ) / P( i )
       ELSE
         data%BREAKP( i ) = ( X_l( i ) - X_0( i ) ) / P( i )
       END IF

!  Compute the maximum feasible distance, TBREAK, allowable in the direction
!  P. Also compute TBMAX, the largest breakpoint

       data%tbreak = MIN( data%tbreak, data%BREAKP( i ) )
       tbmax = MAX( tbmax, data%BREAKP( i ) )
     END DO

!  Check that the curvature is positive

     IF ( ptbp > zero ) THEN

!  Compute the minimizer, T, of quad(  X( T )  ) in the direction P

       data%t = - data%gtp / ptbp

!  Compare the values of T and TBREAK. If the calculated minimizer is the
!  arc miminizer, exit

       IF ( data%t <= data%tbreak ) THEN
         GO TO 500
       ELSE
         IF ( data%pronel .OR. data%prnter ) WRITE( out,                       &
         "( A, 21X, I6, '  1st line mimimizer infeasible. Step = ', ES10.2 )" )&
             prefix, data%iterca, data%t

!  Ensure that the initial value of T for backtracking is no larger than
!  ALPHA_SEARCH times the step to the first line minimum

         data%tamax = MIN( t_max, alpha_search * data%t )
       END IF

!  -----------------------
!  The remaining intervals
!  -----------------------

!  The calculated minimizer is infeasible; prepare to backtrack from T until
!  an approximate arc minimizer is found

       data%t = MIN( data%tamax, tbmax )
     ELSE
       data%t = tbmax
     END IF

!  Calculate p, the difference between the projection of the point
!  x(t) and X_0, and PTP, the square of the norm of this distance

!DIR$ IVDEP
     DO j = 1, n_free
       i = NZ_p( j )
       P( i ) = MAX( MIN( X_0( i ) + data%t * P( i ), X_u( i ) ), X_l( i ) )   &
                - X_0( i )
     END DO

!  Calculate HP = H * P ...

     IF ( data%explicit_h ) THEN
       HP = zero
       DO l = nz_p_start, nz_p_end
         i = NZ_p( l ) ; p_i = P( i )
         DO k = H%ptr( i ), H%ptr( i + 1 ) - 1
           HP( H%col( k ) ) = HP( H%col( k ) ) + H%val( k ) * p_i
         END DO
       END DO

!  ... or obtain the product from a user-provided subroutine ...

     ELSE IF ( data%use_hprod ) THEN
       CALL eval_HPROD( i, userdata, P, HP, NZ_v = NZ_p,                       &
                        nz_v_start = nz_p_start, nz_v_end = nz_p_end )

!  ... or return to the calling program to calculate HP = H * P

     ELSE
       data%branch = 300 ; status = 2 ; RETURN
     END IF

 300 CONTINUE

!  Compute the slope and curvature of q(x) in the direction p

!        data%gtp  = DOT_PRODUCT( P( NZ_p( : n_free ) ), G( NZ_p( : n_free ) ) )
!        ptbp = DOT_PRODUCT( P( NZ_p( : n_free ) ), HP( NZ_p( : n_free ) ) )
     data%gtp = zero ; ptbp = zero
     DO i = 1, n_free
        data%gtp = data%gtp + G( NZ_p( i ) ) * P( NZ_p( i ) )
        ptbp = ptbp + HP( NZ_p( i ) ) * P( NZ_p( i ) )
     END DO

!  Form the gradient at the point x(t)

     data%GRAD( NZ_p( : n_free ) )                                             &
       = HP( NZ_p( : n_free ) ) + G( NZ_p( : n_free ) )

!  Evaluate q(x(t)) and linear(x(t),mu)

     q_t = f + data%gtp + half * ptbp ; data%flxt = f + mu * data%gtp

!  --------------------------------
!  Start of the main iteration loop
!  --------------------------------

 350 CONTINUE
       data%iterca = data%iterca + 1

!  Print details of the current point

       IF ( data%pronel .OR. data%prnter ) WRITE( out,                        &
         "( A, 21X, I6, 3ES12.4)" ) prefix, data%iterca, data%t, q_t, data%flxt

!  Compare q(x(t)) with linear(x(t),mu). If x(t) satisfies the
!  Armijo condition and thus qualifies as an approximate arch mimizer, exit

       IF ( data%iterca > data%itmax .OR. q_t <= data%flxt ) THEN
!DIR$ IVDEP
         DO j = 1, n_free
           i = NZ_p( j )
           X_t( i ) = X_0( i ) + MIN( data%t, data%BREAKP( i ) ) * X_t( i )
         END DO
         GO TO 600
       END IF

!  x(t) is not acceptable. Reduce t

       data%t_new = beta_search * data%t

!  Compute p = x(t_new) - x(t)

!DIR$ IVDEP
       DO j = 1, n_free
         i = NZ_p( j )
         P( i ) = ( MIN( data%t_new,                                           &
             data%BREAKP( i ) ) - MIN( data%t, data%BREAKP( i ) ) ) * X_t( i )
       END DO

!  Calculate HP = H * P ...

       IF ( data%explicit_h ) THEN
         HP = zero
         DO l = nz_p_start, nz_p_end
           i = NZ_p( l ) ; p_i = P( i )
           DO k = H%ptr( i ), H%ptr( i + 1 ) - 1
             HP( H%col( k ) ) = HP( H%col( k ) ) + H%val( k ) * p_i
           END DO
         END DO

!  ... or obtain the product from a user-provided subroutine ...

       ELSE IF ( data%use_hprod ) THEN
         CALL eval_HPROD( i, userdata, P, HP, NZ_v = NZ_p,                     &
                        nz_v_start = nz_p_start, nz_v_end = nz_p_end )

!  ... or return to the calling program to calculate HP = H * P

       ELSE
         data%branch = 400 ; status = 2
         RETURN
       END IF

 400   CONTINUE

!  Compute the slope and curvature of quad( X ) in the direction P

       data%g0tp = zero ; data%gtp = zero ; ptbp = zero
       DO i = 1, n_free
         data%g0tp = data%g0tp + G( NZ_p( i ) ) * P( NZ_p( i ) )
         data%gtp  = data%gtp  + data%GRAD( NZ_p( i ) ) * P( NZ_p( i ) )
         ptbp = ptbp + HP( NZ_p( i ) ) * P( NZ_p( i ) )
       END DO

!  Update the existing gradient to find that at the point X( t_new )

       data%GRAD( NZ_p( : n_free ) )                                           &
         = data%GRAD( NZ_p( : n_free ) ) + HP( NZ_p( : n_free ) )

!  Evaluate quad( X( T ) ) and linear( X( T ), MU )

       q_t = q_t + data%gtp + half * ptbp
       data%flxt = data%flxt + mu * data%g0tp

       data%t = data%t_new
     GO TO 350

!  ------------------------------
!  End of the main iteration loop
!  ------------------------------

 500 CONTINUE

!  The arc minimizer occured in the first interval. Record the point and the
!  value of the quadratic at the point

     q_t = f + data%t * ( data%gtp + half * data%t * ptbp )
     X_t( NZ_p( : n_free ) )                                                   &
       = X_0( NZ_p( : n_free ) ) + data%t * P( NZ_p( : n_free ) )

!  Print details of the arc minimizer

     IF ( data%pronel .OR. data%prnter )                                       &
       WRITE( out, "( A, 21X, I6, 2ES12.4 )" ) prefix, data%iterca, data%t, q_t

!  An approximation to the arc minimizer has been found. Set the
!  array P to the step from the initial point to the approximate minimizer

 600 CONTINUE
     n3 = 0
!DIR$ IVDEP
     DO j = 1, n_free
       i = NZ_p( j )
       P( i ) = X_t( i ) - X_0( i )

!  Find which variables are free at X( T )

       IF ( data%t <= data%BREAKP( i ) ) THEN
         n3 = n3 + 1
       ELSE
         IF ( P( i ) < zero ) VARIABLE_status( i ) = 1
         IF ( P( i ) > zero ) VARIABLE_status( i ) = 2

!  Move the fixed variables to their bounds

         IF ( P( i ) /= zero ) THEN
           IF ( VARIABLE_status( i ) == 1 ) THEN
             X_t( i ) = X_l( i )
           ELSE
             X_t( i ) = X_u( i )
           END IF
         END IF
       END IF
     END DO
     IF ( data%pronel ) WRITE( out, "( A, 27X, I6, ' variables are fixed ' )" )&
       prefix, data%nzero - n3 - 1

!  Record that variables whose gradients were zero at the initial point are
!  free variables

     DO j = data%nzero, n
       n_free = n_free + 1
       NZ_p( n_free ) = NZ_p( j )
     END DO

!  Set return conditions

     nz_p_start = 1 ; nz_p_end = n_free ; status = 0
     RETURN

!  End of subroutine BQP_inexact_arcsearch

     END SUBROUTINE BQP_inexact_arcsearch

!  End of module BQP

   END MODULE GALAHAD_BQP_double
