! THIS VERSION: GALAHAD 2.6 - 15/10/2014 AT 13:20 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ B Q P B   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.4. January 1st 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BQPB_double

!     ---------------------------------------------------------------------
!     |                                                                   |
!     | Solve the convex bound-constrained quadratic program              |
!     |                                                                   |
!     |    minimize     1/2 x(T) H x + g(T) x + f                         |
!     |    subject to     x_l <=  x  <= x_u                               |
!     |                                                                   |
!     | using a preconditioned conjugate-gradient interior-point approach |
!     |                                                                   |
!     ---------------------------------------------------------------------

     USE GALAHAD_SYMBOLS
     USE GALAHAD_STRING_double, ONLY: STRING_integer_6, STRING_real_7
     USE GALAHAD_SPACE_double
     USE GALAHAD_SORT_double, ONLY : SORT_heapsort_build, SORT_heapsort_smallest
     USE GALAHAD_SBLS_double
     USE GALAHAD_QPT_double
     USE GALAHAD_QPP_double
     USE GALAHAD_QPD_double, ONLY: QPD_SIF
     USE GALAHAD_ROOTS_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: BQPB_initialize, BQPB_read_specfile, BQPB_solve,                &
               BQPB_terminate, BQPB_reverse_type, BQPB_data_type,              &
               NLPT_userdata_type, QPT_problem_type,                           &
               SMT_type, SMT_put, SMT_get

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
     REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
     REAL ( KIND = wp ), PARAMETER :: decrease_mu = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: sigma = 0.01_wp
     REAL ( KIND = wp ), PARAMETER :: gamma_b0 = ten ** ( - 5 )
     REAL ( KIND = wp ), PARAMETER :: gamma_f0 = ten ** ( - 5 )
     LOGICAL :: roots_debug = .FALSE.

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: BQPB_control_type
        
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

!  how many CG iterations to perform per BQPB iteration (-ve reverts to n+1)

       INTEGER :: cg_maxit = 1000

!  which stepsize strategy to use (1=step-to-boundary rule,2=barrier,3=Zhang)

       INTEGER :: stepsize_strategy = 3

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

!  the initial barrrier parameter

       REAL ( KIND = wp ) :: mu_zero = - one

!  initial primal variables will not be closer than this from their bounds 

       REAL ( KIND = wp ) :: pr_feas = one

!  initial dual variables will not be closer than this from their bounds 

       REAL ( KIND = wp ) :: du_feas = one

!  the maximum step that will be taken towards the constraint boundary 

       REAL ( KIND = wp ) :: fraction_to_the_boundary = 0.99_wp

!  the CG iteration will be stopped as soon as the current norm of the 
!  preconditioned gradient is smaller than 
!    max( stop_cg_relative * initial preconditioned gradient, stop_cg_absolute )

       REAL ( KIND = wp ) :: stop_cg_relative = ten ** ( - 2 )
       REAL ( KIND = wp ) :: stop_cg_absolute = epsmch

!  threshold below which curvature is regarded as zero

       REAL ( KIND = wp ) :: zero_curvature = ten * epsmch

!  the maximum CPU time allowed (-ve = no limit)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!  use the primal-dual barrier method (preferred) or the primal one

       LOGICAL :: primal_dual = .TRUE.

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
         "BQPBPROB.SIF"  // REPEAT( ' ', 18 )

!  all output lines will be prefixed by a string (max 30 characters)
!    prefix(2:LEN(TRIM(%prefix))-1)
!   where prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'
!
       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SBLS

       TYPE ( SBLS_control_type ) :: SBLS_control
     END TYPE BQPB_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BQPB_time_type

!  total time

       REAL :: total = 0.0

!  time for the analysis phase

       REAL :: analyse = 0.0

!  time for the factorization phase

       REAL :: factorize = 0.0

!  time for the linear solution phase

       REAL :: solve = 0.0
     END TYPE BQPB_time_type
   
!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: BQPB_inform_type

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

!  current value of the complemtary slackness

       REAL ( KIND = wp ) :: slknes = infinity

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  times for various stages

       TYPE ( BQPB_time_type ) :: time

!  inform values from SBLS

       TYPE ( SBLS_inform_type ) :: SBLS_inform
     END TYPE BQPB_inform_type

!  - - - - - - - - - - -
!   reverse derived type
!  - - - - - - - - - - -

     TYPE :: BQPB_reverse_type
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V, PROD
     END TYPE BQPB_reverse_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE :: BQPB_data_type
       INTEGER :: out, error, print_level, start_print, stop_print, print_gap
       INTEGER :: n_free, branch, cg_iter, maxit, cg_maxit
       REAL :: time_start
       REAL ( KIND = wp ) :: q_t, norm_step, step, stop_cg, old_gnrmsq, pnrmsq
       REAL ( KIND = wp ) :: curvature, mu, nu, alpha, gamma_b, gamma_f
       LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
       LOGICAL :: set_printm, printt, printi, printm, printw, printd, printe 
       LOGICAL :: reverse, explicit_h, use_hprod, header
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: FREE
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_new, G, V, PROD
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_l, Z_u, H_diag
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ_l, DZ_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX, PREC, P_cg
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_cg, PG_cg
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HP_cg
       TYPE ( SMT_type ) :: H
       TYPE ( SBLS_data_type ) :: SBLS_data
     END TYPE BQPB_data_type

   CONTAINS

!-*-*-*-*-*-   B Q P B _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE BQPB_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for BQPB. This routine should be called before
!  BQPB_solve
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

     TYPE ( BQPB_data_type ), INTENT( INOUT ) :: data
     TYPE ( BQPB_control_type ), INTENT( OUT ) :: control
     TYPE ( BQPB_inform_type ), INTENT( OUT ) :: inform     

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

!  End of BQPB_initialize

     END SUBROUTINE BQPB_initialize

!-*-*-*-*-   B Q P B _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

     SUBROUTINE BQPB_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by BQPB_initialize could (roughly) 
!  have been set as:

! BEGIN BQPB SPECIFICATIONS (DEFAULT)
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
!  stepsize-strategy                                 3
!  sif-file-device                                   52
!  infinity-value                                    1.0D+19
!  primal-accuracy-required                          1.0D-5
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  identical-bounds-tolerance                        1.0D-15
!  mininum-initial-primal-feasibility                1.0
!  mininum-initial-dual-feasibility                  1.0
!  initial-barrier-parameter                         -1.0
!  cg-relative-accuracy-required                     0.01
!  cg-absolute-accuracy-required                     1.0D-8
!  zero-curvature-threshold                          1.0D-15
!  fraction-to-the-boundary-step-allowed             0.99
!  maximum-cpu-time-limit                            -1.0
!  primal-dual-barrier-used                          T
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  sif-file-name                                     BQPBPROB.SIF
!  output-line-prefix                                ""
! END BQPB SPECIFICATIONS

!  Dummy arguments

     TYPE ( BQPB_control_type ), INTENT( INOUT ) :: control        
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
     INTEGER, PARAMETER :: stepsize_strategy = cg_maxit + 1
     INTEGER, PARAMETER :: infinity = stepsize_strategy + 1
     INTEGER, PARAMETER :: stop_p = infinity + 1
     INTEGER, PARAMETER :: stop_d = stop_p + 1
     INTEGER, PARAMETER :: stop_c = stop_d + 1
     INTEGER, PARAMETER :: identical_bounds_tol = stop_c + 1
     INTEGER, PARAMETER :: pr_feas = identical_bounds_tol + 1
     INTEGER, PARAMETER :: du_feas = pr_feas + 1
     INTEGER, PARAMETER :: mu_zero = du_feas + 1
     INTEGER, PARAMETER :: stop_cg_relative = mu_zero + 1
     INTEGER, PARAMETER :: stop_cg_absolute = stop_cg_relative + 1
     INTEGER, PARAMETER :: fraction_to_the_boundary = stop_cg_absolute + 1
     INTEGER, PARAMETER :: zero_curvature = fraction_to_the_boundary + 1
     INTEGER, PARAMETER :: cpu_time_limit = zero_curvature + 1
     INTEGER, PARAMETER :: primal_dual = cpu_time_limit + 1
     INTEGER, PARAMETER :: space_critical = primal_dual + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: sif_file_name = generate_sif_file + 1
     INTEGER, PARAMETER :: prefix = sif_file_name + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'BQPB'
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
     spec( stepsize_strategy )%keyword = 'stepsize-strategy'
     spec( sif_file_device )%keyword = 'sif-file-device'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( stop_p )%keyword = 'primal-accuracy-required'
     spec( stop_d )%keyword = 'dual-accuracy-required'
     spec( stop_c )%keyword = 'complementary-slackness-accuracy-required'
     spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
     spec( pr_feas )%keyword = 'mininum-initial-primal-feasibility'
     spec( du_feas )%keyword = 'mininum-initial-dual-feasibility'
     spec( mu_zero )%keyword = 'initial-barrier-parameter'
     spec( stop_cg_relative )%keyword = 'cg-relative-accuracy-required'
     spec( stop_cg_absolute )%keyword = 'cg-absolute-accuracy-required'
     spec( fraction_to_the_boundary )%keyword =                                &
       'fraction-to-the-boundary-step-allowed'
     spec( zero_curvature )%keyword = 'zero-curvature-threshold'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'

!  Logical key-words

     spec( primal_dual )%keyword = 'primal-dual-barrier-used'
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
     CALL SPECFILE_assign_value( spec( stepsize_strategy ),                    &
                                 control%stepsize_strategy,                    &
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
     CALL SPECFILE_assign_value( spec( pr_feas ),                              &
                                 control%pr_feas,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( du_feas ),                              &
                                 control%du_feas,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mu_zero ),                              &
                                 control%mu_zero,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_cg_relative ),                     &
                                 control%stop_cg_relative,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_cg_absolute ),                     &
                                 control%stop_cg_absolute,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( fraction_to_the_boundary ),             &
                                 control%fraction_to_the_boundary,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( zero_curvature ),                       &
                                 control%zero_curvature,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( primal_dual ),                          &
                                 control%primal_dual,                          &
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

     END SUBROUTINE BQPB_read_specfile

!-*-*-*-*-*-*-*-   B Q P B _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-

     SUBROUTINE BQPB_solve( prob, B_stat, data, control, inform,               &
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
!  using a preconditioned CG interior-point method.
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
!   on entry to BQPB_solve to indicate which of the simple bound constraints 
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
!  data is a structure of type BQPB_data_type which holds private internal data
!
!  control is a structure of type BQPB_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to BQPB_initialize. See BQPB_initialize 
!   for details
!
!  inform is a structure of type BQPB_inform_type that provides 
!    information on exit from BQPB_solve. The component %status 
!    must be set to 1 on initial entry, and on exit has possible values:
!  
!     0 Normal termination with a locally optimal solution.
!
!     2 The product H * v of the Hessian H with a given output vector v
!       is required from the user. The vector v will be stored in reverse%V
!       and the product H * v must be returned in reverse%PROD, and 
!       BQPB_solve re-entered with all other arguments unchanged. 
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
!                         'DIAGONAL' }
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
!  On exit from BQPB_solve, other components of inform give the 
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
!       solution determined by BQPB_solve.
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
!  reverse is an OPTIONAL structure of type BQPB_reverse_type which is used to 
!   pass intermediate data to and from BQPB_solve. This will only be necessary 
!   if reverse-communication is to be used to form matrix-vector products
!   of the form H * v or preconditioning steps of the form P * v. If 
!   reverse is present (and eval_HPROD is absent), reverse communication
!   will be used and the user must monitor the value of inform%status 
!   (see above) to await instructions about required matrix-vector products.
!
!  eval_HPROD is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product H * v of the given 
!   matrix H and vector v stored in V must be returned in PROD. The status 
!   variable should be set to 0 unless the product is impossible in which
!   case status should be set to a nonzero value. If eval_HPROD is not
!   present, BQPB_solve will either return to the user each time an evaluation 
!   is required (see reverse above) or form the product directly from 
!   user-provided %H.
!
!  eval_PREC is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product P * v of the given 
!   preconditioner P and vector v stored in V must be returned in PV.
!   The status variable should be set to 0 unless the product is impossible 
!   in which case status should be set to a nonzero value. If eval_PREC
!   is not present, BQPB_solve will return to the user each time an evaluation 
!   is required.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     INTEGER, INTENT( INOUT ), DIMENSION( prob%n ) :: B_stat
     TYPE ( BQPB_data_type ), INTENT( INOUT ) :: data
     TYPE ( BQPB_control_type ), INTENT( IN ) :: control
     TYPE ( BQPB_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     TYPE ( BQPB_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse
     OPTIONAL :: eval_HPROD
!    OPTIONAL :: eval_HPROD, eval_PREC

!  interface blocks

     INTERFACE
       SUBROUTINE eval_HPROD( status, userdata, V, PROD )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: PROD
       END SUBROUTINE eval_HPROD
     END INTERFACE
   
!    INTERFACE
!      SUBROUTINE eval_PREC( status, userdata, V, PREC )
!      USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
!      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
!      INTEGER, INTENT( OUT ) :: status
!      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
!      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
!      REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: PREC
!      END SUBROUTINE eval_PREC
!    END INTERFACE

!  Local variables

     INTEGER :: i, j, k, l, nnz
     REAL :: time
     REAL ( KIND = wp ) :: val, av_bnd, x_i, p_i, curvature
     REAL ( KIND = wp ) :: gnrmsq, beta, alpha_x, alpha_z, pr_feas, du_feas
     REAL ( KIND = wp ) :: gl, norm_gl, slkmin, cs, nc
     REAL ( KIND = wp ) :: alpha_b, alpha_f
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
!  that are "essentially" the same, and record the number of unfixed variables

     reset_bnd = .FALSE.
     data%n_free = 0
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
       ELSE
         data%n_free = data%n_free + 1
         IF ( control%cold_start == 0 ) THEN
           IF ( B_stat( i ) < 0 ) THEN
              prob%X_l( i ) =  prob%X_l( i )
             reset_bnd = .TRUE.
           ELSE IF ( B_stat( i ) > 0 ) THEN
              prob%X_l( i ) =  prob%X_u( i )
             reset_bnd = .TRUE.
           END IF
         END IF
       END IF
     END DO   
     IF ( reset_bnd .AND. data%printi ) WRITE( control%out,                    &
       "( ' ', /, A, '   **  Warning: one or more variable bounds reset ' )" ) &
         prefix

!  allocate workspace arrays

     array_name = 'bqpb: data%FREE'
     CALL SPACE_resize_array( data%n_free, data%FREE, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%G'
     CALL SPACE_resize_array( prob%n, data%G, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%Z_l'
     CALL SPACE_resize_array( prob%n, data%Z_l, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%Z_u'
     CALL SPACE_resize_array( prob%n, data%Z_u, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%DX'
     CALL SPACE_resize_array( prob%n, data%DX, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%DZ_l'
     CALL SPACE_resize_array( prob%n, data%DZ_l, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%DZ_u'
     CALL SPACE_resize_array( prob%n, data%DZ_u, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%PREC'
     CALL SPACE_resize_array( prob%n, data%PREC, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%H_diag'
     CALL SPACE_resize_array( prob%n, data%H_diag, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%G_cg'
     CALL SPACE_resize_array( prob%n, data%G_cg, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%PG_cg'
     CALL SPACE_resize_array( prob%n, data%PG_cg, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%P_cg'
     CALL SPACE_resize_array( prob%n, data%P_cg, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%HP_cg'
     CALL SPACE_resize_array( prob%n, data%HP_cg, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'bqpb: data%X_new'
     CALL SPACE_resize_array( prob%n, data%X_new, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     IF ( data%reverse ) THEN
       array_name = 'bqpb: reverse%V'
       CALL SPACE_resize_array( prob%n, reverse%V, inform%status,              &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqpb: reverse%PROD'
       CALL SPACE_resize_array( prob%n, reverse%PROD, inform%status,           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     ELSE
       array_name = 'bqpb: data%V'
       CALL SPACE_resize_array( prob%n, data%V, inform%status,                 &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqpb: data%PROD'
       CALL SPACE_resize_array( prob%n, data%PROD, inform%status,              &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

!  record the indices of unfixed variables

     data%n_free = 0
     DO i = 1, prob%n
       IF ( prob%X_u( i ) /= prob%X_l( i ) ) THEN
         data%n_free = data%n_free + 1
         data%FREE( data%n_free ) = i
       ELSE
         data%G_cg( i ) = zero ; data%PG_cg( i ) = zero ; data%HP_cg( i ) = zero
         data%H_diag( i ) = zero
       END IF
     END DO   

!  Build a copy of H stored by rows (both lower and upper triangles)

 150 CONTINUE
     data%header = .TRUE.
     IF ( data%explicit_h ) THEN

!  allocate space to record row lengths

       array_name = 'bqpb: data%H%ptr'
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

       array_name = 'bqpb: data%H%col'
       CALL SPACE_resize_array( nnz, data%H%col, inform%status,                &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'bqpb: data%H%val'
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

!  if necessary, perturb the starting point so that it is sufficiently
!  feasible with respect to the simple bound constraints

     pr_feas = MAX( control%pr_feas, epsmch )
     du_feas = MAX( control%du_feas, epsmch )
     DO l = 1, data%n_free
       i = data%FREE( l )
       IF ( prob%X_l( i ) >= - control%infinity ) THEN
         IF ( prob%X_u( i ) <= control%infinity ) THEN
           IF ( prob%X_u( i ) - prob%X_l( i ) < two * pr_feas ) THEN
             prob%X( i ) = half * ( prob%X_l( i ) + prob%X_u( i ) )
           ELSE
             prob%X( i ) = MIN( MAX( prob%X_l( i ) + pr_feas, prob%X( i ) ),  &
                                     prob%X_u( i ) - pr_feas )
           END IF
           data%Z_l( i ) = MAX( prob%Z( i ), du_feas )
           data%Z_u( i ) = MAX( - prob%Z( i ), du_feas )
         ELSE
           prob%X( i ) = MAX( prob%X( i ), prob%X_l( i ) + pr_feas )
           data%Z_l( i ) = MAX( prob%Z( i ), du_feas )
           data%Z_u( i ) = zero
         END IF
       ELSE IF ( prob%X_u( i ) <= control%infinity ) THEN
         prob%X( i ) = MIN( prob%X( i ), prob%X_u( i ) - pr_feas )
         data%Z_l( i ) = zero
         data%Z_u( i ) = MAX( - prob%Z( i ), du_feas )
       ELSE
         data%Z_l( i ) = zero
         data%Z_u( i ) = zero
       END IF
     END DO
     data%DZ_l = zero ; data%DZ_u = zero

!  select the initial barrier parameter

     IF ( control%stepsize_strategy == 1 .OR.                                  &
          control%stepsize_strategy == 2 ) THEN
       IF ( control%mu_zero <= zero ) THEN
         data%mu = one
       ELSE
         data%mu = control%mu_zero
       END IF
     ELSE
       IF ( control%mu_zero <= zero ) THEN
         inform%slknes = zero ; nc = zero
         DO l = 1, data%n_free
           i = data%FREE( l )
           IF ( prob%X_l( i ) >= - control%infinity ) THEN
             IF ( prob%X_u( i ) <= control%infinity ) THEN
               cs = ( prob%X( i ) - prob%X_l( i ) ) * data%Z_l( i ) +          &
                    ( prob%X_u( i ) - prob%X( i ) ) * data%Z_u( i )
               nc = nc + two
             ELSE
               cs = ( prob%X( i ) - prob%X_l( i ) ) * data%Z_l( i )
               nc = nc + one
             END IF
           ELSE IF ( prob%X_u( i ) <= control%infinity ) THEN
             cs = ( prob%X_u( i ) - prob%X( i ) ) * data%Z_u( i )
             nc = nc + one
           ELSE
             cs = zero
           END IF
           inform%slknes = inform%slknes + cs
         END DO
         data%mu = sigma * inform%slknes / nc
       ELSE
         data%mu = control%mu_zero
       END IF
     END IF
     data%nu = one

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
         data%G = prob%G + data%PROD
       ELSE
         data%G = prob%G + reverse%PROD
       END IF
       prob%Z = data%G

!  compute the objective function

       inform%obj = half * ( DOT_PRODUCT( data%G, prob%X ) +                   &
                             DOT_PRODUCT( prob%G, prob%X ) ) + prob%f

!  compute the norm of the projected gradient

       inform%norm_pg = MAXVAL( ABS( MAX( MIN( prob%X - data%G, prob%X_u ),    &
                                          prob%X_l ) - prob%X ) )

!  print details of the current iteration

       IF ( ( data%printi .AND. data%header ) .OR. data%printt )               &
         WRITE( data%out, "( /, A, '  #its   #cg            f          ',      &
        &       ' proj gr  cmp slkns    step     mu      time' )" ) prefix
       data%header = .FALSE.
       IF ( data%printi ) THEN
         IF ( inform%iter > 1 ) THEN
           WRITE( data%out, "( A, 2A6, ES22.14, 3ES10.3, ES8.1, A7 )" )        &
             prefix, STRING_integer_6( inform%iter ),                          &
             STRING_integer_6( data%cg_iter ),                                 &
             inform%obj, inform%norm_pg, inform%slknes, data%alpha, data%mu,   &
             STRING_real_7( inform%time%total )
         ELSE IF ( inform%iter == 1 ) THEN
           WRITE( data%out, "( A, 2A6, ES22.14, 3ES10.3, ES8.1, A7 )" )        &
             prefix, STRING_integer_6( inform%iter ),                          &
             STRING_integer_6( data%cg_iter ),                                 &
             inform%obj, inform%norm_pg, inform%slknes, data%alpha, data%mu,   &
             STRING_real_7( inform%time%total )
         ELSE
           WRITE( data%out, "( A, A6, '     -', ES22.14, 2ES10.3,              &
          & '      -   ', ES8.1, A7 )" )                                       &
             prefix, STRING_integer_6( inform%iter ),                          &
             inform%obj, inform%norm_pg, inform%slknes, data%mu,               &
             STRING_real_7( inform%time%total )
         END IF
       END IF

       IF ( data%printw ) THEN
         WRITE( data%out, "( A, '       x_l         x           x_u ',         &
        &                       '       z_l         z_u ' )" ) prefix
         DO i = 1, prob%n
           WRITE( data%out, "( A, 5ES12.4 )" )                                 &
             prefix, prob%X_l( i ), prob%X( i ), prob%X_u( i ), data%Z_l( i ), &
             data%Z_u( i )
         END DO
       END IF

!  compute the norm of the gradient of the Lagrangian and the complementarity 

       norm_gl = zero ; inform%slknes = zero ; nc = zero ; slkmin = infinity
       DO l = 1, data%n_free
         i = data%FREE( l )
         IF ( prob%X_l( i ) >= - control%infinity ) THEN
           IF ( prob%X_u( i ) <= control%infinity ) THEN
             gl = data%G( i ) - data%Z_l( i ) + data%Z_u( i ) 
             cs = ( prob%X( i ) - prob%X_l( i ) ) * data%Z_l( i ) +           &
                  ( prob%X_u( i ) - prob%X( i ) ) * data%Z_u( i )
             slkmin = MIN( slkmin,                                            &
                ( prob%X( i ) - prob%X_l( i ) ) * data%Z_l( i ),              &
                ( prob%X_u( i ) - prob%X( i ) ) * data%Z_u( i ) )
             nc = nc + two
           ELSE
             gl = data%G( i ) - data%Z_l( i )
             cs = ( prob%X( i ) - prob%X_l( i ) ) * data%Z_l( i )
             slkmin = MIN( slkmin, cs )
             nc = nc + one
           END IF
         ELSE IF ( prob%X_u( i ) <= control%infinity ) THEN
           gl = data%G( i ) + data%Z_u( i ) 
           cs = ( prob%X_u( i ) - prob%X( i ) ) * data%Z_u( i )
           slkmin = MIN( slkmin, cs )
           nc = nc + one
         ELSE
           gl = data%G( i )
           cs = zero
         END IF
         norm_gl = MAX( norm_gl, ABS( gl ) )
         inform%slknes = inform%slknes + cs
       END DO
       IF ( nc > zero ) THEN 
         data%gamma_f = gamma_f0 * inform%slknes
         inform%slknes = inform%slknes / nc
         data%gamma_b = gamma_b0 * slkmin / inform%slknes
       ELSE
         data%gamma_f = zero ; data%gamma_b = zero
       END IF

!  test for an approximate first-order critical point based on the Lagrangian
!  function and complementarity

       IF ( norm_gl <= control%stop_d .AND.                                    &
            inform%slknes <= control%stop_c ) THEN
         IF ( data%printi ) WRITE( data%out,                                   &
           "( /, A, ' Exit with Lagrangian gradient = ', ES10.4,               &
          & ' and complementarity = ', ES10.4 )" )                             &
              prefix, norm_gl, inform%slknes
         inform%status = GALAHAD_ok ; GO TO 910
       END IF

!  test for an approximate first-order critical point based on the projected
!  gradient

       IF ( inform%norm_pg <= control%stop_d .AND.                             &
            inform%slknes <= control%stop_c ) THEN
         IF ( data%printi ) WRITE( data%out,                                   &
           "( /, A, ' Exit with projected gradient = ', ES10.4,                &
          & ' and complementarity = ', ES10.4 )" )                             &
              prefix, inform%norm_pg, inform%slknes
         inform%status = GALAHAD_ok ; GO TO 910
       END IF

!  record the gradient, g_cg, of the barrier function

       DO
         DO l = 1, data%n_free
           i = data%FREE( l )
           IF ( prob%X_l( i ) >= - control%infinity ) THEN
             IF ( prob%X_u( i ) <= control%infinity ) THEN
               data%G_cg( i ) = data%G( i )                                    &
                 - data%mu / ( prob%X( i ) - prob%X_l( i ) )                   &
                 + data%mu / ( prob%X_u( i ) - prob%X( i ) )
             ELSE
               data%G_cg( i ) = data%G( i )                                    &
                 - data%mu / ( prob%X( i ) - prob%X_l( i ) )
             END IF
           ELSE IF ( prob%X_u( i ) <= control%infinity ) THEN
             data%G_cg( i ) = data%G( i )                                      &
               + data%mu / ( prob%X_u( i ) - prob%X( i ) )
           ELSE
             data%G_cg( i ) = data%G( i )
           END IF
         END DO

         IF ( control%stepsize_strategy == 3 ) THEN
           data%mu = sigma * inform%slknes ; EXIT
         END IF

!  test to see if the barrier parameter should be decreased

         norm_gl = MAXVAL( ABS( data%G_cg(  data%FREE( : data%n_free ) ) ) )
         IF ( norm_gl > data%mu ) EXIT

!  test for an approximate first-order critical point based on the barrier
!  function

         IF ( data%mu <= control%stop_d ) THEN
         IF ( data%printi ) WRITE( data%out,                                   &
           "( /, A, ' Exit with barrier gradient = ', ES10.4 )" )              &
              prefix, norm_gl
           inform%status = GALAHAD_ok ; GO TO 910
         END IF

!  reduce the barrier parameter

         data%mu = decrease_mu * data%mu
         IF ( data%printm ) WRITE( control%out,                                &
           "( A, ' mu reduced to', ES11.2 )" ) prefix, data%mu
       END DO

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

!  record the Hessian weights

       IF ( control%primal_dual ) THEN
         DO l = 1, data%n_free
           i = data%FREE( l )
           IF ( prob%X_l( i ) >= - control%infinity ) THEN
             IF ( prob%X_u( i ) <= control%infinity ) THEN
               data%H_diag( i ) =                                              &
                 data%Z_l( i ) / ( prob%X( i ) - prob%X_l( i ) )               &
                 + data%Z_u( i ) / ( prob%X_u( i ) - prob%X( i ) )
             ELSE
               data%H_diag( i ) =                                              &
                 data%Z_l( i ) / ( prob%X( i ) - prob%X_l( i ) )
             END IF
           ELSE IF ( prob%X_u( i ) <= control%infinity ) THEN
             data%H_diag( i ) =                                                &
               data%Z_u( i ) / ( prob%X_u( i ) - prob%X( i ) )
           ELSE
             data%H_diag( i ) = zero
           END IF
         END DO
       ELSE
         DO l = 1, data%n_free
           i = data%FREE( l )
           IF ( prob%X_l( i ) >= - control%infinity ) THEN
             IF ( prob%X_u( i ) <= control%infinity ) THEN
               data%H_diag( i ) =                                              &
                 data%mu / ( prob%X( i ) - prob%X_l( i ) ) ** 2                &
                 + data%mu / ( prob%X_u( i ) - prob%X( i ) ) ** 2
             ELSE
               data%H_diag( i ) =                                              &
                 data%mu / ( prob%X( i ) - prob%X_l( i ) ) ** 2
             END IF
           ELSE IF ( prob%X_u( i ) <= control%infinity ) THEN
             data%H_diag( i ) =                                                &
               data%mu / ( prob%X_u( i ) - prob%X( i ) ) ** 2
           ELSE
             data%H_diag( i ) = zero
           END IF
         END DO
       END IF

!  record the diagonal preconditioner

       data%PREC( : prob%n ) = MAX( one, data%H_diag( : prob%n ) )

!  perform CG from x, starting from dx = 0 

       data%DX( : prob%n ) = zero
       data%pnrmsq = zero
       data%q_t = inform%obj

!  - - - - - - - - - -
!  Start the CG loop
!  - - - - - - - - - -

 210     CONTINUE

!  obtain the preconditioned gradient pg_cg = P(inverse) g_cg

         IF ( data%n_free == prob%n ) THEN
           data%PG_cg( : prob%n ) =                                            &
             data%G_cg( : prob%n ) / data%PREC( : prob%n )
         ELSE
           data%PG_cg( data%FREE( : data%n_free ) ) =                          &
             data%G_cg( data%FREE( : data%n_free ) ) /                         &
               data%PREC( data%FREE( : data%n_free ) ) 
         END IF
 300     CONTINUE
         gnrmsq = DOT_PRODUCT( data%PG_cg( : prob%n ), data%G_cg( : prob%n ) )

!  compute the CG stopping tolerance

         IF (  data%cg_iter == 0 ) THEN
           data%stop_cg = MIN( 0.1_wp,                                         &
!                              data%mu )
               MAX( SQRT( ABS( gnrmsq ) ) * control%stop_cg_relative,          &
                                            control%stop_cg_absolute ) )
         END IF

!  print details of the current iteration

         IF ( data%printm ) THEN
           IF ( data%cg_iter == 0 ) THEN
             WRITE( control%out, "( /, A, ' ** CG entered ** ',                &
            &    /, A, '    required gradient =', ES8.1, /, A,                 &
            &    '    iter     model    proj grad    curvature     step')" )   &
             prefix, prefix, data%stop_cg, prefix
             WRITE( control%out,                                               &
               "( A, 1X, I7, 2ES12.4, '      -            -     ' )" )         &
               prefix, data%cg_iter, data%q_t, SQRT( ABS( gnrmsq ) )
           ELSE          
             WRITE( control%out, "( A, 1X, I7, 4ES12.4 )" )                    &
              prefix, data%cg_iter, data%q_t, SQRT( ABS( gnrmsq ) ),           &
              data%curvature, data%step
           END IF
         END IF
       
!  if the gradient of the model is sufficiently small or if the CG iteration 
!  limit is exceeded, exit; record the CG direction
 
         IF ( SQRT( ABS( gnrmsq ) ) <= data%stop_cg .OR.                       &
              data%cg_iter + 1 > data%cg_maxit ) THEN
           IF ( data% reverse ) THEN
             reverse%V( : prob%n ) = data%DX( : prob%n )
           ELSE
             data%V( : prob%n ) = data%DX( : prob%n )
           END IF
           GO TO 410
         END IF

!  compute the search direction, p_cg, and the square of its length

         data%cg_iter = data%cg_iter + 1
         IF ( data%cg_iter > 1 ) THEN
           beta = gnrmsq / data%old_gnrmsq
           data%P_cg( : prob%n ) = - data%PG_cg( : prob%n )                    &
              + beta * data%P_cg( : prob%n )
           data%pnrmsq = gnrmsq + data%pnrmsq * beta ** 2
         ELSE
           data%P_cg( : prob%n ) = - data%PG_cg( : prob%n )
           data%pnrmsq = gnrmsq
         END IF

!  save the norm of the preconditioned gradient

         data%old_gnrmsq = gnrmsq

!  compute PROD = H * p ...

         IF ( data%explicit_h ) THEN
           data%PROD = zero
           IF ( data%n_free == prob%n ) THEN
             DO i = 1, prob%n
               p_i = data%P_cg( i )
               DO k = data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
                 data%PROD( data%H%col( k ) )                                  &
                   = data%PROD( data%H%col( k ) ) + data%H%val( k ) * p_i
               END DO
             END DO
           ELSE
             DO l = 1, data%n_free
               i = data%FREE( l ) ; p_i = data%P_cg( i )
               DO k = data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
                 data%PROD( data%H%col( k ) )                                  &
                   = data%PROD( data%H%col( k ) ) + data%H%val( k ) * p_i
               END DO
             END DO
           END IF

!  ... or obtain the product from a user-provided subroutine ...

         ELSE IF ( data%use_hprod ) THEN
           CALL eval_HPROD( i, userdata, data%P_cg( : prob%n ), data%PROD )

!  ... or return to the calling program to calculate PROD = H * p

         ELSE
           reverse%V( : prob%n ) = data%P_cg( :  prob%n )
           data%branch = 400 ; inform%status = 3 ; RETURN
         END IF

!  record the free components of H * p

 400     CONTINUE
         IF ( data%n_free == prob%n ) THEN
           IF ( data%reverse ) THEN
             data%HP_cg( : prob%n ) = reverse%PROD( : prob%n )
           ELSE
             data%HP_cg( : prob%n ) = data%PROD( data%FREE( : prob%n ) )
           END IF
         ELSE
           IF ( data%reverse ) THEN
             data%HP_cg( data%FREE( : data%n_free ) ) =                        &
               reverse%PROD( data%FREE( : data%n_free ) )
           ELSE
             data%HP_cg( data%FREE( : data%n_free ) ) =                        &
               data%PROD( data%FREE( : data%n_free ) )
           END IF
         END IF

!  add on the contributions from the products of the Hessian weights with p

         IF ( data%n_free == prob%n ) THEN
           data%HP_cg( : prob%n ) = data%HP_cg( : prob%n ) +                   &
               data%H_diag( : prob%n ) * data%P_cg( : prob%n ) 
         ELSE
           data%HP_cg( data%FREE( : data%n_free ) ) =                          &
             data%HP_cg( data%FREE( : data%n_free ) ) +                        &
               data%H_diag( data%FREE( : data%n_free ) ) *                     &
               data%P_cg( data%FREE( : data%n_free ) ) 
         END IF

!  compute the curvature p^T H p along the search direction

         curvature = DOT_PRODUCT( data%HP_cg( : prob%n ),                     &
                                  data%P_cg( : prob%n ) )
         data%curvature = curvature / data%pnrmsq

!  if the curvature is positive, compute the step to the minimizer of
!  the objective along the search direction

         IF ( curvature > control%zero_curvature * data%pnrmsq ) THEN
           data%step = data%old_gnrmsq / curvature

!  otherwise, the objective is unbounded ....

         ELSE IF ( curvature >= - control%zero_curvature * data%pnrmsq ) THEN
!write(6,*) ' curvature ', data%curvature
           IF ( data% reverse ) THEN
             reverse%V( : prob%n ) = data%P_cg( : prob%n )
           ELSE
             data%V( : prob%n ) = data%P_cg( : prob%n )
           END IF
           GO TO 410
         ELSE
write(6,*) ' curvature ', data%curvature
         inform%status = GALAHAD_error_inertia
           GO TO 900
         END IF

!  update the objective value

         data%q_t = data%q_t + data%step                                       &
           * ( - data%old_gnrmsq + half * data%step * curvature )

!  update the estimate of the solution dx

         data%DX( : prob%n ) = data%DX( : prob%n )                             &
           + data%step * data%P_cg( : prob%n ) 

!  update the gradient at the estimate of the solution

         IF ( data%n_free == prob%n ) THEN
           data%G_cg( : prob%n ) = data%G_cg( : prob%n )                       &
               + data%step * data%HP_cg( : prob%n )
         ELSE
           data%G_cg( data%FREE( : data%n_free ) ) =                           &
             data%G_cg( data%FREE( : data%n_free ) )                           &
               + data%step * data%HP_cg( data%FREE( : data%n_free ) ) 
         END IF
         GO TO 210

!  - - - - - - - - -
!  End the CG loop
!  - - - - - - - - -

 410   CONTINUE
       inform%cg_iter = inform%cg_iter + data%cg_iter

!  compute the dual search direction

       DO l = 1, data%n_free
         i = data%FREE( l )
         IF ( prob%X_l( i ) >= - control%infinity )                            &
           data%DZ_l( i ) = ( data%mu - data%Z_l( i ) *                        &
             ( prob%X( i ) - prob%X_l( i ) + data%DX( i ) ) ) /                &
               ( prob%X( i ) - prob%X_l( i ) ) 
         IF ( prob%X_u( i ) <= control%infinity )                              &
           data%DZ_u( i ) = ( data%mu - data%Z_u( i ) *                        &
             ( prob%X_u( i ) - prob%X( i ) - data%DX( i ) ) ) /                &
               ( prob%X_u( i ) - prob%X( i ) )
       END DO

!  compute the stepsize

       IF ( control%stepsize_strategy == 1 ) THEN

!  compute the distance to the constraint boundary

         alpha_x = one ; alpha_z = one
         DO l = 1, data%n_free
           i = data%FREE( l )
           IF ( prob%X_l( i ) >= - control%infinity ) THEN
             IF ( data%DX( i ) < zero ) alpha_x =                              &
               MIN( alpha_x, ( prob%X_l( i ) - prob%X( i ) ) / data%DX( i ) )
             IF ( data%DZ_l( i ) < zero ) alpha_z =                            &
               MIN( alpha_z, - data%Z_l( i ) / data%DZ_l( i ) )
           END IF
           IF ( prob%X_u( i ) <= control%infinity ) THEN
             IF ( data%DX( i ) > zero ) alpha_x =                              &
               MIN( alpha_x, ( prob%X_u( i ) - prob%X( i ) ) / data%DX( i ) )
             IF ( data%DZ_u( i ) < zero ) alpha_z =                            &
               MIN( alpha_z, - data%Z_u( i ) / data%DZ_u( i ) )
           END IF
         END DO

!  step a large fraction to the boundary
       
         data%alpha =                                                          &
          MIN( one, control%fraction_to_the_boundary * MIN( alpha_x, alpha_z ) )

       ELSE IF ( control%stepsize_strategy == 2 ) THEN

!  balance the complementarity using Zhang's stepsize rules

       ELSE
         CALL BQPB_compute_maxstep( prob%n, prob%X, prob%X_l, prob%X_u,        &
                                    data%Z_l, data%Z_u, data%DX,               &
                                    data%DZ_l, data%DZ_u, data%gamma_b,        &
                                    data%gamma_f, data%nu, alpha_b, alpha_f,   &
                                    data%print_level, control, inform )

         data%alpha = MIN( one, alpha_b, alpha_f )
         data%nu = ( one - data%alpha ) * data%nu
       END IF

!  compute the new point

       DO l = 1, data%n_free
         i = data%FREE( l )
         prob%X( i ) = prob%X( i ) + data%alpha * data%DX( i )
         IF ( prob%X_l( i ) >= - control%infinity )                            &
           data%Z_l( i ) = data%Z_l( i ) + data%alpha * data%DZ_l( i )
         IF ( prob%X_u( i ) <= control%infinity )                              &
           data%Z_u( i ) = data%Z_u( i ) + data%alpha * data%DZ_u( i )
       END DO

       inform%obj = data%q_t
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
       WRITE( control%error, 2010 ) prefix, inform%status, 'BQPB_solve'
     END IF
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN  

!  Non-executable statements

2000 FORMAT( /, A, ' --', A, ' BQPB_solve' ) 
2010 FORMAT( A, '   **  Error return ', I0, ' from ', A ) 
2020 FORMAT( /, A, ' BQPB error exit: ', A )
2030 FORMAT( /, A, ' allocation error status ', I0, ' for ', A )

!  End of BQPB_solve

      END SUBROUTINE BQPB_solve

!-*-*-*-*-*-   B Q P B _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE BQPB_terminate( data, control, inform, reverse )

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
!   data    see Subroutine BQPB_initialize
!   control see Subroutine BQPB_initialize
!   inform  see Subroutine BQPB_solve
!   reverse see Subroutine BQPB_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( BQPB_data_type ), INTENT( INOUT ) :: data
     TYPE ( BQPB_control_type ), INTENT( IN ) :: control        
     TYPE ( BQPB_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( BQPB_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse

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

     array_name = 'bqpb: data%H%ptr'
     CALL SPACE_dealloc_array( data%H%ptr,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%H%col'
     CALL SPACE_dealloc_array( data%H%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%H%val'
     CALL SPACE_dealloc_array( data%H%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%FREE'
     CALL SPACE_dealloc_array( data%FREE,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%G'
     CALL SPACE_dealloc_array( data%G,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%PREC'
     CALL SPACE_dealloc_array( data%PREC,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%H_diag'
     CALL SPACE_dealloc_array( data%H_diag,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%Z_l'
     CALL SPACE_dealloc_array( data%Z_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%Z_u'
     CALL SPACE_dealloc_array( data%Z_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%DX'
     CALL SPACE_dealloc_array( data%DX,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%DZ_l'
     CALL SPACE_dealloc_array( data%DZ_l,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%DZ_u'
     CALL SPACE_dealloc_array( data%DZ_u,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%G_cg'
     CALL SPACE_dealloc_array( data%G_cg,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%PG_cg'
     CALL SPACE_dealloc_array( data%PG_cg,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%P_cg'
     CALL SPACE_dealloc_array( data%P_cg,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%HP_cg'
     CALL SPACE_dealloc_array( data%HP_cg,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%X_new'
     CALL SPACE_dealloc_array( data%X_new,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     IF ( PRESENT( reverse ) ) THEN
       array_name = 'bqpb: reverse%V'
       CALL SPACE_dealloc_array( reverse%V,                                    &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN

       array_name = 'bqpb: reverse%PROD'
       CALL SPACE_dealloc_array( reverse%PROD,                                 &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN
     END IF

     array_name = 'bqpb: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'bqpb: data%PROD'
     CALL SPACE_dealloc_array( data%PROD,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine BQPB_terminate

     END SUBROUTINE BQPB_terminate

!-*-*-*-  B Q P B _ C O M P U T E _ M A X S T E P   S U B R O U T I N E  -*-*-*-

     SUBROUTINE BQPB_compute_maxstep( n, X, X_l, X_u, Z_l, Z_u, DX,            &
                                      DZ_l, DZ_u, gamma_b, gamma_f,            &
                                      nu, alpha_max_b, alpha_max_f,            &
                                      print_level, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the maximum allowable stepsizes alpha_max_b, which balances the 
!  complementarity ie, such that
!
!      min (x-l)_i(z_l)_i - (gamma_b / nbds)( <x-l,z_l> + <u-x,z_u> ) >= 0
!       i                                           
!  and
!      min (u-x)_i(z_u)_i - (gamma_b / nbds)( <x-l,z_l> + <u-x,z_u> ) >= 0 ,
!       i                                           
!
!  and alpha_max_f, which favours feasibility over complementarity, 
!  ie, such that
!
!      <x-l,z_l> + <u-x,z_u> >= nu * gamma_f
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     INTEGER, INTENT( IN ) :: n, print_level
     REAL ( KIND = wp ), INTENT( IN ) :: gamma_b, gamma_f, nu
     REAL ( KIND = wp ), INTENT( OUT ) :: alpha_max_b, alpha_max_f 
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, DX 
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: Z_l, Z_u, DZ_l, DZ_u
     TYPE ( BQPB_control_type ), INTENT( IN ) :: control        
     TYPE ( BQPB_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     INTEGER :: i, nbnds, nroots 

!  Local variables

     REAL ( KIND = wp ) :: compc, compl, compq, coef0, coef1, coef2
     REAL ( KIND = wp ) :: coef0_f, coef1_f, coef2_f
     REAL ( KIND = wp ) :: root1, root2, tol, alpha, alp, nu_gamma_f

     alpha_max_b = infinity ; alpha_max_f = infinity 
     inform%status = GALAHAD_ok
     tol = epsmch ** 0.75 

!  ================================================
!             part to compute alpha_max_b
!  ================================================

!  Compute the coefficients for the quadratic expression
!  for the overall complementarity

     coef0_f = zero ; coef1_f = zero ; coef2_f = zero ; nbnds = 0
     DO i = 1, n
       IF ( X_l( i ) == X_u( i ) ) CYCLE
       IF ( X_l( i ) >= - control%infinity ) THEN
         coef0_f = coef0_f + ( X( i ) - X_l( i ) ) * Z_l( i ) 
         coef1_f = coef1_f + ( X( i ) - X_l( i ) ) * DZ_l( i )                 &
                           + DX( i ) * Z_l( i ) 
         coef2_f = coef2_f + DX( i ) * DZ_l( i )
         nbnds = nbnds + 1
       END IF
       IF ( X_u( i ) <= control%infinity ) THEN
         coef0_f = coef0_f + ( X_u( i ) - X( i ) ) * Z_u( i ) 
         coef1_f = coef1_f + ( X_u( i ) - X( i ) ) * DZ_u( i )                 &
                           - DX( i ) * Z_u( i ) 
         coef2_f = coef2_f - DX( i ) * DZ_u( i ) 
         nbnds = nbnds + 1
       END IF
     END DO 
     IF ( nbnds == 0 ) RETURN

!  Scale these coefficients

     compc = - gamma_b * coef0_f / nbnds ; compl = - gamma_b * coef1_f / nbnds
     compq = - gamma_b * coef2_f / nbnds
!    write(6,"( ' gamma_b ', ES12.4, I6 )" ) gamma_b, nbnds
!    write( 6, "( 3ES10.2 )" )  compq, compl, compc

!  Compute the coefficients for the quadratic expression
!  for the individual complementarity

     DO i = 1, n
       IF ( X_l( i ) == X_u( i ) ) CYCLE
       IF ( X_l( i ) >= - control%infinity ) THEN
         coef0 = compc + ( X( i ) - X_l( i ) ) * Z_l( i ) 
         coef1 = compl + ( X( i ) - X_l( i ) ) * DZ_l( i ) + DX( i ) * Z_l( i ) 
         coef2 = compq + DX( i ) * DZ_l( i ) 
         coef0 = MAX( coef0, zero )
         CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1, root2, &
                               roots_debug ) 
         IF ( nroots == 2 ) THEN 
           IF ( coef2 > zero ) THEN 
             IF ( root2 > zero ) THEN 
                alpha = root1 
             ELSE 
                alpha = control%infinity 
             END IF 
           ELSE 
             alpha = root2 
           END IF 
         ELSE IF ( nroots == 1 ) THEN 
           IF ( root1 > zero ) THEN 
             alpha = root1 
           ELSE 
             alpha = control%infinity 
           END IF 
         ELSE 
           alpha = control%infinity 
         END IF 
!        IF ( control%out > 0 .AND. print_level >= 2 ) THEN
!           IF ( nroots == 2 ) THEN 
!             WRITE( control%out, 2000 )                                       &
!               'X', i, 'L', coef0, coef1, coef2, root1, root2
!           ELSE IF ( nroots == 1 ) THEN 
!             WRITE( control%out, 2000 )                                       &
!               'X', i, 'L', coef0, coef1, coef2, root1 
!           ELSE 
!             WRITE( control%out, 2000 ) 'X', i, 'L', coef0, coef1, coef2 
!           END IF 
!           WRITE( control%out, 2010 ) 'X', i, 'L', alpha 
!        END IF 
         IF ( alpha < alpha_max_b ) alpha_max_b = alpha 
       END IF
       IF ( X_u( i ) <= control%infinity ) THEN
         coef0 = compc + ( X_u( i ) - X( i ) ) * Z_u( i ) 
         coef1 = compl + ( X_u( i ) - X( i ) ) * DZ_u( i ) - DX( i ) * Z_u( i ) 
         coef2 = compq - DX( i ) * DZ_u( i ) 
         coef0 = MAX( coef0, zero )
         CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1, root2, &
                               roots_debug ) 
         IF ( nroots == 2 ) THEN 
           IF ( coef2 > zero ) THEN 
             IF ( root2 > zero ) THEN 
                alpha = root1 
             ELSE 
                alpha = control%infinity 
             END IF 
           ELSE 
             alpha = root2 
           END IF 
         ELSE IF ( nroots == 1 ) THEN 
           IF ( root1 > zero ) THEN 
             alpha = root1 
           ELSE 
             alpha = control%infinity 
           END IF 
         ELSE 
           alpha = control%infinity 
         END IF 
 !       IF ( control%out > 0 .AND. print_level >= 2 ) THEN
 !          IF ( nroots == 2 ) THEN 
 !            WRITE( control%out, 2000 )                                       &
 !              'X', i, 'L', coef0, coef1, coef2, root1, root2
 !          ELSE IF ( nroots == 1 ) THEN 
 !            WRITE( control%out, 2000 )                                       &
 !              'X', i, 'L', coef0, coef1, coef2, root1 
 !          ELSE 
 !            WRITE( control%out, 2000 ) 'X', i, 'U', coef0, coef1, coef2 
 !          END IF 
 !          WRITE( control%out, 2010 ) 'X', i, 'U', alpha 
 !       END IF 
         IF ( alpha < alpha_max_b ) alpha_max_b = alpha 
       END IF
     END DO 

     IF ( - compc <= epsmch ** 0.75 ) alpha_max_b = 0.99_wp * alpha_max_b

!  An error has occured. Investigate

     IF ( alpha_max_b <= zero ) THEN 
       IF ( control%out > 0 .AND. print_level >= 2 )                           &
         WRITE( control%out, 2020 ) alpha_max_b
       DO i = 1, n
         IF ( X_l( i ) == X_u( i ) ) CYCLE
         IF ( X_l( i ) >= - control%infinity ) THEN
           coef0 = compc + ( X( i ) - X_l( i ) ) * Z_l( i ) 
           coef1 = compl + ( X( i ) - X_l( i ) ) * DZ_l( i )                   &
                         + DX( i ) * Z_l( i ) 
           coef2 = compq + DX( i ) * DZ_l( i ) 
           CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1,      &
                                 root2, roots_debug )
           IF ( nroots == 2 ) THEN 
             IF ( coef2 > zero ) THEN 
                IF ( root2 > zero ) THEN 
                   alpha = root1 
                ELSE 
                   alpha = infinity 
                END IF 
             ELSE 
                alpha = root2 
             END IF 
           ELSE IF ( nroots == 1 ) THEN 
             IF ( root1 > zero ) THEN 
                alpha = root1 
             ELSE 
                alpha = infinity 
             END IF 
           ELSE 
             alpha = infinity 
           END IF 
           IF ( alpha == alpha_max_b ) THEN
             IF ( control%out > 0 .AND. print_level >= 2 ) THEN
                IF ( nroots == 2 ) THEN 
                  WRITE( control%out, 2000 )                                   &
                    'X', i, 'L', coef0, coef1, coef2, root1, root2
                ELSE IF ( nroots == 1 ) THEN 
                  WRITE( control%out, 2000 )                                   &
                    'X', i, 'L', coef0, coef1, coef2, root1 
                ELSE 
                  WRITE( control%out, 2000 ) 'X', i, 'L', coef0, coef1, coef2 
                END IF 
                WRITE( control%out, 2010 ) 'X', i, 'L', alpha 
             END IF 
           END IF 
         END IF
         IF ( X_u( i ) <= control%infinity ) THEN
           coef0 = compc + ( X_u( i ) - X( i ) ) * Z_u( i ) 
           coef1 = compl + ( X_u( i ) - X( i ) ) * DZ_u( i )                   &
                         - DX( i ) * Z_u( i ) 
           coef2 = compq - DX( i ) * DZ_u( i ) 
           CALL ROOTS_quadratic( coef0, coef1, coef2, tol, nroots, root1,      &
                                 root2, roots_debug )
           IF ( nroots == 2 ) THEN 
             IF ( coef2 > zero ) THEN 
                IF ( root2 > zero ) THEN 
                   alpha = root1 
                ELSE 
                   alpha = infinity 
                END IF 
             ELSE 
                alpha = root2 
             END IF 
           ELSE IF ( nroots == 1 ) THEN 
             IF ( root1 > zero ) THEN 
                alpha = root1 
             ELSE 
                alpha = infinity 
             END IF 
           ELSE 
             alpha = infinity 
           END IF 
           IF ( alpha == alpha_max_b ) THEN
             IF ( control%out > 0 .AND. print_level >= 2 ) THEN
                IF ( nroots == 2 ) THEN 
                  WRITE( control%out, 2000 )                                   &
                    'X', i, 'U', coef0, coef1, coef2, root1, root2
                ELSE IF ( nroots == 1 ) THEN 
                  WRITE( control%out, 2000 )                                   &
                    'X', i, 'U', coef0, coef1, coef2, root1 
                ELSE 
                  WRITE( control%out, 2000 ) 'X', i, 'U', coef0, coef1, coef2 
                END IF 
                WRITE( control%out, 2010 ) 'X', i, 'U', alpha 
             END IF 
           END IF 
         END IF
       END DO 
       DO i = 1, n
         IF ( X_l( i ) == X_u( i ) ) CYCLE
         IF ( X_l( i ) >= - control%infinity ) THEN
           coef0 = ( X( i ) - X_l( i ) ) * Z_l( i ) 
           coef1 = DX( i ) * Z_l( i ) + ( X( i ) - X_l( i ) ) * DZ_l( i )
           coef2 = DX( i ) * DZ_l( i ) 
           alp = alpha_max_b ; alpha = coef0 + alp * ( coef1 + alp * coef2 ) 
           IF ( control%out > 0 .AND. print_level >= 2 )                       &
             WRITE( control%out, 2030 ) 'X', i, 'L', alp, alpha 
         END IF
         IF ( X_u( i ) <= control%infinity ) THEN
           coef0 = ( X_u( i ) - X( i ) ) * Z_u( i ) 
           coef1 = - DX( i ) * Z_u( i ) + ( X_u( i ) - X( i ) ) * DZ_u( i )
           coef2 = - DX( i ) * DZ_u( i ) 
           alp = alpha_max_b ; alpha = coef0 + alp * ( coef1 + alp * coef2 ) 
           IF ( control%out > 0 .AND. print_level >= 2 )                       &
             WRITE( control%out, 2030 ) 'X', i, 'U', alp, alpha 
         END IF
       END DO 
       alp = alpha_max_b ; alpha = compc + alp * ( compl + alp * compq ) 
       IF ( control%out > 0 .AND. print_level >= 2 ) THEN
         WRITE( control%out, 2040 ) alpha 
         WRITE( control%out, 2020 ) alpha_max_b 
       END IF
       WRITE( control%out, "( ' -ve step, no further progress possible ' )" )
       inform%status = GALAHAD_error_tiny_step
       RETURN
     END IF 

!  ================================================
!             part to compute alpha_max_f
!  ================================================

     nu_gamma_f = nu * gamma_f

!  Compute the coefficients for the quadratic expression
!  for the overall complementarity, remembering to first 
!  subtract the term for the feasibility

     coef0_f = coef0_f - nu_gamma_f
     coef1_f = coef1_f + nu_gamma_f

!  Compute the coefficients for the quadratic expression
!  for the individual complementarity
!
     CALL ROOTS_quadratic( coef0_f, coef1_f, coef2_f, tol,                    &
                           nroots, root1, root2, roots_debug )
     IF ( nroots == 2 ) THEN 
       IF ( coef2_f > zero ) THEN 
         IF ( root2 > zero ) THEN 
           alpha = root1 
         ELSE 
           alpha = infinity 
         END IF 
       ELSE 
         alpha = root2 
       END IF 
     ELSE IF ( nroots == 1 ) THEN 
       IF ( root1 > zero ) THEN 
         alpha = root1 
       ELSE 
         alpha = infinity 
       END IF 
     ELSE 
       alpha = infinity 
     END IF 
     IF ( alpha < alpha_max_f ) alpha_max_f = alpha 
     IF ( - compc <= epsmch ** 0.75 ) alpha_max_f = 0.99_wp * alpha_max_f

     RETURN
  
!  Non-executable statements

2000 FORMAT( A1, I6, A1,' coefs', 3ES12.4,' roots', 2ES12.4 ) 
2010 FORMAT( A1, I6, A1,' alpha', ES12.4 ) 
2020 FORMAT( ' alpha_min ', ES12.4 ) 
2030 FORMAT( A1, I6, A1,' value at ', ES12.4,' = ', ES12.4 ) 
2040 FORMAT( ' .vs. ', ES12.4 ) 

!  End of subroutine BQPB_compute_maxstep

     END SUBROUTINE BQPB_compute_maxstep

!  End of module BQPB

   END MODULE GALAHAD_BQPB_double
