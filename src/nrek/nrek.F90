! THIS VERSION: GALAHAD 5.4 - 2025-11-22 AT 13:50 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ N R E K    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Fowkes/Gould/Montoison/Orban, for GALAHAD productions
!  Principal authors: Hussam Al Daas and Nick Gould

!  History -
!   originally released in GALAHAD Version 5.4. November 22nd 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_NREK_precision

     USE GALAHAD_KINDS_precision, ONLY: ip_, rp_, sp_
     USE GALAHAD_CLOCK, ONLY: CLOCK_time
     USE GALAHAD_SYMBOLS, ONLY: GALAHAD_ok, GALAHAD_error_allocate,            &
           GALAHAD_error_deallocate, GALAHAD_error_restrictions,               &
           GALAHAD_error_analysis, GALAHAD_error_factorization,                &
           GALAHAD_error_solve, GALAHAD_error_inertia, GALAHAD_error_lapack,   &
           GALAHAD_error_max_iterations, GALAHAD_error_call_order,             &
           GALAHAD_error_preconditioner, GALAHAD_error_optional,               &
           GALAHAD_error_unknown_storage
     USE GALAHAD_SPACE_precision, ONLY: SPACE_resize_array, SPACE_dealloc_array
     USE GALAHAD_SPECFILE_precision, ONLY: SPECFILE_item_type, SPECFILE_read,  &
           SPECFILE_assign_value
     USE GALAHAD_SMT_precision, ONLY: SMT_type, SMT_put, SMT_get
     USE GALAHAD_RAND_precision, ONLY: RAND_seed, RAND_initialize,             &
           RAND_random_real
     USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
     USE GALAHAD_MOP_precision, ONLY: NREK_Mv => MOP_Ax
     USE GALAHAD_BLAS_inter_precision, ONLY: TBSV
     USE GALAHAD_LAPACK_inter_precision, ONLY: LAENV, GEQRF, ORGQR, SYEV,      &
           PBTRF, PBTRS
     USE GALAHAD_SLS_precision, ONLY: SLS_control_type, SLS_inform_type,       &
           SLS_data_type, SLS_initialize, SLS_read_specfile, SLS_analyse,      &
           SLS_factorize, SLS_solve, SLS_terminate, SLS_keyword
     USE GALAHAD_RQS_precision, ONLY: RQS_control_type, RQS_inform_type,       &
          RQS_read_specfile, RQS_solve_diagonal

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: NREK_initialize, NREK_read_specfile, NREK_solve,                &
               NREK_terminate, NREK_full_initialize, NREK_full_terminate,      &
               NREK_import, NREK_S_import, NREK_solve_problem,                 &
               NREK_reset_control, NREK_information,                           &
               SMT_type, SMT_put, SMT_get

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE NREK_initialize
       MODULE PROCEDURE NREK_initialize, NREK_full_initialize
     END INTERFACE NREK_initialize

     INTERFACE NREK_terminate
       MODULE PROCEDURE NREK_terminate, NREK_full_terminate
     END INTERFACE NREK_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = rp_ ), PARAMETER :: infinity = half * HUGE( one )
     REAL ( KIND = rp_ ), PARAMETER :: h_pert = SQRT( epsmch )
     REAL ( KIND = rp_ ), PARAMETER :: s_pert = SQRT( epsmch )
     REAL ( KIND = rp_ ), PARAMETER :: delta_tiny = ten * epsmch
     REAL ( KIND = rp_ ), PARAMETER :: small = epsmch ** 0.5_rp_
!    REAL ( KIND = rp_ ), PARAMETER :: small = epsmch ** 0.75_rp_
     INTEGER ( KIND = ip_ ) :: ldp = 3 ! P is symmetric, pentadiagonal band
     LOGICAL :: termination_test = .TRUE.

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: NREK_control_type

!   error and warning diagnostics occur on stream error

       INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

       INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  maximum dimension of the extended Krylov space

       INTEGER ( KIND = ip_ ) :: eks_max = - 1

!  maximum iteration count

       INTEGER ( KIND = ip_ ) :: it_max = - 1

!  the constant term, f, in the objective function

        REAL ( KIND = rp_ ) :: f = zero

!  increase factor for subsequent regularization weights

       REAL ( KIND = rp_ ) :: increase = 2.0_rp_

!  stopping tolerance for the cheaply-computed residual

!      || ( H + lambda S ) x + c || < stop_residua max( 1, ||c|| )

       REAL ( KIND = rp_ ) :: stop_residual = ten * SQRT( epsmch )

!  should the incoming Lanczos vectors be re-orthogonalised against the
!   existing ones (this can be expensive)

       LOGICAL :: reorthogonalize = .FALSE.

!  choose between two versions, either that given as Algorithm 5.2  or B.3
!   in the paper, for recurrences when a non-unit S is given

       LOGICAL :: s_version_52 = .TRUE.

!  make a tiny perturbation to the term c to try to protect from the hard case

       LOGICAL :: perturb_c = .FALSE.

!  check for convergence for all system orders, not just even ones

       LOGICAL :: stop_check_all_orders = .FALSE.

!  resolve a previously solved problem with a larger weight

       LOGICAL :: new_weight = .FALSE.

!  solve a problem with the same structure as the previous one but with 
!   different values of H, c and/or S

       LOGICAL :: new_values = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

       LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  symmetric linear equation solver for systems involving H

       CHARACTER ( LEN = 30 ) :: linear_solver = "ssids" // REPEAT( ' ', 25 )

!  symmetric linear equation solver for systems involving S (if needed)

       CHARACTER ( LEN = 30 ) :: linear_solver_for_S = "ssids" //              &
                                                          REPEAT( ' ', 25 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""' // REPEAT( ' ', 28 )

!  control parameters for SLS factorization of (possibly shifted) H

       TYPE ( SLS_control_type ) :: SLS_control

!  control parameters for SLS factorization of S (if needed)

       TYPE ( SLS_control_type ) :: SLS_S_control

!  control parameters for RQS

       TYPE ( RQS_control_type ) :: RQS_control

     END TYPE NREK_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: NREK_time_type

!  total CPU time spent in the package

        REAL ( KIND = rp_ ) :: total = 0.0_rp_

!  CPU time spent building H and S

        REAL ( KIND = rp_ ) :: assemble = 0.0_rp_

!  CPU time spent reordering H and S prior to factorization

        REAL ( KIND = rp_ ) :: analyse = 0.0_rp_

!  CPU time spent factorizing H and S

        REAL ( KIND = rp_ ) :: factorize = 0.0_rp_

!  CPU time spent solving linear systems inolving H and S

        REAL ( KIND = rp_ ) :: solve = 0.0_rp_

!  total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0_rp_

!  clock time spent building H and S

        REAL ( KIND = rp_ ) :: clock_assemble = 0.0_rp_

!  clock time spent reordering H and S prior to factorization

        REAL ( KIND = rp_ ) :: clock_analyse = 0.0_rp_

!  clock time spent factorizing H and S

        REAL ( KIND = rp_ ) :: clock_factorize = 0.0_rp_

!  clock time spent solving linear systems inolving H and S

        REAL ( KIND = rp_ ) :: clock_solve = 0.0_rp_
      END TYPE NREK_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: NREK_inform_type

!  return status. See NREK_solve for details

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  the number of orthogonal vectors required

        INTEGER ( KIND = ip_ ) :: n_vec = - 1

!  the value of the quadratic function

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the value of the regularized quadratic function

        REAL ( KIND = rp_ ) :: obj_regularized = HUGE( one )

!  the S-norm of x, ||x||_S

        REAL ( KIND = rp_ ) :: x_norm = zero

!  the Lagrange multiplier asssociated with the regularization

        REAL ( KIND = rp_ ) :: multiplier = zero

!  the current weight

        REAL ( KIND = rp_ ) :: weight = - one

!  the proposed next weight to be used

        REAL ( KIND = rp_ ) :: next_weight = - one

!  the maximum relative residual error

        REAL ( KIND = rp_ ) :: error = - one

!  time information

        TYPE ( NREK_time_type ) :: time

!  inform parameters for SLS factorization of (possibly shifted) H

        TYPE ( SLS_inform_type ) :: SLS_inform

!  inform parameters for SLS factorization of S (if needed)

        TYPE ( SLS_inform_type ) :: SLS_S_inform

!  inform parameters for RQS

        TYPE ( RQS_inform_type ) :: RQS_inform

      END TYPE NREK_inform_type

!  ...................
!   data derived type
!  ...................

      TYPE, PUBLIC :: NREK_data_type

        INTEGER ( KIND = ip_ ) :: is_max, it_max, k_max, lwork_syev
        INTEGER ( KIND = ip_ ) :: h_ne, s_ne, h_shift_ne
        INTEGER ( KIND = ip_ ) :: k_exit = - 1
        INTEGER ( KIND = ip_ ) :: re_enter = 0
        REAL ( KIND = rp_ ) :: c_norm, last_weight, last_shift, shift_val, ztu
        REAL ( KIND = rp_ ) :: alpha_km1, alpha_mk, beta_k, beta_mk, gamma
        REAL ( KIND = rp_ ) :: delta_mk, delta_k, delta_km1, delta_s
        LOGICAL :: shifted, sparse, perturb_c, c_norm_eq_0
        LOGICAL :: unit_s = .TRUE.
        LOGICAL :: allocated_arrays = .FALSE.
        TYPE ( RAND_seed ) :: seed

!  common workspace arrays

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: MAP
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: U, W, Z, Qp, Qm ! n
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C, D, X, S1, C_pert
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WORK_syev
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: V, P, P_shift
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: Q, S, S2
        TYPE ( SMT_type ) :: H_shift

!  local copy of control

        TYPE ( NREK_control_type ) :: control

!  data for SLS factorization of (possibly shifted) H and S (if needed)

        TYPE ( SLS_data_type ) :: SLS_data
        TYPE ( SLS_data_type ) :: SLS_S_data

      END TYPE NREK_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: NREK_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( NREK_data_type ) :: NREK_data
        TYPE ( NREK_control_type ) :: NREK_control
        TYPE ( NREK_inform_type ) :: NREK_inform
        TYPE ( SMT_type ) :: H, S
        LOGICAL :: use_s
      END TYPE NREK_full_data_type

   CONTAINS

!-*-*-*-*-*-   N R E K _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE NREK_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for NREK. This routine should be called before
!  NREK_solve
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( NREK_data_type ), INTENT( INOUT ) :: data
      TYPE ( NREK_control_type ), INTENT( OUT ) :: control
      TYPE ( NREK_inform_type ), INTENT( OUT ) :: inform

!  local variables

      TYPE ( RQS_control_type ) :: rqs_control
      TYPE ( RQS_inform_type ) :: rqs_inform

      inform%status = GALAHAD_ok

!  initalize random number seed

      CALL RAND_initialize( data%seed )

!  initalize SLS components (now do this later)

!     CALL SLS_INITIALIZE( control%linear_solver,                              &
!                          data%SLS_data, control%SLS_control,                 &
!                          inform%SLS_inform, check = .TRUE. )
!     control%linear_solver = inform%SLS_inform%solver
      control%SLS_control%prefix = '" - SLS:"                    '

!     CALL SLS_INITIALIZE( control%linear_solver_for_S,                        &
!                          data%SLS_S_data, control%SLS_S_control,             &
!                          inform%SLS_S_inform, check = .TRUE. )
!     control%linear_solver_for_S = inform%SLS_S_inform%solver
      control%SLS_S_control%prefix = '" - S SLS:"                  '

!  initialize RQS components

      control%RQS_control = rqs_control
      control%RQS_control%print_level = - 1
      control%RQS_control%prefix = '" - RQS:"                     '
      inform%RQS_inform = rqs_inform

      RETURN

!  End of NREK_initialize

      END SUBROUTINE NREK_initialize

!- G A L A H A D -  N R E K _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE NREK_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for NREK controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NREK_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NREK_control_type ), INTENT( OUT ) :: control
     TYPE ( NREK_inform_type ), INTENT( OUT ) :: inform

     CALL NREK_initialize( data%nrek_data, control, inform )

     RETURN

!  End of subroutine NREK_full_initialize

     END SUBROUTINE NREK_full_initialize

!-*-*-*-*-   N R E K _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE NREK_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by NREK_initialize could (roughly)
!  have been set as:

! BEGIN NREK SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-subspace-dimension                        -1
!  maximum-number-of-iterations                      -1
!  constant-term-in-objective                        0.0E+0
!  weight-increase-factor                            2.0E+0
!  residual-accuracy                                 1.0E-8
!  reorthogonalize-vectors                           F
!  s-version-52                                      T
!  perturb-c                                         F
!  stop-check-all-orders                             F
!  new-weight                                        F
!  new-values                                        F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  linear-equation-solver                            ssids
!  linear-equation-solver-for-S                      ssids
!  output-line-prefix                                ""
! END NREK SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( NREK_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL, INTENT( IN ) :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: eks_max = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: it_max = eks_max + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: f = it_max + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: increase = f + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_residual = increase + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: reorthogonalize = stop_residual + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: s_version_52                        &
                                             = reorthogonalize + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: perturb_c = s_version_52 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_check_all_orders               &
                                             = perturb_c + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: new_weight                          &
                                             = stop_check_all_orders + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: new_values = new_weight + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = new_values + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: linear_solver                       &
                                             = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: linear_solver_for_S                 &
                                             = linear_solver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = linear_solver_for_S + 1

      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'NREK'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( eks_max )%keyword = 'maximum-subspace-dimension'
      spec( it_max )%keyword = 'maximum-number-of-iterations'

!  Real key-words

      spec( f )%keyword = 'constant-term-in-objective'
      spec( increase )%keyword = 'weight-increase-factor'
      spec( stop_residual )%keyword = 'residual-accuracy'

!  Logical key-words

      spec( reorthogonalize )%keyword = 'reorthogonalize-vectors'
      spec( s_version_52 )%keyword = 's-version-52'
      spec( perturb_c )%keyword = 'perturb-c'
      spec( stop_check_all_orders )%keyword = 'stop-check-all-orders'
      spec( new_weight )%keyword = 'new-weight'
      spec( new_values )%keyword = 'new-values'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( linear_solver )%keyword = 'linear-equation-solver'
      spec( linear_solver_for_S )%keyword = 'linear-equation-solver-for-S'
      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' nrek: ', alt_specname

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
     CALL SPECFILE_assign_value( spec( eks_max ),                              &
                                 control%eks_max,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( it_max ),                               &
                                 control%it_max,                               &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( f ),                                    &
                                 control%f,                                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( increase ),                            &
                                 control%increase,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_residual ),                        &
                                 control%stop_residual,                        &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( reorthogonalize ),                      &
                                 control%reorthogonalize,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( s_version_52 ),                         &
                                 control%s_version_52,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( perturb_c ),                            &
                                 control%perturb_c,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_check_all_orders ),                &
                                 control%stop_check_all_orders,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( new_weight ),                           &
                                 control%new_weight,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( new_values ),                           &
                                 control%new_values,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
!  Set character values


     CALL SPECFILE_assign_value( spec( linear_solver ),                        &
                                 control%linear_solver,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( linear_solver_for_S ),                  &
                                 control%linear_solver_for_S,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Read the specfiles for SLS and RQS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS')
        CALL SLS_read_specfile( control%SLS_S_control, device,                 &
                                alt_specname = TRIM( alt_specname ) // '-SLS-S')
        CALL RQS_read_specfile( control%RQS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-RQS')
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
        CALL SLS_read_specfile( control%SLS_S_control, device,                 &
                                alt_specname = 'SLS-S' )
        CALL RQS_read_specfile( control%RQS_control, device )
      END IF

      RETURN

      END SUBROUTINE NREK_read_specfile

!-*-*-*-*-*-*-*-*-   N R E K _ S O L V E    S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE NREK_solve( n, H, C, power, weight, X, data, control, inform,  &
                            S )

!  Given an n x n symmetric matrix H, an n-vector c, a power > 2, a weight > 0,
!  approximately solve the regularization subproblem
!
!    min 1/2 x'Hx + c'x + f + (weight/power) ||x||_S^power
!
!  using an extended Krylov subspace method.
!
!  The method uses the "backward" extended-Krylov subspace
!
!    K_2k+1 = {c, S H^{-1} c, H S^{-1} c, ..., (S H^{-1})^k c, (H S^{-1})^k c},
!
!  (see module EKS)
!
!  Input:
!   n - number of unknowns
!   H - symmetric coefficient matrix, H, from the quadratic term,
!       in any symmetric format supported by the SMT type
!   C - vector c from linear term
!   power - scalar regularization power > 2
!   weight - scalar regularization weight > 0
!   control - parameters structure (see preamble)
!   inform - output structure (see preamble)
!   data - prvate internal workspace
!   S - (optional) symmetric, positive definite scaling matrix, S, from the
!       norm term, in any symmetric format supported by the SMT type
!
!   Output:
!   X - solution vector x

!  dummy arguments

     INTEGER ( KIND = ip_ ),  INTENT( IN ) :: n
     TYPE ( SMT_type ), INTENT( IN ) :: H
!    TYPE ( SMT_type ), INTENT( INOUT ) :: H
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: C
     REAL ( KIND = rp_ ), INTENT( IN ) :: power, weight
     REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n  ) :: X
     TYPE ( NREK_data_type ), INTENT( INOUT ) :: data
     TYPE ( NREK_control_type ), INTENT( IN ) :: control
     TYPE ( NREK_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( SMT_type ), OPTIONAL, INTENT( IN ) :: S

!  local variables

     INTEGER ( KIND = ip_ ) :: i, j, jj, j_max, l
     INTEGER ( KIND = ip_ ) :: k, km1, k2, k2m1, k2p1, mkm1, mkp1
     INTEGER ( KIND = ip_ ) :: k_start, lapack_info, nb, error, out
     INTEGER ( KIND = ip_ ) :: shift_status
     REAL ( KIND = rp_ ) :: alpha, beta, delta, x_norm
!    REAL ( KIND = rp_ ) :: e11, e12, e22
     REAL ( KIND = rp_ ) :: shift, error_r, s_norm
     REAL ( KIND = sp_ ) :: time_start, time_now, time_record
     REAL ( KIND = rp_ ) :: clock_start, clock_now, clock_record
     LOGICAL :: printe, printi, printm, printd, printh
     LOGICAL :: initial, restart, new_vals, shifted_structure, s_ok
     CHARACTER ( LEN = 80 ) :: array_name

!  temporary debug variables - ultimately remove

     INTEGER ( KIND = ip_ ) :: ii
     INTEGER ( KIND = ip_ ),  PARAMETER :: p_dim = 10
     REAL ( KIND = rp_) :: P_calc( p_dim, p_dim )
     REAL ( KIND = rp_) :: R( n ), SOL( n )

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  record error output values

     error = control%error
     printe = control%print_level > 0 .AND. error > 0

!  see if this is an initial- or a re-entry

      initial = .NOT. control%new_weight

!  check solve has been called before resolve

     IF ( .NOT. initial ) THEN
       IF ( data%k_exit <= 0 ) THEN
         IF ( printe ) WRITE( error,                                           &
            "( A, ' solve not called before resolve' )" ) prefix
         inform%status = GALAHAD_error_call_order ; GO TO 920
       END IF
     END IF

!  see if the H-c-S data is new

     restart = control%new_values .AND. data%k_exit > 0

     IF ( restart ) THEN
       initial = .FALSE.
       data%k_exit = - 1
     END IF

!  see if new values are to be recorded

     new_vals = initial .OR. restart

!  check input dimensions are consistent

     IF ( n <= 0 ) THEN
       IF ( printe ) WRITE( error, "( A, ' n = ', I0, ' <= 0' )" ) prefix, n
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

     IF ( n /= H%n ) THEN
       IF ( printe ) WRITE( error, "( A, ' n = ', I0, ' and dimension H = ',   &
      &  I0, ' are different' )" ) prefix, n, H%n
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

!  check H is of recognised type

     IF ( .NOT. SLS_keyword( H%type ) ) THEN
       IF ( printe ) WRITE( error, "( A, ' H matrix type ', A,                 &
      &  ' not recognised' )" ) prefix, TRIM( SMT_get( H%type ) )
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

!  check S is consistent if provided

     IF ( PRESENT( S ) ) THEN
       IF ( n /= S%n ) THEN
         IF ( printe ) WRITE( error, "( A, ' n = ', I0, ' and dimension S = ', &
        &  I0, ' are different' )" ) prefix, n, S%n
         inform%status = GALAHAD_error_restrictions ; GO TO 920
       END IF
       IF ( .NOT. SLS_keyword( S%type ) ) THEN
         IF ( printe ) WRITE( error, "( A, ' S matrix type ', A,               &
        &  ' not recognised' )" ) prefix, TRIM( SMT_get( S%type ) )
         inform%status = GALAHAD_error_restrictions ; GO TO 920
       END IF

!  check that S is appropriate (primarily, is it strictly diagonally dominant)

       s_ok = .TRUE.
       SELECT CASE ( SMT_get( S%type ) )
       CASE ( 'ZERO', 'NONE' )
         s_ok = .FALSE.
       CASE ( 'SCALED_IDENTITY' )
         s_ok = S%val( 1 ) > zero
       CASE ( 'DIAGONAL' )
         s_ok = COUNT( S%val( : S%n ) > zero ) == S%n
       END SELECT
       IF ( .NOT. s_ok ) THEN
         IF ( printe ) WRITE( error, "( A, ' S matrix is inappropriate' )" )   &
           prefix
         inform%status = GALAHAD_error_preconditioner  ; GO TO 920
       END IF

!  see if S is the identity matrix

!      data%unit_s = SMT_get( S%type ) == 'IDENTITY'
       data%unit_s = .FALSE.
     ELSE
       data%unit_s = .TRUE.
     END IF

!  check initial power value is greater than two

     IF ( power <= two ) THEN
       IF ( printe ) WRITE( error, "( A, ' weight no bigger than 2 provided')")&
         prefix
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

!  check initial weight value is positive

     IF ( new_vals ) THEN
       IF ( weight <= zero ) THEN
         IF ( printe ) WRITE( error, "( A, ' non-positive weight provided' )" )&
           prefix
         inform%status = GALAHAD_error_restrictions ; GO TO 920
       END IF

!  on reentry, also check the new weight value is smaller than the previous one

     ELSE
       IF ( weight <= zero .OR. weight <= data%last_weight ) THEN
         IF ( printe ) WRITE( error, "( A, ' inappropriate weight provided' )")&
           prefix
         inform%status = GALAHAD_error_restrictions ; GO TO 920
       END IF

!  record the weight, and set the potential next one

       inform%next_weight = weight * data%control%increase
     END IF
     inform%weight = weight
     data%last_weight = weight

!  record output values

     out = control%out
     printi = control%print_level > 0 .AND. out > 0
     printm = control%print_level > 2 .AND. out > 0
     printd = control%print_level > 5 .AND. out > 0

     IF ( .NOT. data%unit_s .AND. printd ) THEN
       IF ( control%s_version_52 ) THEN
          WRITE( out, "( A, ' version 5.2 used' )" ) prefix
       ELSE
          WRITE( out, "( A, ' version A.3 used' )" ) prefix
       END IF
     END IF

!  initial entry - allocate data structures
!  ----------------------------------------

     IF ( initial ) THEN

!  record the iteration limit

       data%control = control
       IF ( data%control%RQS_control%max_factorizations < 0 )                  &
         data%control%RQS_control%max_factorizations = 10
       IF ( data%control%RQS_control%print_level < 0 )                         &
         data%control%RQS_control%print_level = control%print_level - 1
!      data%k_max = MAX( control%it_max, 1 )
       IF ( control%eks_max > 0 ) THEN
         data%k_max = control%eks_max
       ELSE
         data%k_max = 100
       END IF
       IF ( control%it_max > 0 ) THEN
         data%it_max = control%it_max
       ELSE
         data%it_max = 100
       END IF
       data%it_max = MIN( data%it_max, data%k_max )
       data%is_max = 2 * data%k_max
       data%k_exit = - 1
       data%re_enter = 0

!   provide space for, and initiate, the components of the projected solution

       IF ( .NOT. data%allocated_arrays ) THEN
         data%allocated_arrays = .TRUE.

         array_name = 'nrek: data%V'
         CALL SPACE_resize_array(  1_ip_, n, - data%k_max, data%k_max, data%V, &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  set the projected matrix P and solution s

         array_name = 'nrek: data%P'
         CALL SPACE_resize_array( ldp, data%is_max, data%P,                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: data%P_shift'
         CALL SPACE_resize_array( ldp, data%is_max, data%P_shift,              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: data%S1'
         CALL SPACE_resize_array( data%is_max, data%S1,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: data%S2'
         CALL SPACE_resize_array( data%is_max, 1_ip_, data%S2,                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  set workspace arrays

         array_name = 'nrek: data%U'
         CALL SPACE_resize_array( n, data%U,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: data%Q'
         CALL SPACE_resize_array( data%is_max, data%is_max, data%Q,            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: data%D'
         CALL SPACE_resize_array( data%is_max, data%D,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: data%C'
         CALL SPACE_resize_array( data%is_max, data%C,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: data%X'
         CALL SPACE_resize_array( data%is_max, data%X,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  discover the size of, and allocate, workspace needed for the eigensolver

         nb = LAENV( 1_ip_, 'DSYTRD', 'L', data%is_max,                        &
                     - 1_ip_, - 1_ip_, - 1_ip_ )
         data%lwork_syev                                                       &
           = MAX( 1_ip_, 3 * data%is_max - 1, ( nb + 2 ) * data%is_max )

         ALLOCATE( data%WORK_syev( data%lwork_syev ),                          &
                   STAT = inform%alloc_status )
         IF ( inform%alloc_status /= 0 ) THEN
           inform%bad_alloc = 'nrek: data%WORK_syev'
           inform%status = GALAHAD_error_allocate ; GO TO 910
         END IF
       END IF

       IF ( data%unit_s ) THEN

!  set an additional workspace array for debugging

         IF ( printd ) THEN
           array_name = 'nrek: data%Z'
           CALL SPACE_resize_array( n, data%Z,                                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910
         END IF

!  if S is not the identity, find its Cholesky factors

       ELSE

!   first initialize and analyse S

         CALL SLS_initialize( control%linear_solver_for_S, data%SLS_S_data,    &
                              data%control%SLS_S_control,                      &
                              inform%SLS_S_inform, check = .TRUE. )
         data%control%SLS_S_control%pivot_control = 2

         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
         CALL SLS_analyse( S, data%SLS_S_data, data%control%SLS_S_control,     &
                           inform%SLS_S_inform )
         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%analyse = inform%time%analyse + time_now - time_record
         inform%time%clock_analyse =                                           &
           inform%time%clock_analyse + clock_now - clock_record

!  check if the analysis failed

         IF ( inform%SLS_S_inform%status < 0 ) THEN
           IF ( printe ) WRITE( error, "( A, ' analysis of S failed,',         &
          & ' SLS inform%status = ', I0 )" ) prefix, inform%SLS_S_inform%status
           inform%status = GALAHAD_error_analysis ; GO TO 920

!  otherwise factorize S, while checking that it is definite

         ELSE
           CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
           CALL SLS_factorize( S, data%SLS_S_data, data%control%SLS_S_control, &
                               inform%SLS_S_inform )
           CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
           inform%time%factorize =                                             &
             inform%time%factorize + time_now - time_record
           inform%time%clock_factorize =                                       &
             inform%time%clock_factorize + clock_now - clock_record

!  S is indefinite

           IF ( inform%SLS_S_inform%status == GALAHAD_error_inertia .OR.       &
                inform%SLS_S_inform%negative_eigenvalues > 0 ) THEN
             IF ( printe )                                                     &
               WRITE( error, "( A, ' S appears to be indefinite' )" ) prefix
             inform%status = GALAHAD_error_preconditioner ; GO TO 920

!  the factorization failed

           ELSE IF ( inform%SLS_S_inform%status < 0 ) THEN
             IF ( printe ) WRITE( error, "( A, ' factorization of S failed, ', &
            & 'SLS inform%status = ', I0 )" ) prefix, inform%SLS_S_inform%status
             inform%status = GALAHAD_error_factorization ; GO TO 920
           END IF
         END IF

!  set up workspace q_+ and q_- to store the auxiliary vectors q_i = S v_i

         array_name = 'nrek: data%Qp'
         CALL SPACE_resize_array( n, data%Qp,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: data%Qm'
         CALL SPACE_resize_array( n, data%Qm,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  set additional workspace arrays

         array_name = 'nrek: data%W'
         CALL SPACE_resize_array( n, data%W,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: data%Z'
         CALL SPACE_resize_array( n, data%Z,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910
       END IF

!   initialize and analyse H

       CALL SLS_initialize( control%linear_solver, data%SLS_data,              &
                            data%control%SLS_control,                          &
                            inform%SLS_inform, check = .TRUE. )
       data%control%sls_control%pivot_control = 2

       CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
       CALL SLS_analyse( H, data%sls_data, data%control%sls_control,           &
                         inform%sls_inform )
       CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
       inform%time%analyse = inform%time%analyse + time_now - time_record
       inform%time%clock_analyse =                                             &
         inform%time%clock_analyse + clock_now - clock_record

!  check if the analysis failed

       IF ( inform%sls_inform%status < 0 .AND.                                 &
            inform%sls_inform%status /= GALAHAD_error_inertia ) THEN
         IF ( printe ) WRITE( error, "( A, ' analysis of H failed,',           &
        &  ' SLS inform%status = ', I0 )" ) prefix, inform%SLS_inform%status
         inform%status = GALAHAD_error_analysis ; GO TO 920

!  skip the factorization if the analysis finds that H is structurally
!  indefinite

       ELSE IF ( inform%sls_inform%status == GALAHAD_error_inertia ) THEN
         shifted_structure = .TRUE.
         data%shifted = .TRUE.
       ELSE

!  otherwise factorize H, while checking that it is definite

         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
         CALL SLS_factorize( H, data%sls_data, data%control%sls_control,       &
                             inform%sls_inform )
         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%factorize = inform%time%factorize + time_now - time_record
         inform%time%clock_factorize =                                         &
           inform%time%clock_factorize + clock_now - clock_record

!  H is indefinite

         IF ( inform%sls_inform%status == GALAHAD_error_inertia .OR.           &
              inform%sls_inform%negative_eigenvalues > 0 .OR.                  &
              inform%sls_inform%rank < n ) THEN
           shifted_structure = .TRUE.
           data%shifted = .TRUE.

!  the factorization failed

         ELSE IF ( inform%sls_inform%status < 0 ) THEN
           IF ( printe ) WRITE( error, "( A, ' factorization of H failed, ',   &
          &  'SLS inform%status = ', I0 )" ) prefix, inform%SLS_inform%status
           inform%status = GALAHAD_error_factorization ; GO TO 920

!  H is definite

         ELSE
           shifted_structure = .FALSE.
           data%shifted = .FALSE.
         END IF
       END IF
     END IF

!write(6,*) ' shifted ', data%shifted, 'new vals ', new_vals

!  new values have appeared and there is no need to re-analyse the structure
!  of H and (perhaps) S. Proceed straight to the factorization stages

     IF ( new_vals ) THEN

!  factorize non unit S

       IF ( .NOT. data%unit_s ) THEN
         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
         CALL SLS_factorize( S, data%SLS_S_data, data%control%SLS_S_control,   &
                             inform%SLS_S_inform )
         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%factorize =                                               &
           inform%time%factorize + time_now - time_record
         inform%time%clock_factorize =                                         &
           inform%time%clock_factorize + clock_now - clock_record

!  S is indefinite

         IF ( inform%SLS_S_inform%status == GALAHAD_error_inertia .OR.         &
              inform%SLS_S_inform%negative_eigenvalues > 0 ) THEN
           IF ( printe )                                                       &
             WRITE( error, "( A, ' new S appears to be indefinite' )" ) prefix
           inform%status = GALAHAD_error_preconditioner ; GO TO 920

!  the factorization failed

         ELSE IF ( inform%SLS_S_inform%status < 0 ) THEN
           IF ( printe ) WRITE( error, "( A, ' factorization of S failed, ',   &
          & 'SLS inform%status = ', I0 )" ) prefix, inform%SLS_S_inform%status
           inform%status = GALAHAD_error_factorization ; GO TO 920
         END IF
       END IF

       IF ( restart ) THEN
         IF ( .NOT. data%shifted ) THEN

!  factorize H, again checking that it is definite

           CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
           CALL SLS_factorize( H, data%SLS_data, data%control%SLS_control,     &
                               inform%SLS_inform )
           CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
           inform%time%factorize =                                             &
             inform%time%factorize + time_now - time_record
           inform%time%clock_factorize =                                       &
             inform%time%clock_factorize + clock_now - clock_record

!  H is indefinite

           IF ( inform%SLS_inform%status == GALAHAD_error_inertia .OR.         &
                inform%SLS_inform%negative_eigenvalues > 0 ) THEN
             shifted_structure = .TRUE.

!  the factorization failed

           ELSE IF ( inform%sls_inform%status < 0 ) THEN
             IF ( printe ) WRITE( error, "( A, ' factorization of H failed, ', &
            &  'SLS inform%status = ', I0 )" ) prefix, inform%SLS_inform%status
             inform%status = GALAHAD_error_factorization ; GO TO 920

!  H is definite

           ELSE
             shifted_structure = .FALSE.
           END IF
         END IF
       END IF
!write(6,*) ' shifted structure = ', shifted_structure

!  H is indefinite. Shift the diagonals so that the H + shifted S is definite,
!  using a separate H data structure, data%H_shift

       IF ( shifted_structure ) THEN
         data%shifted = .TRUE.
         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )

!  build the data structure for the shifted H. Start by counting the number of
!  nonzeros in the lower triangle of H

         SELECT CASE ( SMT_get( H%type ) )
         CASE ( 'ZERO', 'NONE' )
           data%h_ne = 0
         CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
           data%h_ne = n
         CASE ( 'DENSE' )
           data%h_ne = ( n * ( n + 1 ) ) / 2
         CASE ( 'SPARSE_BY_ROWS' )
           data%h_ne = H%ptr( n + 1 ) - 1
         CASE ( 'COORDINATE' )
           data%h_ne = H%ne
         END SELECT

!  do the same for S

         IF ( data%unit_s ) THEN
           data%s_ne = n
         ELSE
           SELECT CASE ( SMT_get( S%type ) )
           CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
             data%s_ne = n
           CASE ( 'DENSE' )
             data%s_ne = ( n * ( n + 1 ) ) / 2
           CASE ( 'SPARSE_BY_ROWS' )
             data%s_ne = S%ptr( n + 1 ) - 1
           CASE ( 'COORDINATE' )
             data%s_ne = S%ne
           END SELECT
         END IF

!  print statistics about H and S if desired

         IF ( printd ) THEN
           WRITE( out, "( A, ' ||H|| = ', ES10.4 )" )                          &
             prefix, MAXVAL( ABS( H%val( : data%h_ne ) ) )
           IF ( .NOT. data%unit_s ) WRITE( out, "( A, ' ||S|| = ', ES10.4 )" ) &
             prefix, MAXVAL( ABS( S%val( : data%s_ne ) ) )
         END IF

!  allocate space to hold the matrix in co-ordinate form

          data%H_shift%n = n
          data%H_shift%ne = data%h_ne + data%s_ne

         array_name = 'nrek: H_shift%row'
         CALL SPACE_resize_array( data%H_shift%ne, data%H_shift%row,           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: H_shift%col'
         CALL SPACE_resize_array( data%H_shift%ne, data%H_shift%col,           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'nrek: H_shift%val'
         CALL SPACE_resize_array( data%H_shift%ne, data%H_shift%val,           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         CALL SMT_put( data%H_shift%type, 'COORDINATE', inform%alloc_status )
         IF ( inform%alloc_status /= 0 ) THEN
           inform%status = GALAHAD_error_allocate
           GO TO 910
         END IF

!  fit the structural data from H into the coordinate storage scheme provided

         SELECT CASE ( SMT_get( H%type ) )
         CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
           DO i = 1, n
             data%H_shift%row( i ) = i ; data%H_shift%col( i ) = i
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, n
             DO j = 1, i
               l = l + 1
               data%H_shift%row( l ) = i ; data%H_shift%col( l ) = j
             END DO
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, n
             DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
               data%H_shift%row( l ) = i
               data%H_shift%col( l ) = H%col( l )
             END DO
           END DO
         CASE ( 'COORDINATE' )
           data%H_shift%row( : data%h_ne ) = H%row( : data%h_ne )
           data%H_shift%col( : data%h_ne ) = H%col( : data%h_ne )
         END SELECT

!  append the structural data from S into the same scheme if required

         IF ( .NOT. data%unit_s ) THEN
           SELECT CASE ( SMT_get( S%type ) )
           CASE ( 'DIAGONAL' )
             DO i = 1, n
               data%H_shift%row( data%h_ne + i ) = i
               data%H_shift%col( data%h_ne + i ) = i
             END DO
           CASE ( 'DENSE' )
             l = data%h_ne
             DO i = 1, n
               DO j = 1, i
                 l = l + 1
                 data%H_shift%row( l ) = i ; data%H_shift%col( l ) = j
               END DO
             END DO
           CASE ( 'SPARSE_BY_ROWS' )
             DO i = 1, n
               DO l = S%ptr( i ), S%ptr( i + 1 ) - 1
                 data%H_shift%row( data%h_ne + l ) = i
                 data%H_shift%col( data%h_ne + l ) = S%col( l )
               END DO
             END DO
           CASE ( 'COORDINATE' )
             data%H_shift%row( data%h_ne + 1 : data%h_ne + data%s_ne )         &
               = S%row( : data%s_ne )
             data%H_shift%col( data%h_ne + 1 : data%h_ne + data%s_ne )         &
               = S%col( : data%s_ne )
           END SELECT

!  otherwise append the data for shift * I

         ELSE
           DO i = 1, n
             data%H_shift%row( data%h_ne + i ) = i
             data%H_shift%col( data%h_ne + i ) = i
           END DO
         END IF
       END IF

!  if necessary, compute the Gershgorin shift, shift_val

       IF ( data%shifted ) THEN
         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
         IF ( data%unit_s ) THEN
           CALL NREK_find_shift( H, data%shift_val, data%U, shift_status )
         ELSE
           CALL NREK_find_shift( H, data%shift_val, data%U, shift_status,      &
                                 S = S, S_diag = data%Z, S_offd = X )
         END IF
         IF ( shift_status /= 0 ) THEN
           IF ( printe ) WRITE( error, "( A, ' S matrix is inappropriate' )" ) &
             prefix
           inform%status = GALAHAD_error_preconditioner  ; GO TO 920
         END IF
         IF ( printi ) WRITE( out, "( A, ' perturbing H ...',                  &
        &    ' Gershgorin shift =', ES11.4 )" ) prefix, data%shift_val

!  introduce the numerical values from H ...

         IF ( SMT_get( H%type ) == 'IDENTITY' ) THEN
           data%H_shift%val( : data%h_ne ) = one
         ELSE IF ( SMT_get( H%type ) == 'SCALED_IDENTITY' ) THEN
           data%H_shift%val( : data%h_ne ) = H%val( 1 )
         ELSE
           data%H_shift%val( : data%h_ne ) = H%val( : data%h_ne )
         END IF

!  ... and from shift * S

         IF ( data%unit_s ) THEN
           data%H_shift%val( data%h_ne + 1 : data%H_shift%ne ) = data%shift_val
         ELSE IF ( SMT_get( S%type ) == 'SCALED_IDENTITY' ) THEN
           data%H_shift%val( data%h_ne + 1 : data%H_shift%ne ) =               &
             data%shift_val * S%val( 1 )
         ELSE
           data%H_shift%val( data%h_ne + 1 : data%H_shift%ne ) =               &
             data%shift_val * S%val( 1 : data%s_ne )
         END IF

         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%assemble = inform%time%assemble + time_now - time_record
         inform%time%clock_assemble =                                          &
           inform%time%clock_assemble + clock_now - clock_record

!  deallocate all arrays previously allocated within SLS

         CALL SLS_terminate( data%SLS_data, control%SLS_control,               &
                             inform%SLS_inform )

         CALL SLS_initialize( control%linear_solver, data%sls_data,            &
                              data%control%sls_control,                        &
                              inform%sls_inform, check = .TRUE. )
!        data%control%sls_control%print_level = 3
         data%control%sls_control%pivot_control = 2

!  re-analyse the shifted H

         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
         CALL SLS_analyse( data%H_shift, data%sls_data,                        &
                           data%control%sls_control, inform%sls_inform )
         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%analyse = inform%time%analyse + time_now - time_record
         inform%time%clock_analyse =                                           &
           inform%time%clock_analyse + clock_now - clock_record
         IF ( inform%sls_inform%status < 0 ) THEN
           inform%status = GALAHAD_error_analysis ; GO TO 920
         END IF

!  factorize the shifted H, while checking again that it is definite

         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
         CALL SLS_factorize( data%H_shift, data%sls_data,                      &
                             data%control%sls_control, inform%sls_inform )
         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%factorize = inform%time%factorize + time_now - time_record
         inform%time%clock_factorize =                                         &
           inform%time%clock_factorize + clock_now - clock_record
         IF ( inform%sls_inform%status < 0 ) THEN
           WRITE( 6, "( ' H still indefinite after perturbation' )" )
           inform%status = GALAHAD_error_factorization ; GO TO 920
         END IF
       END IF

!  compute ||c|| and record if it is zero

       data%c_norm = TWO_NORM( C )
       data%c_norm_eq_0 = TWO_NORM( C ) == zero

!  check to see whether to pertub c to try to avoid the hard case

       data%perturb_c = control%perturb_c .OR. data%c_norm_eq_0

!  if desired, perturb C (and store the perturbed version in C_pert)

       IF ( data%perturb_c ) THEN
         array_name = 'nrek: data%C_pert'
         CALL SPACE_resize_array( n, data%C_pert,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  make a random perturbation of C

         DO i = 1, n
           CALL RAND_random_real( data%seed, .FALSE., delta )
           data%C_pert( i ) = C( i ) + MAX( one, ABS( C( i ) ) ) * small * delta
         END DO

!  record its norm

         data%c_norm = TWO_NORM( data%C_pert )
       END IF

!  re-entry
!  --------

     ELSE

!  initialise the shift as that from the previous solve

       shift = data%last_shift
     END IF

!  ****************************************************************************
!
!  build and use the backward extended Krylov space
!
!    K_2k+1 = {c, S H^{-1} c, H S^{-1} c, ..., (S H^{-1})^k c, (H S^{-1})^k c},
!
!  Do this by building an orthogonal basis matrix
!
!    V_2k+1 = [ v_0, v_{-1}, v_1, ... , v_{-k}, v_k ]
!
!  and the pentadiagonal projected matrix P_2k = V_2k' A V_2k, with
!  columns p_i, whose subdiagonal columns are
!
!    p_2k-1 = ( ( 1 - beta_{-k+1} delta_k - alpha_k delta_{-k} ) / beta_{k-1} )
!             (                     alpha_{k-1}                               )
!             (           - delta_{-k} delta_{k+1} /beta_{k-1}                )
!
!  and
!
!    p_2k = ( alpha_{-k} )
!           (  delta_k   )
!           (     0      )
!
!  where the alpha_i, beta_i and delta_i are defined below
!
!  For details, see
!    H. Al Daas & N. I. M. Gould
!    Extended-Krylov-subspace methods for trust-region
!    and norm-regularization subproblems
!    Working note STFC-RAL 2025
!
!  ****************************************************************************

     IF ( new_vals ) THEN

!  compute the Newton step x = - H^{-1} c

       IF ( data%perturb_c ) THEN
         X = - data%C_pert
       ELSE
         X = - C
       END IF
       CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
       IF ( data%shifted ) THEN
         CALL SLS_solve( data%H_shift, X, data%SLS_data,                       &
                         data%control%SLS_control, inform%SLS_inform )
       ELSE
         CALL SLS_solve( H, X, data%SLS_data,                                  &
                         data%control%SLS_control, inform%SLS_inform )
       END IF
       CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
       inform%time%solve = inform%time%solve + time_now - time_record
       inform%time%clock_solve =                                               &
         inform%time%clock_solve + clock_now - clock_record
       IF ( inform%SLS_inform%status < 0 ) THEN
         IF ( printe ) WRITE( error, "( A, ' solve involving H failed, ',      &
        & 'SLS inform%status = ', I0 )" ) prefix, inform%SLS_inform%status
         inform%status = GALAHAD_error_solve ; GO TO 920
       END IF

!  compute u = S x (if necessary) and ||x||_S

       IF ( data%unit_s ) THEN
         x_norm = TWO_NORM( X )
       ELSE
         CALL NREK_Mv( one, S, X, zero, data%U, out, error,                    &
                       symmetric = .TRUE. )
         x_norm = SQRT( DOT_PRODUCT( data%U, X ) )
       END IF

       inform%next_weight = weight * data%control%increase

!  segment for the simple case where S is I

       IF ( data%unit_s ) THEN

!  set delta_0 = ||c||

         data%delta_k = data%c_norm

!  set v_0 = - c / delta_0 and u = x / delta_0

         IF ( data%perturb_c ) THEN
           data%V( 1 : n, 0 ) = - data%C_pert / data%c_norm
         ELSE
           data%V( 1 : n, 0 ) = - C / data%c_norm
         END IF
         data%U = X / data%c_norm

!  compute ||u||^2 (in ztu)

!        data%ztu = DOT_PRODUCT( data%U, data%U )
         data%ztu = ( x_norm / data%c_norm ) ** 2

!  set beta_0 = u' v_0

         beta = DOT_PRODUCT( data%U, data%V( 1 : n, 0 ) )
         data%beta_k = beta

!  set where u = u - beta_0 v_0

         data%U = data%U - beta * data%V( 1 : n, 0 )

!  set delta_{-1} = ||u||

         data%delta_mk = TWO_NORM( data%U )

!  segment for the general case where S is not I

       ELSE

!  set w = S^{-1} c

         IF ( data%perturb_c ) THEN
           data%W = data%C_pert
         ELSE
           data%W = C
         END IF
         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
         CALL SLS_solve( S, data%W, data%SLS_S_data,                           &
                         data%control%SLS_S_control, inform%SLS_S_inform )
         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%solve = inform%time%solve + time_now - time_record
         inform%time%clock_solve =                                             &
           inform%time%clock_solve + clock_now - clock_record
         IF ( inform%SLS_S_inform%status < 0 ) THEN
           IF ( printe ) WRITE( error, "( A, ' solve involving S failed, ',    &
          & 'SLS inform%status = ', I0 )" ) prefix, inform%SLS_S_inform%status
           inform%status = GALAHAD_error_solve ; GO TO 920
         END IF

!  set delta_0 = sqrt(c' w)

         IF ( data%perturb_c ) THEN
           data%c_norm = SQRT( DOT_PRODUCT( data%W, data%C_pert ) )
         ELSE
           data%c_norm = SQRT( DOT_PRODUCT( data%W, C ) )
         END IF
         data%delta_k = data%c_norm

!  set v_0 = - w / delta_0

         data%V( 1 : n, 0 ) = - data%W / data%c_norm

!  set q_0 = - b / delta_0

         IF ( data%perturb_c ) THEN
           data%Qp = - data%C_pert / data%c_norm
         ELSE
           data%Qp = - C / data%c_norm
         END IF

!  reset w = u / delta_0

         data%W = data%U / data%c_norm

!  reset u = x / delta_0

         data%U = X / data%c_norm

!  set beta_0 = u' q_0

         beta = DOT_PRODUCT( data%U, data%Qp )
         data%beta_k = beta

!  set where u = u - beta_0 v_0

         data%U = data%U - beta * data%V( 1 : n, 0 )

!  set w = S u

         CALL NREK_Mv( one, S, data%U, zero, data%W, out, error,               &
                       symmetric = .TRUE. )

!  set delta_{-1} = sqrt( w' u )

         data%delta_mk = SQRT( DOT_PRODUCT( data%W, data%U ) )
       END IF

!  initialise the shift as zero

       shift = zero

!  start the iteration loop from the begining

       k_start = 1

!  start from the previous iteration data%k_exit, as data up to then has
!  already been generated

     ELSE
       k_start = data%k_exit
     END IF

!  ------------------------------------------------------------
!  start of main forward iteration loop (in comments, iter = k)
!  ------------------------------------------------------------

     inform%status = GALAHAD_error_max_iterations

     DO k = k_start, data%it_max
       inform%iter = k
       km1 = k - 1 ; mkp1 = - k + 1 ; k2 = 2 * k
       k2m1 = k2 - 1 ; k2p1 = k2 + 1
       printh = printi .AND. ( k == 1 .OR.                                     &
                               control%SLS_control%print_level > 0 .OR.        &
                               control%SLS_S_control%print_level > 0 .OR.      &
                               control%RQS_control%print_level > 0 )

!  print iteration details if required

       IF ( printm ) WRITE( out, "( A, ' iteration ', I0 )" ) prefix, k

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  segment involving the product with H
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

       IF ( data%re_enter == 0 ) THEN  ! segment A
         IF ( printd ) WRITE( out, "( A, ' *** segment B' )" ) prefix
         IF ( ABS( data%delta_mk ) >= delta_tiny ) THEN

!  segment for the simple case where S is I

           IF ( data%unit_s ) THEN

!  set v_{-k} = u / delta_{-k}

             data%V( 1 : n, - k ) = data%U / data%delta_mk

!  set u = H v_{-k}

             IF ( data%shifted ) THEN
               CALL NREK_Mv( one, data%H_shift, data%V( 1 : n, - k ), zero,    &
                             data%U, out, error, symmetric = .TRUE. )
             ELSE
               CALL NREK_Mv( one, H, data%V( 1 : n, - k ), zero,               &
                             data%U, out, error, symmetric = .TRUE. )
             END IF

!  set alpha {k-1} = u' v_{k-1} and u = u - alpha_{k-1} v_{k-1}

             alpha = DOT_PRODUCT( data%U, data%V( 1 : n, km1 ) )
             data%alpha_km1 = alpha
             data%U = data%U - alpha * data%V( 1 : n, km1 )

!  set alpha_{-k} = u' v_{-k} and u = u _ alpha_{-k} v_{-k}

             alpha = DOT_PRODUCT( data%U, data%V( 1 : n, - k ) )
             data%alpha_mk = alpha
             data%U = data%U - alpha * data%V( 1 : n, - k )

!   set delta_{k} = ||u||

             data%delta_s = TWO_NORM( data%U )

!  segment for the general case where S is not I

           ELSE

!  set v_{-k} = u / delta_{-k}

             data%V( 1 : n, - k ) = data%U / data%delta_mk

!  set q_{-k} = w / delta_{-k}

             data%Qm = data%W / data%delta_mk

!  set w = H v_{-k}

             IF ( data%shifted ) THEN
               CALL NREK_Mv( one, data%H_shift, data%V( 1 : n, - k ), zero,    &
                             data%W, out, error, symmetric = .TRUE. )
             ELSE
               CALL NREK_Mv( one, H, data%V( 1 : n, - k ), zero,               &
                             data%W, out, error, symmetric = .TRUE. )
             END IF

!  set u = S^{-1} w

             data%U = data%W
             CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
             CALL SLS_solve( S, data%U, data%SLS_S_data,                       &
                             data%control%SLS_S_control, inform%SLS_S_inform )
             CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
             inform%time%solve = inform%time%solve + time_now - time_record
             inform%time%clock_solve =                                         &
               inform%time%clock_solve + clock_now - clock_record
             IF ( inform%SLS_S_inform%status < 0 ) THEN
               IF ( printe ) WRITE( error,                                     &
                "( A, ' solve involving S failed, SLS inform%status = ',       &
               &   I0 )" ) prefix, inform%SLS_S_inform%status
               inform%status = GALAHAD_error_solve ; GO TO 920
             END IF

!  version 5.2 update

             IF ( control%s_version_52 ) THEN

!  set alpha {k-1} = w' v_{k-1}

               alpha = DOT_PRODUCT( data%W, data%V( 1 : n, km1 ) )
               data%alpha_km1 = alpha

!  set  u = u - alpha_{k-1} v_{k-1}

               data%U = data%U - alpha * data%V( 1 : n, km1 )

!  set  w = w - alpha_{k-1} q_{k-1}

               data%W = data%W - alpha * data%Qp

!  set alpha_{-k} = w' v_{-k}

               alpha = DOT_PRODUCT( data%W, data%V( 1 : n, - k ) )
               data%alpha_mk = alpha

!  set u = u _ alpha_{-k} v_{-k}

               data%U = data%U - alpha * data%V( 1 : n, - k )

!  set  w = w - alpha_{k-1} q_{-k}

               data%W = data%W - alpha * data%Qm

!  version A.3 update

             ELSE

!  set alpha {k-1} = u' q_{k-1} and u = u - alpha_{k-1} v_{k-1}

               alpha = DOT_PRODUCT( data%U, data%Qp )
               data%alpha_km1 = alpha
               data%U = data%U - alpha * data%V( 1 : n, km1 )

!  set alpha_{-k} = u' q_{-k} and u = u _ alpha_{-k} v_{-k}

               alpha = DOT_PRODUCT( data%U, data%Qm )
               data%alpha_mk = alpha
               data%U = data%U - alpha * data%V( 1 : n, - k )

!   set w = S u

               CALL NREK_Mv( one, S, data%U, zero, data%W, out, error,         &
                             symmetric = .TRUE. )
             END IF

!  set delta_{k} = sqrt(u' w)

             data%delta_s = SQRT( DOT_PRODUCT( data%U, data%W ) )
           END IF

!  record delta_{k} and gamma

           data%delta_km1 = data%delta_k
           data%delta_k = data%delta_s
           data%gamma = - data%delta_mk * data%delta_s / data%beta_k

!  special case, v_{-k} = 0 if delta_{-k} = 0

         ELSE
!          data%delta_s = zero
           data%V( 1 : n, - k ) = zero
         END IF

!  save column 2k-1 of P

         IF ( ABS( data%DELTA_mk ) >= delta_tiny ) THEN
           IF ( k > 1 ) THEN
             data%P( 1, k2m1 ) =                                               &
               ( one - data%beta_mk * data%delta_km1                           &
                 - data%delta_mk * data%alpha_km1 ) / data%beta_k
           ELSE
             data%P( 1, k2m1 ) = ( one - data%delta_mk                         &
                                   * data%alpha_km1 ) / data%beta_k
           END IF
           data%P( 2, k2m1 ) = data%alpha_km1
           data%P( 3, k2m1 ) = data%gamma
         ELSE
           IF ( k > 1 ) THEN
             data%P( 1, k2m1 ) = ( one                                         &
               - data%beta_mk * data%delta_km1 ) / data%beta_k
           ELSE
             data%P( 1, k2m1 ) = one / data%beta_k
           END IF
           data%P( 2, k2m1 ) = zero
           data%P( 3, k2m1 ) = zero
         END IF
         IF ( printm .AND. k2m1 == n ) WRITE( out, 2020 ) prefix
!        IF ( k2m1 == n ) WRITE( out, 2020 ) prefix
       END IF

!  ....................--------------------------------
!  optionally test system of order 2k-1 for termination
!  ....................--------------------------------

       IF ( data%re_enter <= 1 ) THEN ! segment B
         IF ( printd ) WRITE( out, "( A, ' *** segment B' )" ) prefix

!  compute the projected solution to ( P + shift I ) s = delta_0 e_1

         IF ( control%stop_check_all_orders .OR.                               &
              data%DELTA_mk == zero ) THEN
           j_max = 1

!  save the lower triangle of P in Q

           data%Q( 1 : k2m1, 1 : k2m1 ) = zero
           DO i = 1, k2m1
             data%Q( i, i ) = data%P( 1, i )
           END DO
           DO i = 1, k2m1 - 1
             data%Q( i + 1, i ) = data%P( 2, i )
           END DO
           DO i = 1, k2m1 - 2
             data%Q( i + 2, i ) = data%P( 3, i )
           END DO

           IF ( printd ) THEN
             WRITE( out, "( ' P = ')" )
             DO i = 1, k2m1
               WRITE( out, "( 4( 2I4, ES12.4 ) )" )                            &
                 ( i, j, data%Q( i, j ), j = 1, i )
             END DO
           END IF

!  compute the eigenvalues (in D) and eigenvalues (overwrite Q)

           CALL SYEV( 'V', 'L', k2m1, data%Q, data%is_max, data%D,             &
                      data%WORK_syev, data%lwork_syev, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  if H has been shifted, unshift the eigenvalues

           IF ( data%shifted ) data%D( : k2m1 )                                &
                                 = data%D( : k2m1 ) - data%shift_val

!  form c' = - Q^T ||c|| e_1

           IF ( data%c_norm_eq_0 ) THEN
             data%C( 1 : k2m1 ) = zero
           ELSE
             data%C( 1 : k2m1 ) = - data%c_norm * data%Q( 1, 1 : k2m1 )
           END IF

!  solve the diagonal norm-regularization problem
!    min 1/2 x'^T D x' + c'^T x' + (weight/power) ||x'||^power

           data%control%RQS_control%use_initial_multiplier = .TRUE.
           data%control%RQS_control%initial_multiplier = shift
           CALL RQS_solve_diagonal( k2m1, power, weight, control%f,            &
                                    data%C, data%D, data%X,                    &
                                    data%control%RQS_control,                  &
                                    inform%RQS_inform )
           shift = inform%RQS_inform%multiplier
           inform%multiplier = shift
           inform%obj = inform%RQS_inform%obj
           inform%obj_regularized = inform%RQS_inform%obj_regularized
           inform%x_norm = inform%RQS_inform%x_norm

!  recover x = Q x'

           data%S1( 1 : k2m1 )                                                 &
             = MATMUL( data%Q( : k2m1 , : k2m1 ), data%X( : k2m1 ) )

!  record ||r_{2k-1}|| = |s_{2k-1}| sqrt{p^2_{2k,2k-1} + p^2_{2k+1,2k-1}}

           IF ( ABS( data%delta_mk ) >= delta_tiny ) THEN
             IF ( k > 1 ) THEN
               error_r = ABS( data%S1( k2m1 ) ) *                              &
                  SQRT( data%P( 2, k2m1 ) ** 2 + data%P( 3, k2m1 ) ** 2 )
             ELSE
               error_r = ABS( data%S1( 1 ) ) *                                 &
                  SQRT( data%P( 2, 1 ) ** 2 + data%P( 3, 1 ) ** 2 )
             END IF
           ELSE
             error_r = zero
           END IF

!  compute ||s||

           s_norm = TWO_NORM( data%S1( 1 : k2m1 ) )
           IF ( printm ) WRITE( out,                                           &
             "(' k = ', I0, ' ||s||, weight, multiplier = ', 3ES11.4 )" )      &
                k, s_norm, weight, shift

!  record ||s|| and the current shift

           inform%x_norm = s_norm
           inform%multiplier = shift

!  debug - compare predicted and actual error

!          IF ( .TRUE. ) THEN
           IF ( printd ) THEN
             SOL = zero
             DO ii = 1, k2m1
               IF ( ii == 1 ) THEN
                 i = 0
               ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
                 i = - i - 1
               ELSE
                 i = - i
               END IF
               SOL = SOL + data%V( : n, i ) * data%S1( ii )
             END DO
             IF ( data%perturb_c ) THEN
               R = data%C_pert
             ELSE
               R = C
             END IF
             IF ( data%unit_s ) THEN
               R = R + shift * SOL
             ELSE
               CALL NREK_Mv( shift, S, SOL, one, R, out, error,                &
                             symmetric = .TRUE. )
             END IF
             CALL NREK_Mv( one, H, SOL, one, R, out, error, symmetric = .TRUE. )
             WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" )                     &
                TWO_NORM( R ), error_r
           END IF

           IF ( printh ) WRITE( out, 2000 ) prefix
           printh = .FALSE.
           IF ( printi ) WRITE( out, 2010 ) prefix, k, k2m1,                   &
                s_norm, weight, shift, error_r, inform%obj

!  check for termination

           error_r = error_r / MAX( one, data%c_norm )
           IF ( error_r < data%control%stop_residual ) THEN
             inform%n_vec = k2m1

!  recover the solution

             X( : n ) = zero
             DO ii = 1, k2m1
               IF ( ii == 1 ) THEN
                 i = 0
               ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
                 i = - i - 1
               ELSE
                 i = - i
               END IF
               X( : n ) = X( : n ) + data%V( : n, i ) * data%S1( ii )
             END DO

!  exit the main loop

             data%k_exit = k
             data%re_enter = - 1
             data%last_shift = shift
             IF ( printd ) WRITE( out, "( A, ' *** exit segment B' )" ) prefix
             inform%status = GALAHAD_ok ; EXIT
           ELSE
             data%re_enter = 0
           END IF
         END IF
       END IF

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!  segment involving the product with H inverse
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

       IF ( data%re_enter == 0 ) THEN ! segment C
         IF ( printd ) WRITE( out, "( A, ' *** segment C' )" ) prefix
         IF ( ABS( data%delta_k ) >= delta_tiny ) THEN

!  segment for the simple case where S is I

           IF ( data%unit_s ) THEN

!  set v_k = u / delta_k

             data%V( 1 : n, k ) = data%U / data%delta_s

!  set u = v_k

             data%U = data%V( 1 : n, k )

!  segment for the general case where S is not I

           ELSE

!  set v_k = u / delta_k

             data%V( 1 : n, k ) = data%U / data%delta_s

!  set  q_k = w / delta_k

             data%Qp = data%W / data%delta_s

!  set u = q_k

             data%U = data%Qp
           END IF

!  orthogonalise u wrt the remaining vectors if desired

           IF ( control%reorthogonalize ) THEN
             IF ( data%unit_s ) THEN
               DO i = mkp1, km1 - 1
                 beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
                 data%U = data%U - beta * data%V( 1 : n, i )
               END DO
               data%delta_s = TWO_NORM( data%U )
             ELSE
               DO i = mkp1, km1 - 1
                 CALL NREK_Mv( one, S, data%V( 1 : n, i ), zero, data%Z,       &
                               out, error, symmetric = .TRUE. )
                 beta = DOT_PRODUCT( data%Z, data%U )
                 data%U = data%U - beta * data%V( 1 : n, i )
               END DO
               CALL NREK_Mv( one, S, data%U, zero, data%Z,                     &
                             out, error, symmetric = .TRUE. )
               data%delta_s = SQRT( DOT_PRODUCT( data%Z, data%U ) )
             END IF
           END IF

!  debug inner products

           IF ( printd ) THEN
             IF ( data%unit_s ) THEN
               WRITE( out, "( ' p ', 3ES22.14 )" )                             &
                 DOT_PRODUCT( data%V( 1 : n, k ), data%V( 1 : n, - k ) ),      &
                 DOT_PRODUCT( data%V( 1 : n, k ), data%V( 1 : n, km1 ) ),      &
                 DOT_PRODUCT( data%V( 1 : n, k ), data%V( 1 : n, k ) )
             ELSE
               CALL NREK_Mv( one, S, data%V( 1 : n, k ), zero, data%Z,         &
                             out, error, symmetric = .TRUE. )
               WRITE( out, "( ' p ', 3ES22.14 )" )                             &
                 DOT_PRODUCT( data%Z, data%V( 1 : n, - k ) ),                  &
                 DOT_PRODUCT( data%Z, data%V( 1 : n, km1 ) ),                  &
                 DOT_PRODUCT( data%Z, data%V( 1 : n, k ) )
             END IF
           END IF

!   reset u = H^{-1} u

           CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
           IF ( data%shifted ) THEN
             CALL SLS_solve( data%H_shift, data%U, data%SLS_data,              &
                             data%control%SLS_control, inform%SLS_inform )
           ELSE
             CALL SLS_solve( H, data%U, data%SLS_data,                         &
                             data%control%SLS_control, inform%SLS_inform )
           END IF
           CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
           inform%time%solve = inform%time%solve + time_now - time_record
           inform%time%clock_solve =                                           &
             inform%time%clock_solve + clock_now - clock_record
           IF ( inform%SLS_inform%status < 0 ) THEN
             IF ( printe ) WRITE( error, "( A, ' solve involving H failed, ',  &
            & 'SLS inform%status = ', I0 )" ) prefix, inform%SLS_inform%status
             inform%status = GALAHAD_error_solve ; GO TO 920
           END IF

!  segment for the simple case where S is I

           IF ( data%unit_s ) THEN

!  set beta_{-k} := u' v_{-k} and u = u - beta_{-k} v_{-k}

             beta = DOT_PRODUCT( data%U, data%V( 1 : n, - k ) )
             data%beta_mk = beta
             data%U = data%U - beta * data%V( 1 : n, - k )

!  set beta k = u' v_k and w := w - beta_k v_k

             beta = DOT_PRODUCT( data%U, data%V( 1 : n, k ) )
             data%beta_k = beta
             data%U = data%U - beta * data%V( 1 : n, k )

!  segment for the general case where S is not I

           ELSE

!  version 5.2 update

             IF ( control%s_version_52 ) THEN

!   set w = S u

               CALL NREK_Mv( one, S, data%U, zero, data%W, out, error,         &
                             symmetric = .TRUE. )

!  set beta_{-k} = w' v_{-k}

               beta = DOT_PRODUCT( data%W, data%V( 1 : n, - k ) )
               data%BETA_mk = beta

!  set u = u - beta_{-k} v_{-k}

               data%U = data%U - beta * data%V( 1 : n, - k )

!  set w = w - beta_{-k} q_{-k}

               data%W = data%W - beta * data%Qm

!  set beta k = w' v_k

               beta = DOT_PRODUCT( data%W, data%V( 1 : n, k ) )
               data%beta_k = beta

!  set u = u - beta_k v_k

               data%U = data%U - beta * data%V( 1 : n, k )

!  set w = w - beta_k q_k

               data%W = data%W - beta * data%Qp

!  version A.3 update

             ELSE

!  set beta_{-k} := u' q_{-k} and u = u - beta_{-k} v_{-k}

               beta = DOT_PRODUCT( data%U, data%Qm )
               data%beta_mk = beta
               data%U = data%U - beta * data%V( 1 : n, - k )

!  set beta k = u' q_k and w := w - beta_k v_k

               beta = DOT_PRODUCT( data%U, data%Qp )
               data%beta_k = beta
               data%U = data%U - beta * data%V( 1 : n, k )
             END IF
           END IF

!  special case, v_k = 0 if delta_k = 0

         ELSE
           data%V( 1 : n, k ) = zero
         END IF

!  save column 2k of P

         data%P( 1, k2 ) = data%alpha_mk
         data%P( 2, k2 ) = data%delta_s
         data%P( 3, k2 ) = zero
         IF ( printm .AND. k2 == n ) WRITE( out, 2020 ) prefix
       END IF

!  ....................-------------------
!  test system of order 2k for termination
!  ....................-------------------

       IF ( data%re_enter <= 2 ) THEN ! segment D
         IF ( printd ) WRITE( out, "( A, ' *** segment D' )" ) prefix

!  compute the projected solution ( P + shift I ) s = delta_0 e_1

         IF ( termination_test .OR. data%delta_k == zero ) THEN
           j_max = 1

!  save the lower triangle of P in Q

           data%Q( 1 : k2, 1 : k2 ) = zero
           DO i = 1, k2
             data%Q( i, i ) = data%P( 1, i )
           END DO
           DO i = 1, k2 - 1
             data%Q( i + 1, i ) = data%P( 2, i )
           END DO
           DO i = 1, k2 - 2
             data%Q( i + 2, i ) = data%P( 3, i )
           END DO

           IF ( printd ) THEN
             WRITE( out, "( ' P = ')" )
             DO i = 1, k2
               WRITE( out, "( 4( 2I4, ES12.4 ) )" )                            &
                 ( i, j, data%Q( i, j ), j = 1, i )
             END DO
           END IF

!  compute the eigenvalues (in D) and eigenvalues (overwrite Q)

           CALL SYEV( 'V', 'L', k2, data%Q, data%is_max, data%D,               &
                      data%WORK_syev, data%lwork_syev, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  if H has been shifted, unshift the eigenvalues

           IF ( data%shifted ) data%D( : k2 ) = data%D( : k2 ) - data%shift_val

!  form c' = - Q^T ||c|| e_1

           IF ( data%c_norm_eq_0 ) THEN
             data%C( 1 : k2 ) = zero
           ELSE
             data%C( 1 : k2 ) = - data%c_norm * data%Q( 1, 1 : k2 )
           END IF

!  solve the diagonal norm-regularization problem
!    min 1/2 x'^T D x' + c'^T x' + (weight/power) ||x'||^power

           data%control%RQS_control%use_initial_multiplier = .TRUE.
           data%control%RQS_control%initial_multiplier = shift
           data%control%RQS_control%initial_multiplier = shift
           CALL RQS_solve_diagonal( k2, power, weight, control%f,              &
                                    data%C, data%D, data%X,                    &
                                    data%control%RQS_control,                  &
                                    inform%RQS_inform )
           shift = inform%RQS_inform%multiplier
           inform%multiplier = shift
           inform%obj = inform%RQS_inform%obj
           inform%obj_regularized = inform%RQS_inform%obj_regularized
           inform%x_norm = inform%RQS_inform%x_norm

!  recover x = Q x'

           data%S1( 1 : k2 ) = MATMUL( data%Q( : k2 , : k2 ), data%X( : k2 ) )

!  record ||r_{2k}|| = | p_{2k+1,2k-1} s_{2k-1} + p_{2k+1,2k} s_{2k} |

           IF ( ABS( data%delta_k ) >= delta_tiny ) THEN
             error_r = ABS( data%gamma * data%S1( k2m1 )                       &
                              + data%delta_s * data%S1( k2 ) )
           ELSE
             error_r = zero
           END IF

!  compute ||s||

           s_norm = TWO_NORM( data%S1( 1 : k2 ) )
           IF ( printm ) WRITE( out,                                           &
             "(' k = ', I0, ' ||s||, weight, multiplier = ', 3ES11.4 )" )      &
                k, s_norm, weight, shift

!  record ||s|| and the current shift

           inform%x_norm = s_norm
           inform%multiplier = shift

!  debug - compare predicted and actual error

!          IF ( .TRUE. ) THEN
           IF ( printd ) THEN
             SOL = zero
             DO ii = 1, k2
               IF ( ii == 1 ) THEN
                 i = 0
               ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
                 i = - i - 1
               ELSE
                 i = - i
               END IF
               SOL = SOL + data%V( : n, i ) * data%S1( ii )
             END DO
             IF ( data%perturb_c ) THEN
               R = data%C_pert
             ELSE
               R = C
             END IF
             IF ( data%unit_s ) THEN
               R = R + shift * SOL
             ELSE
               CALL NREK_Mv( shift, S, SOL, one, R, out, error,                &
                             symmetric = .TRUE. )
             END IF
             CALL NREK_Mv( one, H, SOL, one, R, out, error, symmetric = .TRUE. )
             WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" )                     &
               TWO_NORM( R ), error_r
           END IF

           IF ( printh ) WRITE( out, 2000 ) prefix
           IF ( printi ) WRITE( out, 2010 ) prefix, k, k2, s_norm, weight,     &
                                            shift, error_r, inform%obj

!  check for termination

           error_r = error_r / MAX( one, data%c_norm )
           IF ( error_r < data%control%stop_residual ) THEN
             inform%n_vec = k2

!  recover the solution

             X( : n ) = zero
             DO ii = 1, k2
               IF ( ii == 1 ) THEN
                 i = 0
               ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
                 i = - i - 1
               ELSE
                 i = - i
               END IF
               X( : n ) = X( : n ) + data%V( : n, i ) * data%S1( ii )
             END DO

!  exit the main loop

             data%k_exit = k
             data%re_enter = 2
             data%last_shift = shift
             IF ( printd ) WRITE( out, "( A, ' *** exit segment D' )" ) prefix
             inform%status = GALAHAD_ok ; EXIT
           ELSE
             data%re_enter = 0
           END IF
         END IF

!  skip if this is the last pass

         IF ( k < data%k_max ) THEN

!  set delta_{-k-1} := ||u||

           IF ( data%unit_s ) THEN
               delta = TWO_NORM( data%U )

!   set w = S u if necessary

           ELSE
             IF ( .NOT. control%s_version_52 )                                 &
               CALL NREK_Mv( one, S, data%U, zero, data%W, out, error,         &
                             symmetric = .TRUE. )

!   set delta_{-k-1} = sqrt(u' w)

             delta = SQRT( DOT_PRODUCT( data%U, data%W ) )
           END IF

!  orthogonalise wrt the remaining vectors

           IF ( control%reorthogonalize ) THEN
             IF ( data%unit_s ) THEN
               DO i = mkp1 + 1, k - 1
                 beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
                 data%U = data%U - beta * data%V( 1 : n, i )
               END DO
               delta = TWO_NORM( data%U )
             ELSE
               DO i = mkp1 + 1, k - 1
                 CALL NREK_Mv( one, S, data%V( 1 : n, i ), zero, data%Z,       &
                               out, error, symmetric = .TRUE. )
                 beta = DOT_PRODUCT( data%Z, data%U )
                 data%U = data%U - beta * data%V( 1 : n, i )
               END DO
               CALL NREK_Mv( one, S, data%U, zero, data%Z,                     &
                             out, error, symmetric = .TRUE. )
               delta = SQRT( DOT_PRODUCT( data%Z, data%U ) )
             END IF
           END IF

!  save delta_{-k-1}

           mkm1 = - k - 1
           data%delta_mk = delta

!  debug inner products

           IF ( ABS( data%delta_mk ) >= delta_tiny ) THEN
             IF ( printd ) THEN
               IF ( data%unit_s ) THEN
                 data%V( 1 : n, mkm1 ) = data%U / data%delta_mk
                 WRITE( out, "( ' p ', 3ES22.14 )" )                           &
                   DOT_PRODUCT( data%V( 1 : n, mkm1 ), data%V( 1 : n, - k ) ), &
                   DOT_PRODUCT( data%V( 1 : n, mkm1 ), data%V( 1 : n, k ) ),   &
                   DOT_PRODUCT( data%V( 1 : n, mkm1 ), data%V( 1 : n, mkm1 ) )

!   set z = S v_{-m-1}

               ELSE
                 data%V( 1 : n, mkm1 ) = data%U / data%delta_mk
                 CALL NREK_Mv( one, S, data%V( 1 : n, mkm1 ), zero, data%Z,    &
                               out, error, symmetric = .TRUE. )
                 WRITE( out, "( ' p ', 3ES22.14 )" )                           &
                   DOT_PRODUCT( data%Z, data%V( 1 : n, - k ) ),                &
                   DOT_PRODUCT( data%Z, data%V( 1 : n, k ) ),                  &
                   DOT_PRODUCT( data%Z, data%V( 1 : n, mkm1 ) )
               END IF
             END IF
           END IF
         END IF
       END IF

!  ------------------------------------
!  end of main backward iteration loop
!  ------------------------------------

     END DO
     inform%error = error_r

!  debug - check the theoretical P against the computed one

!    IF  ( .true. ) THEN
     IF  ( .false. ) THEN
!      DO ii = 1, inform%n_vec
       DO ii = 1, p_dim
         IF ( ii == 1 ) THEN
           i = 0
         ELSE IF ( MOD( ii, 2 ) == 0 ) THEN
           i = - ( i + 1 )
         ELSE
           i = - i
         END IF
         IF ( data%shifted ) THEN
           CALL NREK_Mv( one, data%H_shift, data%V( 1 : n, i ), zero,          &
                         data%U, out, error, symmetric = .TRUE. )
         ELSE
           CALL NREK_Mv( one, H, data%V( 1 : n, i ), zero,                     &
                         data%U, out, error, symmetric = .TRUE. )
         END IF
         DO jj = 1, ii
           IF ( jj == 1 ) THEN
             j = 0
           ELSE IF ( MOD( jj, 2 ) == 0 ) THEN
             j = - ( j + 1 )
           ELSE
             j = - j
           END IF
           P_calc( ii, jj ) = DOT_PRODUCT( data%U, data%V( 1 : n, j ) )
         END DO
       END DO
       DO ii = 1, p_dim
         IF ( ii + 2 <= p_dim ) THEN
           WRITE( out, "( ' P(', I3, ') = ', 3ES22.14 )" )                     &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ),                    &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) ),                &
                 ABS( data%P( 3, ii ) - P_calc( ii + 2, ii ) )
         ELSE IF ( ii + 1 <= p_dim ) THEN
           WRITE( out, "( ' P(', I3, ') = ', 2ES22.14 )" )                     &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ),                    &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) )
         ELSE
           WRITE( out, "( ' P(', I3, ') = ', ES22.14 )" )                      &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) )
         END IF
       END DO
     END IF

!  successful return

     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total = inform%time%clock_total + clock_now - clock_start
     RETURN

!  allocation error

 910 CONTINUE
     IF ( printe ) WRITE( error, "( A, '   **  Allocation error return ', I0,  &
    &                               ' from NREK ' )" ) prefix, inform%status
     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total = inform%time%clock_total + clock_now - clock_start
     RETURN

!  other error returns

 920 CONTINUE
     IF ( printe ) WRITE( error, "( A, '   **  Error return ', I0,             &
    &                               ' from NREK ' )" ) prefix, inform%status
     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total = inform%time%clock_total + clock_now - clock_start
     RETURN

!  non-executable statements

2000 FORMAT( A, '    k d(K)   ||x||     weight      shift      error   ',      &
             '         f')
2010 FORMAT( A, 2I5, 4ES11.4, ES18.10 )
2020 FORMAT( A, 1X, 19( '-' ), ' Krylov space is full space ', 18( '-' ) )

!  end of subroutine NREK_solve

     END SUBROUTINE NREK_solve

!-*-*-*-*-   N R E K _ H _ f i n d _ s h i f t   S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE NREK_find_shift( H, shift, H_low, status, S, S_diag, S_offd )

!  find a (small) shift so that H + shift S is positive definite, where
!  S is I if the arguments S is absent

!  On exit, status indicates the return status
!    status = 0, the required shift is given
!    status = - 1, S is present but S_diag and S_offd are absent
!    status = - 2, S is not strictly diagonally dominant

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ),  INTENT( OUT ) :: status
     TYPE ( SMT_type ), INTENT( IN ) :: H
     REAL ( KIND = rp_), INTENT( OUT ) :: shift
     REAL ( KIND = rp_), INTENT( OUT ), DIMENSION( H%n ) :: H_low
     TYPE ( SMT_type ), OPTIONAL, INTENT( IN ) :: S
     REAL ( KIND = rp_), OPTIONAL, INTENT( OUT ), DIMENSION( H%n ) :: S_diag
     REAL ( KIND = rp_), OPTIONAL, INTENT( OUT ), DIMENSION( H%n ) :: S_offd

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, j, l
     REAL ( KIND = rp_) :: h_max, s_max, val, abs_val
     LOGICAL :: unit_s

     status = 0

!  decide whether to compute a Gershgorin or a generalized-Gershgorin bound

     IF ( PRESENT( S ) ) THEN
       unit_s = SMT_get( S%type ) == 'IDENTITY' .OR.                           &
              ( SMT_get( S%type ) == 'SCALED_IDENTITY' .AND. S%val( 1 ) == one )
     ELSE
       unit_s = .TRUE.
     END IF

     IF ( .NOT. unit_s ) THEN

!  if S is present, check to see if S_diag and S_offd are also present

       IF ( PRESENT( S_diag ) .AND. PRESENT( S_offd ) ) THEN

!  compute the sums of the absolute values of its off-diagonal terms
!  (in S_offd), its diagonal terms (in S_diag) and the larger of one and
!  the largest absolute value (in s_max)

         S_diag = zero ; S_offd = zero
         SELECT CASE ( SMT_get( S%type ) )
         CASE ( 'SCALED_IDENTITY' )
           S_diag = S%val( 1 )
           s_max = MAX( one, ABS( S%val( 1 ) ) )
         CASE ( 'DIAGONAL' )
           S_diag = S%val( : S%n )
           s_max = MAX( one, MAXVAL( ABS( S%val( : S%n ) ) ) )
         CASE ( 'COORDINATE' )
           s_max = one
           DO l = 1, S%ne
             i =  S%row( l ) ; j = S%col( l ) ; val =  S%val( l )
             abs_val = ABS( val ) ; s_max = MAX( s_max, abs_val )
             IF ( i == j ) THEN
               S_diag( i ) = S_diag( i ) + val
             ELSE
               S_offd( i ) = S_offd( i ) + abs_val
               S_offd( j ) = S_offd( j ) + abs_val
             END IF
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           s_max = one
           DO i = 1, S%n
             DO l = S%ptr( i ), S%ptr( i + 1 ) - 1
               j = S%col( l ) ; val =  S%val( l )
               abs_val = ABS( val ) ; s_max = MAX( s_max, abs_val )
               IF ( i == j ) THEN
                 S_diag( i ) = S_diag( i ) + val
               ELSE
                 S_offd( i ) = S_offd( i ) + abs_val
                 S_offd( j ) = S_offd( j ) + abs_val
               END IF
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0 ; s_max = one
           DO i = 1, S%n
             DO j = 1, i
               l = l + 1 ; val =  S%val( l )
               abs_val = ABS( val ) ; s_max = MAX( s_max, abs_val )
               IF ( i == j ) THEN
                 S_diag( i ) = S_diag( i ) + val
               ELSE
                 S_offd( i ) = S_offd( i ) + abs_val
                 S_offd( j ) = S_offd( j ) + abs_val
               END IF
             END DO
           END DO
         END SELECT

!  check to see that S is strictly diagonally dominant

         DO i = 1, S%n
           IF ( S_diag( i ) <= S_offd( i ) ) THEN
             status = 2 ; RETURN
           END IF
         END DO
       ELSE
         status = 1 ; RETURN
       END IF
     END IF

!  H_low gives the lower bound on each Gershgorin interval for H, and
!  h_max gives the larger of one and the largest absolute value in H

     H_low( : H%n ) = zero
     SELECT CASE ( SMT_get( H%type ) )
     CASE ( 'ZERO', 'NONE' )
       h_max = one
     CASE ( 'SCALED_IDENTITY' )
       H_low( : H%n ) = H%val( 1 )
       h_max = MAX( one, ABS( H%val( 1 ) ) )
     CASE ( 'DIAGONAL' )
       H_low( : H%n ) = H%val( : H%n )
       h_max = MAX( one, MAXVAL( ABS( H%val( : H%n ) ) ) )
     CASE ( 'COORDINATE' )
       h_max = one
       DO l = 1, H%ne
         i =  H%row( l ) ; j = H%col( l ) ; val = H%val( l )
         abs_val = ABS( val ) ; h_max = MAX( h_max, abs_val )
         IF ( i == j ) THEN
           H_low( i ) = H_low( i ) + val
         ELSE
           H_low( i ) = H_low( i ) - abs_val
           H_low( j ) = H_low( j ) - abs_val
         END IF
       END DO
     CASE ( 'SPARSE_BY_ROWS' )
       h_max = one
       DO i = 1, H%n
         DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
           j = H%col( l ) ; val =  H%val( l )
           abs_val = ABS( val ) ; h_max = MAX( h_max, abs_val )
           IF ( i == j ) THEN
             H_low( i ) = H_low( i ) + val
           ELSE
             H_low( i ) = H_low( i ) - abs_val
             H_low( j ) = H_low( j ) - abs_val
           END IF
         END DO
       END DO
     CASE ( 'DENSE' )
       l = 0 ; h_max = one
       DO i = 1, H%n
         DO j = 1, i
           l = l + 1 ; val =  H%val( l )
           abs_val = ABS( val ) ; h_max = MAX( h_max, abs_val )
           IF ( i == j ) THEN
             H_low( i ) = H_low( i ) + val
           ELSE
             H_low( i ) = H_low( i ) - abs_val
             H_low( j ) = H_low( j ) - abs_val
           END IF
         END DO
       END DO
     END SELECT

!  compute the Gershgorin bound on the eigenvalues of H

     IF ( unit_s ) THEN

!  the required shift is minus the lowest lower interval bound

       shift = - MINVAL( H_low( : H%n ) )

!  add a very small perturbation

       shift = shift + h_pert * h_max

!  compute the Gershgorin-like lower bound on the matrix pencil (H,S)

     ELSE
        shift = infinity
        DO i = 1, H%n
          shift = MIN( shift, H_low( i ) / ( S_diag( i ) + S_offd( i ) ),      &
                              H_low( i ) / ( S_diag( i ) - S_offd( i ) ) )
        END DO

!  the required shift is minus the lowest lower interval bound

       shift = - shift

!  add a very small perturbation

       shift = shift + MAX( h_pert * h_max, s_pert * s_max )
     END IF

!  end of SUBROUTINE NREK_find_shift

     END SUBROUTINE NREK_find_shift

! -*-*-  G A L A H A D -  N R E K _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE NREK_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine NREK_initialize
!   control see Subroutine NREK_initialize
!   inform  see Subroutine NREK_find

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( NREK_data_type ), INTENT( INOUT ) :: data
      TYPE ( NREK_control_type ), INTENT( IN ) :: control
      TYPE ( NREK_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

      data%allocated_arrays = .FALSE.
      data%k_exit = - 1
      data%re_enter = 0
      data%unit_s = .TRUE.

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      inform%status = inform%SLS_inform%status
      IF ( inform%SLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'nrek: data%SLS'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaing allocated arrays

      array_name = 'nrek: data%MAP'
      CALL SPACE_dealloc_array( data%MAP,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%P_shift'
      CALL SPACE_dealloc_array( data%P_shift,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%V'
      CALL SPACE_dealloc_array( data%V,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%S1'
      CALL SPACE_dealloc_array( data%S1,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%S2'
      CALL SPACE_dealloc_array( data%S2,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%Q'
      CALL SPACE_dealloc_array( data%Q,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%D'
      CALL SPACE_dealloc_array( data%D,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%X'
      CALL SPACE_dealloc_array( data%X,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%Z'
      CALL SPACE_dealloc_array( data%Z,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%C_pert'
      CALL SPACE_dealloc_array( data%C_pert,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%Qp'
      CALL SPACE_dealloc_array( data%Qp,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%Qm'
      CALL SPACE_dealloc_array( data%Qm,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%WORK_syev'
      CALL SPACE_dealloc_array( data%WORK_syev,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%H_shift%ptr'
      CALL SPACE_dealloc_array( data%H_shift%ptr,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%H_shift%col'
      CALL SPACE_dealloc_array( data%H_shift%col,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'nrek: data%H_shift%val'
      CALL SPACE_dealloc_array( data%H_shift%val,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      IF ( ALLOCATED( data%H_shift%type ) )                                    &
        DEALLOCATE( data%H_shift%type, STAT = inform%alloc_status )

      RETURN

!  End of subroutine NREK_terminate

      END SUBROUTINE NREK_terminate

! -  G A L A H A D -  N R E K _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE NREK_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NREK_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NREK_control_type ), INTENT( IN ) :: control
     TYPE ( NREK_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

!    CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL NREK_terminate( data%nrek_data, control, inform )

!  deallocate any internal problem arrays

!    array_name = 'nrek: data%prob%X'
!    CALL SPACE_dealloc_array( data%prob%X,                                    &
!       inform%status, inform%alloc_status, array_name = array_name,           &
!       bad_alloc = inform%bad_alloc, out = control%error )
!    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine NREK_full_terminate

     END SUBROUTINE NREK_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-  G A L A H A D -  N R E K _ i m p o r t _ S U B R O U T I N E -*-*-*-*-

     SUBROUTINE NREK_import( control, data, status, n,                         &
                             H_type, H_ne, H_row, H_col, H_ptr )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to NREK_solve
!
!  data is a scalar variable of type NREK_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0 or requirement that the types contain
!       a relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' or 'IDENTITY' has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  H_type is a character string that specifies the Hessian storage scheme
!   used. It should be one of 'coordinate', 'sparse_by_rows', 'dense' or
!   'diagonal'. Lower or upper case variants are allowed.
!
!  H_ne is a scalar variable of type default integer, that holds the number of
!   entries in the  lower triangular part of H in the sparse co-ordinate
!   storage scheme. It need not be set for any of the other schemes.
!
!  H_row is a rank-one array of type default integer, that holds
!   the row indices of the  lower triangular part of H in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other
!   three schemes, and in this case can be of length 0
!
!  H_col is a rank-one array of type default integer,
!   that holds the column indices of the  lower triangular part of H in either
!   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!   be set when the dense, diagonal, scaled identity, identity or zero schemes
!   are used, and in this case can be of length 0
!
!  H_ptr is a rank-one array of dimension n+1 and type default
!   integer, that holds the starting position of  each row of the lower
!   triangular part of H, as well as the total number of entries plus one,
!   in the sparse row-wise storage scheme. It need not be set when the
!   other schemes are used, and in this case can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NREK_control_type ), INTENT( INOUT ) :: control
     TYPE ( NREK_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
     INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: H_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: H_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: H_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: H_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: H_ptr

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     WRITE( control%out, "( '' )", ADVANCE = 'no' ) ! prevents ifort bug
     data%nrek_control = control

     error = data%nrek_control%error
     space_critical = data%nrek_control%space_critical
     deallocate_error_fatal = data%nrek_control%space_critical

!  flag that S and A are not currently used

     data%use_s = .FALSE.

!  set H appropriately in its storage type

     SELECT CASE ( H_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( H_row ) .AND. PRESENT( H_col ) ) ) THEN
         data%nrek_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%H%type, 'COORDINATE', data%nrek_inform%alloc_status )
       data%H%n = n ; data%H%ne = H_ne

       array_name = 'nrek: data%H%row'
       CALL SPACE_resize_array( data%H%ne, data%H%row,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       array_name = 'nrek: data%H%col'
       CALL SPACE_resize_array( data%H%ne, data%H%col,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       array_name = 'nrek: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%H%row( : data%H%ne ) = H_row( : data%H%ne )
         data%H%col( : data%H%ne ) = H_col( : data%H%ne )
       ELSE
         data%H%row( : data%H%ne ) = H_row( : data%H%ne ) + 1
         data%H%col( : data%H%ne ) = H_col( : data%H%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( H_col ) .AND. PRESENT( H_ptr ) ) ) THEN
         data%nrek_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%H%type, 'SPARSE_BY_ROWS',                            &
                     data%nrek_inform%alloc_status )
       data%H%n = n
       IF ( data%f_indexing ) THEN
         data%H%ne = H_ptr( n + 1 ) - 1
       ELSE
         data%H%ne = H_ptr( n + 1 )
       END IF

       array_name = 'nrek: data%H%ptr'
       CALL SPACE_resize_array( n + 1, data%H%ptr,                             &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       array_name = 'nrek: data%H%col'
       CALL SPACE_resize_array( data%H%ne, data%H%col,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       array_name = 'nrek: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%H%ptr( : n + 1 ) = H_ptr( : n + 1 )
         data%H%col( : data%H%ne ) = H_col( : data%H%ne )
       ELSE
         data%H%ptr( : n + 1 ) = H_ptr( : n + 1 ) + 1
         data%H%col( : data%H%ne ) = H_col( : data%H%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%H%type, 'DENSE', data%nrek_inform%alloc_status )
       data%H%n = n ; data%H%ne = ( n * ( n + 1 ) ) / 2

       array_name = 'nrek: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

     CASE ( 'diagonal', 'DIAGONAL' )
       CALL SMT_put( data%H%type, 'DIAGONAL', data%nrek_inform%alloc_status )
       data%H%n = n ; data%H%ne = n

       array_name = 'nrek: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%nrek_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     status = GALAHAD_ok
     RETURN

!  error returns

 900 CONTINUE
     status = data%nrek_inform%status
     RETURN

!  End of subroutine NREK_import

     END SUBROUTINE NREK_import

!-*-*-  G A L A H A D -  N R E K _ i m p o r t _ S _ S U B R O U T I N E -*-*-

     SUBROUTINE NREK_s_import( data, status, S_type, S_ne, S_row, S_col, S_ptr )

!  import fixed problem data for the scaling matrix S into internal
!  storage prior to solution. Arguments are as follows:

!  data is a scalar variable of type NREK_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The requirement that the types contain a relevant string 'DENSE', 
!       'COORDINATE', 'SPARSE_BY_ROWS', 'DIAGONAL' or 'IDENTITY' 
!       has been violated.
!
!  S_type is a character string that specifies the scaling matrix storage scheme
!   used. It should be one of 'coordinate', 'sparse_by_rows', 'dense',
!   'diagonal' or 'identity'. Lower or upper case variants are allowed.
!
!  S_ne is a scalar variable of type default integer, that holds the number of
!   entries in the lower triangular part of S in the sparse co-ordinate
!   storage scheme. It need not be set for any of the other schemes.
!
!  S_row is a rank-one array of type default integer, that holds
!   the row indices of the  lower triangular part of S in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other
!   three schemes, and in this case can be of length 0
!
!  S_col is a rank-one array of type default integer,
!   that holds the column indices of the  lower triangular part of S in either
!   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!   be set when the dense, diagonal, scaled identity, identity or zero schemes
!   are used, and in this case can be of length 0
!
!  S_ptr is a rank-one array of dimension n+1 and type default
!   integer, that holds the starting position of  each row of the lower
!   triangular part of M, as well as the total number of entries plus one,
!   in the sparse row-wise storage scheme. It need not be set when the
!   other schemes are used, and in this case can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NREK_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: S_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: S_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: S_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: S_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: S_ptr

!  local variables

     INTEGER ( KIND = ip_ ) :: n, error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     WRITE( data%nrek_control%out, "('')", ADVANCE = 'no' ) ! prevents ifort bug
     error = data%nrek_control%error
     space_critical = data%nrek_control%space_critical
     deallocate_error_fatal = data%nrek_control%space_critical

!  recover the dimension

     n = data%H%n

!  set S appropriately in its storage type

     SELECT CASE ( S_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( S_ne ) .AND. PRESENT( S_row ) .AND.               &
                    PRESENT( S_col ) ) ) THEN
         data%nrek_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%S%type, 'COORDINATE', data%nrek_inform%alloc_status )
       data%S%n = n ; data%S%ne = S_ne

       array_name = 'nrek: data%S%row'
       CALL SPACE_resize_array( data%S%ne, data%S%row,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       array_name = 'nrek: data%S%col'
       CALL SPACE_resize_array( data%S%ne, data%S%col,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       array_name = 'nrek: data%S%val'
       CALL SPACE_resize_array( data%S%ne, data%S%val,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%S%row( : data%S%ne ) = S_row( : data%S%ne )
         data%S%col( : data%S%ne ) = S_col( : data%S%ne )
       ELSE
         data%S%row( : data%S%ne ) = S_row( : data%S%ne ) + 1
         data%S%col( : data%S%ne ) = S_col( : data%S%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( S_col ) .AND. PRESENT( S_ptr ) ) ) THEN
         data%nrek_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%S%type, 'SPARSE_BY_ROWS',                            &
                     data%nrek_inform%alloc_status )
       data%S%n = n
       IF ( data%f_indexing ) THEN
         data%S%ne = S_ptr( n + 1 ) - 1
       ELSE
         data%S%ne = S_ptr( n + 1 )
       END IF

       array_name = 'nrek: data%S%ptr'
       CALL SPACE_resize_array( n + 1, data%S%ptr,                             &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       array_name = 'nrek: data%S%col'
       CALL SPACE_resize_array( data%S%ne, data%S%col,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       array_name = 'nrek: data%S%val'
       CALL SPACE_resize_array( data%S%ne, data%S%val,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%S%ptr( : n + 1 ) = S_ptr( : n + 1 )
         data%S%col( : data%S%ne ) = S_col( : data%S%ne )
       ELSE
         data%S%ptr( : n + 1 ) = S_ptr( : n + 1 ) + 1
         data%S%col( : data%S%ne ) = S_col( : data%S%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%S%type, 'DENSE', data%nrek_inform%alloc_status )
       data%S%n = n ; data%S%ne = ( n * ( n + 1 ) ) / 2

       array_name = 'nrek: data%S%val'
       CALL SPACE_resize_array( data%S%ne, data%S%val,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

     CASE ( 'diagonal', 'DIAGONAL' )
       CALL SMT_put( data%S%type, 'DIAGONAL', data%nrek_inform%alloc_status )
       data%S%n = n ; data%S%ne = n

       array_name = 'nrek: data%S%val'
       CALL SPACE_resize_array( data%S%ne, data%S%val,                         &
              data%nrek_inform%status, data%nrek_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%nrek_inform%bad_alloc, out = error )
       IF ( data%nrek_inform%status /= 0 ) GO TO 900

     CASE ( 'identity', 'IDENTITY' )
       CALL SMT_put( data%S%type, 'IDENTITY', data%nrek_inform%alloc_status )
       data%S%n = n ; data%S%ne = 0

     CASE DEFAULT
       data%nrek_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     data%use_s = .TRUE.
     status = GALAHAD_ok
     RETURN

!  error returns

 900 CONTINUE
     status = data%nrek_inform%status
     RETURN

!  End of subroutine NREK_S_import

     END SUBROUTINE NREK_S_import

!-  G A L A H A D -  N R E K _ r e s e t _ c o n t r o l   S U B R O U T I N E -

     SUBROUTINE NREK_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See NREK_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NREK_control_type ), INTENT( IN ) :: control
     TYPE ( NREK_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%nrek_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine NREK_reset_control

     END SUBROUTINE NREK_reset_control

!-  G A L A H A D -  N R E K _ s o l v e _ p r o b l e m  S U B R O U T I N E  -

     SUBROUTINE NREK_solve_problem( data, status, H_val, C, power, weight,     &
                                    X, S_val )

!  solve the norm-regularization problem whose structure was previously
!  imported. See NREK_solve for a description of the required arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type NREK_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction power > 2, weight > 0, or requirement that the types 
!       contain a relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' or 'IDENTITY' has been violated.
!
!  H_val is a one-dimensional array of size h_ne and type default real
!   that holds the values of the entries of the lower triangular part of
!   the Hessian matrix H in the storage scheme specified in nrek_import.
!
!  C is a rank-one array of dimension n and type default
!   real, that holds the vector of linear terms of the objective, c.
!   The j-th component of C, j = 1, ... , n, contains (c)_j.
!
!  power is a scalar of type default real, that holds the value of the 
!  regularization power larger than two.
!
!  weight is a scalar of type default real, that holds the positive value
!   of the regularization weight.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  S_val is an optional one-dimensional array of size S_ne and type default
!   real that holds the values of the entries of the lower triangular part of
!   the scaling matrix S in the storage scheme specified in nrek_import. This
!   need not be given if S is the identity matrix
!
!  resolve is an optional scalar of type default logical that should be
!   .true. if the previous problem is to be resolved with a smaller weight
!
!  new_values is an optional scalar of type default logical that should be
!   .true. if a problem with identical structure to the last, but with
!   new values is to be solved

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( NREK_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), INTENT( IN ) :: power, weight
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: C
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: H_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: S_val

!  local variables

     INTEGER ( KIND = ip_ ) :: n

!  recover the dimension

     n = data%H%n

!  save the Hessian entries

     IF ( data%H%ne > 0 ) data%H%val( : data%H%ne ) = H_val( : data%H%ne )

!  call the solver

     IF ( .NOT. data%use_s ) THEN
       CALL NREK_solve( n, data%H, C, power, weight, X, data%nrek_data,        &
                        data%nrek_control, data%nrek_inform )
     ELSE
       IF ( .NOT. PRESENT( S_val ) ) THEN
         data%nrek_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       IF ( data%S%ne > 0 ) data%S%val( : data%S%ne ) = S_val( : data%S%ne )
       CALL NREK_solve( n, data%H, C, power, weight, X, data%nrek_data,        &
                        data%nrek_control, data%nrek_inform, S = data%S )
     END IF

     status = data%nrek_inform%status
     RETURN

!  error returns

 900 CONTINUE
     status = data%nrek_inform%status
     RETURN

!  End of subroutine NREK_solve_problem

     END SUBROUTINE NREK_solve_problem

!-  G A L A H A D -  N R E K _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE NREK_information( data, inform, status )

!  return solver information during or after solution by NREK
!  See NREK_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NREK_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NREK_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%nrek_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine NREK_information

     END SUBROUTINE NREK_information

!  end of module NREK_precision

   END MODULE GALAHAD_NREK_precision

