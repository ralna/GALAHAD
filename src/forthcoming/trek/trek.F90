! THIS VERSION: GALAHAD 5.3 - 2025-07-01 AT 08:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ T R E K    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Fowkes/Gould/Montoison/Orban, for GALAHAD productions
!  Principal author: Hussam Al Daas and Nick Gould

!  History -
!   originally released in GALAHAD Version 5.2. February 15th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_TREK_precision

     USE GALAHAD_KINDS_precision
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPACE_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SMT_precision
     USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
     USE GALAHAD_MOP_precision, ONLY: TREK_Hv => MOP_Ax
     USE GALAHAD_BLAS_inter_precision, ONLY: TBSV
     USE GALAHAD_LAPACK_inter_precision, ONLY: LAENV, GEQRF, ORGQR, SYEV,      &
                                               PBTRF, PBTRS
     USE GALAHAD_SLS_precision, ONLY: SLS_control_type, SLS_inform_type,       &
                                      SLS_data_type, SLS_initialize,           &
                                      SLS_read_specfile, SLS_analyse,          &
                                      SLS_factorize, SLS_solve, SLS_terminate
     USE GALAHAD_TRS_precision, ONLY: TRS_control_type, TRS_inform_type,       &
                                      TRS_read_specfile, TRS_solve_diagonal

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: TREK_initialize, TREK_read_specfile, TREK_solve,                &
               TREK_terminate, TREK_full_initialize, TREK_full_terminate,      &
               SMT_type, SMT_put, SMT_get

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = rp_ ), PARAMETER :: violation_max = 0.0001_rp_
     REAL ( KIND = rp_ ), PARAMETER :: h_pert = SQRT( epsmch )

     INTEGER ( KIND = ip_ ) :: ldp = 3 ! P is symmetric, pentadiagonal band

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: TREK_control_type

!   error and warning diagnostics occur on stream error

       INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

       INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  maximum iteration count

       INTEGER ( KIND = ip_ ) :: maxit = - 1

!  reduction factor for subsequent trust-region radii, i = 2,...,m

       REAL ( KIND = rp_ ) :: reduction = 0.5_rp_

!  stopping tolerance for the cheaply-computed residual

!      ||A X + X D + ce'|| < tol ||c e'||

       REAL ( KIND = rp_ ) :: stop_residual = ten * SQRT( epsmch )

!  should the incoming Lanczos vectors be re-orthogonalised against the
!  existing ones (this can be expensive)

       LOGICAL :: reorthogonalize = .FALSE.

!  should the exact shifts for each radius be computed at every iteration
!  (this can be expensive)

!      LOGICAL :: exact_shift = .FALSE.
       LOGICAL :: exact_shift = .TRUE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

       LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  positive-definite linear equation solver

       CHARACTER ( LEN = 30 ) :: solver = "ssids" // REPEAT( ' ', 25 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""' // REPEAT( ' ', 28 )

!  control parameters for SLS

       TYPE ( SLS_control_type ) :: sls_control

!  control parameters for TRS

       TYPE ( TRS_control_type ) :: trs_control

     END TYPE TREK_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: TREK_time_type

!  total CPU time spent in the package

        REAL ( KIND = rp_ ) :: total = 0.0

!  CPU time spent building H + lambda * I

        REAL ( KIND = rp_ ) :: assemble = 0.0

!  CPU time spent reordering H + lambda * I prior to factorization

        REAL ( KIND = rp_ ) :: analyse = 0.0

!  CPU time spent factorizing H + lambda * I

        REAL ( KIND = rp_ ) :: factorize = 0.0

!  CPU time spent solving linear systems inolving H + lambda * M

        REAL ( KIND = rp_ ) :: solve = 0.0

!  total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0

!  clock time spent building H + lambda * I

        REAL ( KIND = rp_ ) :: clock_assemble = 0.0

!  clock time spent reordering H + lambda * I prior to factorization

        REAL ( KIND = rp_ ) :: clock_analyse = 0.0

!  clock time spent factorizing H + lambda * I

        REAL ( KIND = rp_ ) :: clock_factorize = 0.0

!  clock time spent solving linear systems inolving H + lambda * M

        REAL ( KIND = rp_ ) :: clock_solve = 0.0
      END TYPE TREK_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: TREK_inform_type

!  return status. See TREK_solve for details

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

!  the M-norm of x, ||x||_M

        REAL ( KIND = rp_ ) :: x_norm = zero

!  the Lagrange multiplier corresponding to the trust-region constraint

        REAL ( KIND = rp_ ) :: multiplier = zero

!  the current radius

        REAL ( KIND = rp_ ) :: radius = - one

!  the proposed next radius to be used

        REAL ( KIND = rp_ ) :: next_radius = - one

!  the maximum relative residual error

        REAL ( KIND = rp_ ) :: error = - one

!  time information

        TYPE ( TREK_time_type ) :: time

!  inform parameters for SLS

        TYPE ( SLS_inform_type ) :: sls_inform

!  inform parameters for TRS

        TYPE ( TRS_inform_type ) :: trs_inform

      END TYPE TREK_inform_type

!  ...................
!   data derived type
!  ...................

      TYPE, PUBLIC :: TREK_data_type

        INTEGER ( KIND = ip_ ) :: is_max, k_max, lwork_syev, n_v
        INTEGER ( KIND = ip_ ) :: k_exit = - 1
        REAL ( KIND = rp_ ) :: c_norm, last_radius, last_shift, shift_val
        LOGICAL :: shifted, sparse
        LOGICAL :: allocated_arrays = .FALSE.

!  common workspace arrays

        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: MAP
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: V, P, P_shift
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: Q, S, S2
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: ALPHA, BETA, DELTA
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: U, U1, S1, W, SHIFT
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C, X, D, WORK_syev
        TYPE ( SMT_type ) :: H

!  local copy of control

        TYPE ( TREK_control_type ) :: control

!  data for SLS

        TYPE ( SLS_data_type ) :: sls_data
      END TYPE TREK_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: TREK_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( TREK_data_type ) :: TREK_data
        TYPE ( TREK_control_type ) :: TREK_control
        TYPE ( TREK_inform_type ) :: TREK_inform
      END TYPE TREK_full_data_type

   CONTAINS

!-*-*-*-*-*-   T R E K _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE TREK_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for TREK. This routine should be called before
!  TREK_solve
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

      TYPE ( TREK_data_type ), INTENT( INOUT ) :: data
      TYPE ( TREK_control_type ), INTENT( OUT ) :: control
      TYPE ( TREK_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Initalize SLS components

      CALL SLS_initialize( control%solver, data%SLS_data,                     &
                           control%SLS_control, inform%SLS_inform )
!     control%SLS_control%perturb_to_make_definite = .FALSE.
!     control%SLS_control%preconditioner = 2
      control%SLS_control%prefix = '" - SLS:"                    '

      RETURN

!  End of TREK_initialize

      END SUBROUTINE TREK_initialize

!- G A L A H A D -  T R E K _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE TREK_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for TREK controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( TREK_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( TREK_control_type ), INTENT( OUT ) :: control
     TYPE ( TREK_inform_type ), INTENT( OUT ) :: inform

     CALL TREK_initialize( data%trek_data, control, inform )

     RETURN

!  End of subroutine TREK_full_initialize

     END SUBROUTINE TREK_full_initialize

!-*-*-*-*-   T R E K _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE TREK_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by TREK_initialize could (roughly)
!  have been set as:

! BEGIN TREK SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-number-of-iterations                      -1
!  start-error-search                                0
!  radius-reduction-factor                           0.5E+0
!  residual-accuracy                                 1.0E-8
!  reorthogonalize-vectors                           F
!  exact-shift                                       F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  symmetric-linear-equation-solver                  ssids
!  output-line-prefix                                ""
! END TREK SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( TREK_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxit = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: reduction = maxit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_residual = reduction + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: reorthogonalize = stop_residual + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: exact_shift = reorthogonalize + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = exact_shift + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: solver = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = solver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'TREK'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( maxit )%keyword = 'maximum-number-of-iterations'

!  Real key-words

      spec( reduction )%keyword = 'radius-reduction-factor'
      spec( stop_residual )%keyword = 'residual-accuracy'

!  Logical key-words

      spec( reorthogonalize )%keyword = 'reorthogonalize-vectors'
      spec( exact_shift )%keyword = 'exact-shift'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( solver )%keyword = 'symmetric-linear-equation-solver'
      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' trek: ', alt_specname

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
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( reduction ),                            &
                                 control%reduction,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_residual ),                        &
                                 control%stop_residual,                        &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( reorthogonalize ),                      &
                                 control%reorthogonalize,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( exact_shift ),                          &
                                 control%exact_shift,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
!  Set character values

     CALL SPECFILE_assign_value( spec( solver ),                               &
                                 control%solver,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Read the specfiles for SLS and TRS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS')
        CALL TRS_read_specfile( control%TRS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-TRS')
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
        CALL TRS_read_specfile( control%TRS_control, device )
      END IF

      RETURN

      END SUBROUTINE TREK_read_specfile

!-*-*-*-*-*-*-*-*-   T R E K _ S O L V E    S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE TREK_solve( n, H, C, radius, X, data, control, inform,         &
                            resolve, new_values )

!  Given an n x n symmetric matrix H, an n-vector c, and a radius > 0,
!  approximately solve the trust-region subproblem
!
!    min 1/2 x'Hx + c'x : ||x|| <= radius
!
!  using an extended Krylov subspace method.
!
!  The method uses the "backward" extended-Krylov subspace
!
!    K_2k = { c, H^{-1} c, H c, ..., H^{-k} c },
!
!  (see module EKS)
!
!  Input:
!   n - number of unknowns
!   H - symmetric coefficient matrix, H, from the quadratic term, 
!       in any symmetric format supported by the SMT type
!   C - vector c from linear term
!   radius - scalar trust-region radius > 0
!   control - parameters structure (see preamble)
!   inform - output structure (see preamble)
!   data - prvate internal workspace
!   resolve - (optional) resolve a previously solved problem with a smaller
!             radius
!   new_values - (optional) solve a problem with the same structure as the
!           previuos one but with different values of H and/or c
!
!   Output:
!   X - solution vector x

!  dummy arguments

     INTEGER ( KIND = ip_ ),  INTENT( IN ) :: n
!    TYPE ( SMT_type ), INTENT( IN ) :: H
     TYPE ( SMT_type ), INTENT( INOUT ) :: H
     REAL ( KIND = rp_), INTENT( IN ), DIMENSION( n ) :: C
     REAL ( KIND = rp_), INTENT( IN ) :: radius
     REAL ( KIND = rp_), INTENT( OUT ), DIMENSION( n  ) :: X
     TYPE ( TREK_data_type ), INTENT( INOUT ) :: data
     TYPE ( TREK_control_type ), INTENT( IN ) :: control
     TYPE ( TREK_inform_type ), INTENT( INOUT ) :: inform
     LOGICAL, OPTIONAL, INTENT( IN ) :: resolve, new_values

!  local variables

     INTEGER ( KIND = IP_ ) :: i, j, jj, j_max, k, km1, k2, k2m1, k2p1, mkp1, l
     INTEGER ( KIND = IP_ ) :: k_start, lapack_info, nb, out, h_ne, nz
     REAL ( KIND = rp_) :: alpha, beta, gamma, delta, newton_norm
     REAL ( KIND = rp_) :: e11, e12, e22, error, error_j, s_norm, w_norm
     REAL ( KIND = rp_) :: violation, shift
     REAL :: time_start, time_now, time_record
     REAL ( KIND = rp_ ) :: clock_start, clock_now, clock_record
     LOGICAL :: printi, printm, printd, printh
     LOGICAL :: initial, restart, new_h, shifted_structure
     LOGICAL :: termination_test = .TRUE.
     CHARACTER ( LEN = 80 ) :: array_name

!  temporary debug variables - ultimately remove

     INTEGER ( KIND = IP_ ) :: ii
     INTEGER ( KIND = IP_ ),  PARAMETER :: p_dim = 10
     REAL ( KIND = rp_) :: P_calc( p_dim, p_dim )
     REAL ( KIND = rp_) :: R( n ), SOL( n )

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  see if this is an initial- or a re-entry

     IF ( PRESENT( resolve ) ) THEN
       initial = .NOT. resolve
     ELSE
       initial = .TRUE.
     END IF

!  check solve has been called before resolve

     IF ( .NOT. initial ) THEN
       IF ( data%k_exit <= 0 ) THEN
         inform%status = GALAHAD_error_call_order ; GO TO 920
       END IF
     END IF

!  see if the H-c data is new

     IF ( PRESENT( new_values ) ) THEN
       restart = new_values .AND. data%k_exit > 0
     ELSE
       restart = .FALSE.
     END IF

     IF ( restart ) THEN
       initial = .FALSE.
       data%k_exit = - 1
     END IF

     new_h = initial .OR. restart

!  check input dimensions are consistent

     IF ( n /= H%n ) THEN
       inform%status = GALAHAD_error_restrictions ; GO TO 920
     END IF

!  check initial radius value is positive

     IF ( new_h ) THEN
       IF ( radius <= zero ) THEN
         inform%status = GALAHAD_error_restrictions ; GO TO 920
       END IF

!  on reentry, also check the new radius value is smaller than the previous one

     ELSE
       IF ( radius <= zero .OR. radius >= data%last_radius ) THEN
         inform%status = GALAHAD_error_restrictions ; GO TO 920
       END IF

!  record the radiius, and set the potential next one

       inform%next_radius = radius * data%control%reduction
     END IF
     inform%radius = radius
     data%last_radius = radius

!  record output values

     out = control%out
     printi = control%print_level > 0 .AND. out > 0
     printm = control%print_level > 2 .AND. out > 0
     printd = control%print_level > 5 .AND. out > 0

!  initial entry - allocate data structures
!  ----------------------------------------

     IF ( initial ) THEN

!  record the iteration limit

       data%control = control
       IF ( data%control%trs_control%max_factorizations < 0 )                  &
         data%control%trs_control%max_factorizations = 10
!      data%k_max = MAX( control%maxit, 1 )
       IF ( control%maxit > 0 ) THEN
         data%k_max = control%maxit
       ELSE
         data%k_max = 100
       END IF
       data%is_max = 2 * data%k_max
       data%k_exit = - 1

!   provide space for, and initiate, the components of the projected solution

       IF ( .NOT. data%allocated_arrays ) THEN
         data%allocated_arrays = .TRUE.

         array_name = 'trek: data%V'
         CALL SPACE_resize_array(  1_ip_, n, - data%k_max, data%k_max, data%V, &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  set the projected matrix P and solution s

         array_name = 'trek: data%P'
         CALL SPACE_resize_array( ldp, data%is_max, data%P,                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%P_shift'
         CALL SPACE_resize_array( ldp, data%is_max, data%P_shift,              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%S1'
         CALL SPACE_resize_array( data%is_max, data%S1,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%S2'
         CALL SPACE_resize_array( data%is_max, 1, data%S2,                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%W'
         CALL SPACE_resize_array( data%is_max, data%W,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  set workspace arrays

         array_name = 'trek: data%ALPHA'
         CALL SPACE_resize_array( - data%k_max, data%k_max, data%ALPHA,        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%BETA'
         CALL SPACE_resize_array( - data%k_max, data%k_max, data%BETA,         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%DELTA'
         CALL SPACE_resize_array( - data%k_max, data%k_max, data%DELTA,        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         array_name = 'trek: data%U'
         CALL SPACE_resize_array( n, data%U,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         IF ( control%exact_shift ) THEN
           array_name = 'trek: data%Q'
           CALL SPACE_resize_array( data%is_max, data%is_max, data%Q,          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910

           array_name = 'trek: data%D'
           CALL SPACE_resize_array( data%is_max, data%D,                       &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910

           array_name = 'trek: data%C'
           CALL SPACE_resize_array( data%is_max, data%C,                       &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910

           array_name = 'trek: data%X'
           CALL SPACE_resize_array( data%is_max, data%X,                       &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910

!  discover the size of, and allocate, workspace needed for the eigensolver

           nb = LAENV( 1_ip_, 'DSYTRD', 'L', data%is_max,                      &
                      - 1_ip_, - 1_ip_, - 1_ip_ )
           data%lwork_syev                                                     &
             = MAX( 1_ip_, 3 * data%is_max - 1, ( nb + 2 ) * data%is_max )

           ALLOCATE( data%WORK_syev( data%lwork_syev ),                        &
                     STAT = inform%alloc_status )
           IF ( inform%alloc_status /= 0 ) THEN
             inform%bad_alloc = 'trek: data%WORK_syev'
             inform%status = GALAHAD_error_allocate ; GO TO 910
           END IF
         END IF
       END IF

       IF ( printd ) THEN
         array_name = 'trek: data%U1'
         CALL SPACE_resize_array( n, data%U1,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910
       END IF

!   initialize and analyse H

       CALL SLS_initialize( control%solver, data%sls_data,                     &
                            data%control%sls_control,                          &
                            inform%sls_inform, check = .TRUE. )
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
         inform%status = GALAHAD_error_analysis ; GO TO 920

!  skip the factorization if the analysis finds that H is structurally
!  indefinite

       ELSE IF ( inform%sls_inform%status == GALAHAD_error_inertia ) THEN
         shifted_structure = .TRUE.
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
              inform%sls_inform%negative_eigenvalues > 0 ) THEN
           shifted_structure = .TRUE.

!  the factorization failed

         ELSE IF ( inform%sls_inform%status < 0 ) THEN
           inform%status = GALAHAD_error_factorization ; GO TO 920

!  H is definite

         ELSE
           shifted_structure = .FALSE.
           data%shifted = .FALSE.
         END IF
       END IF
     END IF

!  factorize 

     IF ( new_h ) THEN
       IF ( restart ) THEN
         IF ( .NOT. data%shifted ) THEN

!  factorize H, again checking that it is definite

           CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
           CALL SLS_factorize( H, data%sls_data, data%control%sls_control,     &
                               inform%sls_inform )
           CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
           inform%time%factorize =                                             &
             inform%time%factorize + time_now - time_record
           inform%time%clock_factorize =                                       &
             inform%time%clock_factorize + clock_now - clock_record

!  H is indefinite

           IF ( inform%sls_inform%status == GALAHAD_error_inertia .OR.         &
                inform%sls_inform%negative_eigenvalues > 0 ) THEN
             shifted_structure = .TRUE.

!  the factorization failed

           ELSE IF ( inform%sls_inform%status < 0 ) THEN
             inform%status = GALAHAD_error_factorization ; GO TO 920

!  H is definite

           ELSE
             shifted_structure = .FALSE.
           END IF
         END IF
       END IF
!write(6,*) ' shifted structure = ', shifted_structure

!  H is indefinite. Shift the diagonals so that the result is definite,
!  using a separate H data structure, data%H

       IF ( shifted_structure ) THEN
         data%shifted = .TRUE.
         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )

!  build the data structure for the shifted H. Start by counting the number of
!  nonzeros in the whole (i.e., with all diagonals)  of the lower triangle of H
     
         SELECT CASE ( SMT_get( H%type ) )
         CASE ( 'COORDINATE' )
           nz = H%n ; h_ne = H%ne ; data%sparse = .TRUE.
           DO l = 1, H%ne
             IF ( H%row( l ) /= H%col( l ) ) nz = nz + 1
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           nz = H%n ; h_ne = H%ptr( H%n + 1 ) - 1 ; data%sparse = .TRUE.
           DO i = 1, H%n
             DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
               IF ( H%col( l ) /= i ) nz = nz + 1
             END DO
           END DO
         CASE ( 'DENSE' )
           nz = H%ne * ( H%ne + 1 ) / 2 ; data%sparse = .FALSE.
         END SELECT
!write(6,*) ' nz ', nz

!  allocate space for the whole of the lower triangle

         array_name = 'trek: data%H%val'
         CALL SPACE_resize_array( nz, data%H%val,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

         IF ( data%sparse ) THEN
           array_name = 'trek: data%H%col'
           CALL SPACE_resize_array( nz, data%H%col,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910

           array_name = 'trek: data%H%ptr'
           CALL SPACE_resize_array( H%n + 1, data%H%ptr,                       &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910

           array_name = 'trek: data%MAP'
           CALL SPACE_resize_array( h_ne, data%MAP,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910
         END IF

!  store the whole of the lower triangle by rows; the last entry in each row
!  is the diagonal. Firstly, count the number of nonzeros in each row - the
!  number in row i is stored in data%h%ptr(i+1). Also temporarily use 
!  data%h%col(i) to hold the poistion in H of the ith diagonal. This
!  is not needed when H is dense, as then H is already in the right order

         data%H%n = H%n
         IF ( data%sparse ) THEN
           CALL SMT_put( data%H%type, 'SPARSE_BY_ROWS', inform%alloc_status )
           data%H%ptr( 2 : H%n + 1 ) = 1
           data%H%col( : H%n ) = 0

           SELECT CASE ( SMT_get( H%type ) )
           CASE ( 'COORDINATE' )
             DO l = 1, H%ne
               i = H%row( l ) ; j = H%col( l )
               IF ( i > j ) THEN
                 data%H%ptr( i + 1 ) = data%H%ptr( i + 1 ) + 1
               ELSE IF ( i < j ) THEN
                 data%H%ptr( j + 1 ) = data%H%ptr( j + 1 ) + 1
               ELSE
                 data%H%col( i ) = l
               END IF
             END DO
!write(6,*) ' lengths ', data%H%ptr( 2 : H%n + 1 )
           CASE ( 'SPARSE_BY_ROWS' )
             DO i = 1, H%n
               DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                 j = H%col( l )
                 IF ( i > j ) THEN
                   data%H%ptr( i + 1 ) = data%H%ptr( i + 1 ) + 1
                 ELSE IF ( i < j ) THEN
                   data%H%ptr( j + 1 ) = data%H%ptr( j + 1 ) + 1
                 ELSE
                   data%H%col( i ) = l
                 END IF
               END DO
             END DO
           END SELECT

!  now store the starting address for each row

           data%H%ptr( 1 ) = 1
           DO i = 1, H%n
             data%H%ptr( i + 1 ) = data%H%ptr( i ) + data%H%ptr( i + 1 )

!  map the ith diagonal in the original matrix to its new poition

             l = data%H%col( i )
             IF ( l /= 0 ) data%map( l ) = data%H%ptr( i + 1 ) - 1
           END DO

!  store the location of each diagonal

!           DO i = 1, H%n
!             write(6,*) ' start ', i, data%H%ptr( i )
!           END DO

           DO i = 1, H%n
!             write(6,*) ' diag ', i, data%H%ptr( i + 1 ) - 1
             data%H%col( data%H%ptr( i + 1 ) - 1 ) = i
           END DO

!  map the remaining entries, and set up the coilumn indices

           SELECT CASE ( SMT_get( H%type ) )
           CASE ( 'COORDINATE' )
             DO l = 1, H%ne
               i = H%row( l ) ; j = H%col( l )
               IF ( i > j ) THEN
                 k = data%H%ptr( i )
                 data%H%col( k ) = j
                 data%H%ptr( i ) = k + 1
                 data%map( l ) = k
               ELSE IF ( i < j ) THEN
                 k = data%H%ptr( j )
!write(6,*) ' i, j, k', i, j, k
                 data%H%col( k ) = i
                 data%H%ptr( j ) = k + 1
                 data%map( l ) = k
               END IF
             END DO
           CASE ( 'SPARSE_BY_ROWS' )
             DO i = 1, H%n
               DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                 j = H%col( l )
                 IF ( i > j ) THEN
                   k = data%H%ptr( i )
                   data%H%col( k ) = j
                   data%H%ptr( i ) = k + 1
                   data%map( l ) = k
                 ELSE IF ( i < j ) THEN
                   k = data%H%ptr( j )
                   data%H%col( k ) = i
                   data%H%ptr( j ) = k + 1
                   data%map( l ) = k
                 END IF
               END DO
             END DO
           END SELECT

!  restore the starting addresses for each row

           DO i = H%n, 1, - 1
              data%H%ptr( i + 1 ) = data%H%ptr( i ) + 1
           END DO
           data%H%ptr( 1 ) = 1

!           DO i = 1, H%n
!             write(6,*) ' start ', i, data%H%ptr( i )
!           END DO

         ELSE
           CALL SMT_put( data%H%type, 'DENSE', inform%alloc_status )
         END IF
       END IF

!  now copy the triangular matrix to the new whole triangular one,
!  makimg sure that absent diagonals are zeroed

       IF ( data%shifted ) THEN
         IF ( data%sparse ) THEN
           DO i = 1, H%n
             data%H%val( data%H%ptr( i + 1 ) - 1 ) = zero
           END DO
           DO l = 1, h_ne
             data%H%val( data%map( l ) ) = H%val( l )
           END DO 
         ELSE
           data%H%val( : h_ne ) = H%val( : h_ne )
         END IF

!  compute the Gershgorin shift, shift_val

         CALL TREK_find_shift( H, data%shift_val, data%U, i )
         IF ( printi ) WRITE( out, "( A, ' perturbing H ...',                  &
        &    ' Gershgorin shift =', ES11.4 )" ) prefix, data%shift_val

!  finally add the Gershgorin shift

         IF ( data%sparse ) THEN
           DO i = 1, H%n
             l = data%H%ptr( i + 1 ) - 1
             data%H%val( l ) = data%H%val( l ) + data%shift_val
           END DO

!do i = 1, n
! write(6,*) data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
! do l = data%H%ptr( i ), data%H%ptr( i + 1 ) - 1
!  write(6,*) i, data%H%col( l ), data%H%val( l )
! end do
!end do
!write(6,*) ' type ', data%H%type
!write(6,*) ' ptr ', data%H%ptr
!write(6,*) ' col ', data%H%col
!write(6,*) ' val ', data%H%val
         ELSE
           DO i = 1, H%n
             l = i * ( i + 1 ) / 2
             data%H%val( l ) = data%H%val( l ) + data%shift_val
           END DO
         END IF
         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%assemble = inform%time%assemble + time_now - time_record
         inform%time%clock_assemble =                                          &
           inform%time%clock_assemble + clock_now - clock_record

!  Deallocate all arrays allocated within SLS

         CALL SLS_terminate( data%SLS_data, control%SLS_control,               &
                             inform%SLS_inform )

         CALL SLS_initialize( control%solver, data%sls_data,                   &
                              data%control%sls_control,                        &
                              inform%sls_inform, check = .TRUE. )
!        data%control%sls_control%print_level = 3
         data%control%sls_control%pivot_control = 2

!  re-analyse the shifted H

         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
         CALL SLS_analyse( data%H, data%sls_data, data%control%sls_control,    &
                           inform%sls_inform )
         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%analyse = inform%time%analyse + time_now - time_record
         inform%time%clock_analyse =                                           &
           inform%time%clock_analyse + clock_now - clock_record
         IF ( inform%sls_inform%status < 0 ) THEN
           inform%status = GALAHAD_error_analysis ; GO TO 920
         END IF

!  factorize the shifted H, while checking again that it is definite

         CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
         CALL SLS_factorize( data%H, data%sls_data, data%control%sls_control,  &
                             inform%sls_inform )
         CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%factorize = inform%time%factorize + time_now - time_record
         inform%time%clock_factorize =                                         &
           inform%time%clock_factorize + clock_now - clock_record
         IF ( inform%sls_inform%status < 0 ) THEN
           WRITE( 6, "( ' H still indefinite after perturbation' )" )
           inform%status = GALAHAD_error_factorization ; GO TO 920
         END IF
       END IF

!  re-entry
!  --------

     ELSE

!  initialise the shift as that from the previous solve

       shift = data%last_shift

!  use the data from the previous solve to see if that subspace is sufficiently
!  rich to contain the solution with the new (reduced) radius

       k = data%k_exit
       inform%iter = k
       k2 = 2 * k
       k2m1 = k2 - 1
       k2p1 = k2 + 1

!  ....................-------------------
!  test system of order 2k for termination
!  ....................-------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

       IF (  termination_test ) THEN   
         j_max = 1

!  exact shifts are required

         IF ( control%exact_shift ) THEN

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
                WRITE( out, "( 4( 2I4, ES12.4 ) )" )                           &
                  ( i, j, data%Q( i, j ), j = 1, i ) 
              END DO
            END IF

!  compute the eigenvalues (in D) and eigenvalues (overwrite Q)

           CALL SYEV( 'V','L', k2, data%Q, data%is_max, data%D,                &
                     data%WORK_syev, data%lwork_syev, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  form c' = Q^T ||c|| e_1

           data%C( 1 : k2 ) = data%c_norm * data%Q( 1, 1 : k2 )

!  solve the diagonal trust-region problem
!    min 1/2 x'^T D x' + c'^T x' : ||x'|| <= radius

           data%control%trs_control%use_initial_multiplier = .TRUE.
           data%control%trs_control%initial_multiplier = shift

           IF ( data%shifted ) THEN
             data%control%trs_control%initial_multiplier                       &
               = shift - data%shift_val
             CALL TRS_solve_diagonal( k2, radius, zero,                        &
                                      data%C, data%D, data%X,                  &
                                      data%control%trs_control,                &
                                      inform%trs_inform,                       &
                                      shift = data%shift_val )
             shift = inform%trs_inform%multiplier + data%shift_val
           ELSE
             CALL TRS_solve_diagonal( k2, radius, zero,                        &
                                      data%C, data%D, data%X,                  &
                                      data%control%trs_control,                &
                                      inform%trs_inform )
             shift = inform%trs_inform%multiplier
           END IF
           inform%multiplier = shift
           inform%obj = inform%trs_inform%obj
           inform%x_norm = inform%trs_inform%x_norm

!  recover x = Q x'

           data%S1( 1 : k2 ) = MATMUL( data%Q( : k2 , : k2 ), data%X( : k2 ) )

!  exact shifts are not required

         ELSE

!  form and factorize the shifted matrix P + shift(j) I

           data%P_shift( 1, : k2 ) = data%P( 1, : k2 ) + shift
           data%P_shift( 2 : 3, : k2 ) = data%P( 2 : 3, : k2 )
           CALL PBTRF( 'L', k2, 2, data%P_shift, ldp, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  solve the shifted system to find the solution s(j)

           data%S2( 1, 1 ) = data%c_norm ; data%S2( 2 : k2, 1 ) = zero
           CALL PBTRS( 'L', k2, 2, 1, data%P_shift, ldp,                       &
                       data%S2, data%is_max, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF
           data%S1( 1 : k2 ) = data%S2( 1 : k2, 1 )
         END IF

!  record ||r_2k(t)||^2 = ( alpha_k s_2k-1(j))^2 + ( alpha_-k s_2k(j))^2

         delta = data%DELTA( k )
         gamma = data%P( 3, k2m1 )
         error = ABS( gamma * data%S1( k2m1 ) + delta * data%S1( k2 ) )

!  compute ||s|| and the violation

         s_norm = TWO_NORM( data%S1( 1 : k2 ) )
         violation = s_norm - radius
         IF ( printm ) WRITE( out,                                             &
           "(' k = ', I0, ' ||s||, radius, multiplier = ', 3ES11.4 )" )        &
              k, s_norm, radius, shift

!  if ||s|| > radius, update shift. First, solve L w = s ...

         IF ( .NOT. control%exact_shift ) THEN
           inform%obj                                                          &
             = half * ( data%c_norm * data%S1( 1 ) - shift * s_norm ** 2 )
!          IF ( s_norm > radius ) THEN
           IF ( .TRUE. ) THEN
             data%W( : k2 ) = data%S1( : k2 )
             CALL TBSV( 'L', 'N', 'N', k2, 2, data%P_shift, ldp,               &
                        data%W, 1_ip_)

!  ... then use the norm of w to improve the shift

             w_norm = TWO_NORM( data%W( 1 : k2 ) )
             shift = shift + ( s_norm - radius ) * s_norm ** 2                 &
                       / ( RADIUS * w_norm ** 2 )
           END IF
         END IF

!  record ||s|| and the current shift

         inform%multiplier = shift
         inform%x_norm = s_norm

!  debug - compare predicted and actual error

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
           R = shift * SOL + C
           IF ( data%shifted ) THEN
             CALL TREK_Hv( one, data%H, SOL, one, R, out, control%error,       &
                           symmetric = .TRUE. )
           ELSE
             CALL TREK_Hv( one, H, SOL, one, R, out, control%error,            &
                           symmetric = .TRUE. )
           END IF
           WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" ) TWO_NORM( R ), error
         END IF

         IF ( printi ) THEN
           WRITE( out, 2000 ) prefix
           WRITE( out, 2010 ) prefix, k, k2,                                   &
             s_norm, radius, shift, error, inform%obj
         END IF
         error = error / data%c_norm

!  check for termination

         IF ( error < data%control%stop_residual .AND.                         &
              violation <= violation_max ) THEN
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
           data%last_shift = shift
           inform%error = error
           inform%status = GALAHAD_ok
           GO TO 900
         END IF
       END IF

!  set delta_{-k-1} := ||u|| 

       delta = TWO_NORM( data%U )

!  set v_{-k-1} = u/delta_{-k-1}

       IF ( k < data%k_max ) THEN
         data%DELTA( - k - 1 ) = delta
         data%V( 1 : n, - k - 1 ) = data%U / delta
         data%n_v = data%n_v + 1
         IF ( data%n_v == n ) WRITE( out, 2020 ) prefix
       END IF

     END IF

!  ****************************************************************************
!
!  build and use the backward extended Krylov space
!
!    K_2k = { c, H^{-1} c, H c, ..., H^{-k} c }
!
!  Do this by building an orthogonal basis matrix
!
!    V_2k = [ v_0, v_{-1}, v_1, ... , v_{-k} ]
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
!    Extended-Krylov-subspace trust-region methods
!    Working note STFC-RAL 2025
!
!  ****************************************************************************

     IF ( new_h ) THEN

!  record the initial error

       data%c_norm = TWO_NORM( C( 1 : n ) )

!  normalize the input vector, delta_0 = ||c|| and v_0 = -c/delta_0

       data%DELTA( 0 ) = data%c_norm
       data%V( 1 : n, 0 ) = - C( 1 : n ) / data%c_norm

!  set u = H^{−1} v_0

       data%U = data%V( 1 : n, 0 )
       CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
       IF ( data%shifted ) THEN
         CALL SLS_solve( data%H, data%U, data%sls_data,                        &
                         data%control%sls_control, inform%sls_inform )
       ELSE
         CALL SLS_solve( H, data%U, data%sls_data,                             &
                         data%control%sls_control, inform%sls_inform )
       END IF
       CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
       inform%time%solve = inform%time%solve + time_now - time_record
       inform%time%clock_solve =                                         &
         inform%time%clock_solve + clock_now - clock_record
       IF ( inform%sls_inform%status < 0 ) THEN
         inform%status = GALAHAD_error_solve ; GO TO 920
       END IF

!  compute ||x_newton|| = ||H^{-1} c||

       newton_norm = data%c_norm * TWO_NORM( data%U ) 

!  if the Newton step is inside the trust region, record the step

       IF ( .NOT. data%shifted .AND. newton_norm <= radius ) THEN
         X( 1 : n ) = data%c_norm * data%U( 1 : n )

!  compute the next potential radius depending on whether ||x_newton|| <= radius

         inform%next_radius = radius
         DO
           inform%next_radius = inform%next_radius * data%control%reduction
           IF ( newton_norm > inform%next_radius ) EXIT
         END DO
         data%k_exit = 1
         data%last_shift = zero
         inform%obj = half * DOT_PRODUCT( X, C )
         inform%multiplier = zero
       ELSE
         inform%next_radius = radius * data%control%reduction
       END IF

!  set beta_0 = v_0' u and u = u - beta_0 v_0
 
       beta = DOT_PRODUCT( data%V( 1 : n, 0 ), data%U )
       data%BETA( 0 ) = beta
       data%U = data%U - beta * data%V( 1 : n, 0 )

!  set delta_{-1} = ||u||; v_{-1} = u / delta_{-1}

       data%DELTA( - 1 ) = TWO_NORM( data%U )
       data%V( 1 : n, - 1 ) = data%U / data%DELTA( - 1 )
       data%n_v = 2

!  if the Newton step is inside the trust region, exit
     
       IF ( .NOT. data%shifted .AND. newton_norm <= radius ) THEN
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

!  initialise the shift as zero

       shift = zero

!  start the iteration loop from the begining

       k_start = 1

!  start from the previous iteration data%k_exit, as data up to then has
!  already been generated

     ELSE
!      k_start = data%k_exit
       k_start = data%k_exit + 1
     END IF

!  ------------------------------------------------------------
!  start of main forward iteration loop (in comments, iter = k)
!  ------------------------------------------------------------

     inform%status = GALAHAD_error_max_iterations

     DO k = k_start, data%k_max
       inform%iter = k
       km1 = k - 1
       mkp1 = - k + 1
       k2 = 2 * k
       k2m1 = k2 - 1
       k2p1 = k2 + 1
       printh = printi .AND. ( k == 1 .OR.                                     &
                               control%sls_control%print_level > 0 .OR.        &
                               control%trs_control%print_level > 0 )

!  print iteration details if required

       IF ( printm ) WRITE( out, "( A, ' iteration ', I0 )" ) prefix, k

! for testing

       IF ( printd ) THEN 
         IF ( data%shifted ) THEN
           CALL TREK_Hv( one, data%H, data%V( 1 : n, mkp1 ), zero, data%U,     &
                         control%out, control%error, symmetric = .TRUE. )
           CALL TREK_Hv( one, data%H, data%V( 1 : n, k ), zero, data%U1,       &
                         control%out, control%error, symmetric = .TRUE. )
         ELSE
           CALL TREK_Hv( one, H, data%V( 1 : n, mkp1 ), zero, data%U,          &
                         control%out, control%error, symmetric = .TRUE. )
           CALL TREK_Hv( one, H, data%V( 1 : n, k ), zero, data%U1,            &
                         control%out, control%error, symmetric = .TRUE. )
         END IF
         e11 = DOT_PRODUCT( data%V( 1 : n, mkp1 ), data%U )
         e12 = DOT_PRODUCT( data%V( 1 : n, mkp1 ), data%U1 )
         e22 = DOT_PRODUCT( data%V( 1 : n, k ), data%U1 )
       END IF

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!  segment involving the product with A
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  set u = H v_{-k}

       IF ( data%shifted ) THEN
         CALL TREK_Hv( one, data%H, data%V( 1 : n, - k ), zero, data%U,        &
                       out, control%error, symmetric = .TRUE. )
       ELSE
         CALL TREK_Hv( one, H, data%V( 1 : n, - k ), zero, data%U,             &
                       out, control%error, symmetric = .TRUE. )
       END IF

!  set alpha_{-k} = v_{-k}' u and u = u _ alpha_{-k} v_{-k}

       alpha = DOT_PRODUCT( data%V( 1 : n, - k ), data%U )
       data%ALPHA( - k ) = alpha
       data%U = data%U - alpha * data%V( 1 : n, - k )

!  set alpha {k-1} = v_{k-1}' u and u = u - alpha_{k-1} v_{k-1}

       alpha = DOT_PRODUCT( data%V( 1 : n, km1 ), data%U )
       data%ALPHA( km1 ) = alpha
       data%U = data%U - alpha * data%V( 1 : n, km1 )

!  orthogonalise wrt the remaining vectors

!      IF ( .true. ) THEN
       IF ( control%reorthogonalize ) THEN
         DO i = mkp1, km1 - 1
           beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
           data%U = data%U - beta * data%V( 1 : n, i )
         END DO
       END IF

!   set delta_{k} = ||u|| and v_{k} = u/delta_{k}

       delta = TWO_NORM( data%U )
       gamma = - data%DELTA( - k ) * delta / data%BETA( km1 )
       data%DELTA( k ) = delta
       data%V( 1 : n, k ) = data%U / delta

!  save column 2k-1 of P

       IF ( k > 1 ) THEN
         data%P( 1, k2m1 ) = ( one - data%DELTA( - k ) * data%ALPHA( km1 )     &
                                   - data%BETA( mkp1 ) * data%DELTA( km1 ) )   &
                               / data%BETA( km1 )
       ELSE
         data%P( 1, k2m1 ) = ( one - data%DELTA( - k ) * data%ALPHA( km1 ) )   &
                               / data%BETA( km1 )
       END IF
       data%P( 2, k2m1 ) = data%ALPHA( km1 )
       data%P( 3, k2m1 ) = gamma

!  ....................---------------------
!  test system of order 2k-1 for termination
!  ....................---------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

!      IF ( k > 1 .AND. termination_test ) THEN   
       IF ( k > 1 .AND. .FALSE. ) THEN   

!  form and factorize the shifted matrix P + shift(j) I = LL'

         data%P_shift( 1, : k2m1 ) = data%P( 1, : k2m1 ) + shift
         data%P_shift( 2 : 3, : k2m1 ) = data%P( 2 : 3, : k2m1 )
         CALL PBTRF( 'L', k2m1, 2, data%P_shift, ldp, lapack_info )
         IF ( lapack_info < 0 ) THEN
           inform%status = GALAHAD_error_lapack ; GO TO 920
         END IF

!  solve the shifted system to find the solution s(j)

         data%S2( 1, 1 ) = data%c_norm ; data%S2( 2 : k2m1, 1 ) = zero
         CALL PBTRS( 'L', k2m1, 2, 1, data%P_shift, ldp,                       &
                     data%S2, data%is_max, lapack_info )
         IF ( lapack_info < 0 ) THEN
           inform%status = GALAHAD_error_lapack ; GO TO 920
         END IF
         data%S1( : k2m1 ) = data%S2( : k2m1, 1 )

!  record ||r_2k(t)||^2 = ( alpha_k^2 + gamma^2 ) s_2k(j)^2 

         error = SQRT( data%ALPHA( km1 ) ** 2 + gamma ** 2 )                   &
                   * ABS( data%S1( k2m1 ) )

!  compute ||s(j)||

         s_norm = TWO_NORM( data%S1( 1 : k2m1 ) )
         WRITE( out, "('k = ', I3, ' ||s||, radius = ', 2ES12.4 )" )           &
           k, s_norm, radius

!  if ||s(j)|| > radius_j, update shift(j). First, solve L w_j = s_j ...

         violation = s_norm - radius
         IF ( s_norm > radius ) THEN
           data%W( : k2m1 ) = data%S1( : k2m1 )
           CALL TBSV( 'L', 'N', 'N', k2m1, 2, data%P_shift, ldp,               &
                      data%W, 1_ip_)

!  ... then use the norm of w_j to improve the shift

           w_norm = TWO_NORM( data%W( 1 : k2m1 ) )
           shift = shift + ( s_norm - radius ) * s_norm ** 2                   &
                     / ( radius * w_norm ** 2 )
         END IF

!  debug - compare predicted and actual error

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
           R = shift * SOL + C
           IF ( data%shifted ) THEN
             CALL TREK_Hv( one, data%H, SOL, one, R, out, control%error,       &
                           symmetric = .TRUE. )
           ELSE
             CALL TREK_Hv( one, H, SOL, one, R, out, control%error,            &
                           symmetric = .TRUE. )
           END IF
           WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" )                       &
             TWO_NORM( R ), error_j
         END IF

         IF ( printh ) WRITE( out, 2000 ) prefix
         IF ( printi ) WRITE( out, 2010 ) prefix, k, k2m1,                     &
              s_norm, radius, shift, error, inform%obj
         error = error / data%c_norm

!  check for termination

         IF ( error < data%control%stop_residual .AND.                         &
              violation <= violation_max ) THEN
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

           inform%status = GALAHAD_ok ; EXIT
         END IF
       END IF

!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!  segment involving the product with A inverse
!  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  set u = H^{−1} v_k

       data%U = data%V( 1 : n, k )
       CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
       IF ( data%shifted ) THEN
         CALL SLS_solve( data%H, data%U, data%sls_data,                        &
                         data%control%sls_control, inform%sls_inform )
       ELSE
         CALL SLS_solve( H, data%U, data%sls_data,                             &
                         data%control%sls_control, inform%sls_inform )
       END IF
       CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
       inform%time%solve = inform%time%solve + time_now - time_record
       inform%time%clock_solve =                                         &
         inform%time%clock_solve + clock_now - clock_record
       IF ( inform%sls_inform%status < 0 ) THEN
         inform%status = GALAHAD_error_solve ; GO TO 920
       END IF

!  set beta_{-k} := v_{-k}' u and u = u - beta_{-k} v_{-k}
 
       beta = DOT_PRODUCT( data%V( 1 : n, - k ), data%U )
       data%BETA( - k ) = beta
       data%U = data%U - beta * data%V( 1 : n, - k )

!  set beta k = v_k' u and w := w - beta_k v_k

       beta = DOT_PRODUCT( data%V( 1 : n, k ), data%U )
       data%BETA( k ) = beta
       data%U = data%U - beta * data%V( 1 : n, k )

!  orthogonalise wrt the remaining vectors

!      IF ( .true. ) THEN
       IF ( control%reorthogonalize ) THEN
         DO i = mkp1 + 1, k - 1
           beta = DOT_PRODUCT( data%V( 1 : n, i ), data%U )
           data%U = data%U - beta * data%V( 1 : n, i )
         END DO
       END IF

!  test

       IF ( printd ) THEN
         DO i = - k, k
           DO j = - k, i
             alpha =  DOT_PRODUCT( data%V( 1 : n, i ), data%V( 1 : n, j ) )
             IF ( ( i == j  .AND. ABS( alpha - one ) > ten ** ( -5 ) ) .OR.    &
                  ( i /= j  .AND. ABS( alpha ) > ten ** ( -5 ) ) )  &
               WRITE( out, "( ' i, j, v_ij ', 2I3, ES12.4 )" ) i, j, alpha
           END DO
         END DO
       END IF

!  save column 2k of P

       data%P( 1, k2 ) = data%ALPHA( - k )
       data%P( 2, k2 ) = delta
       data%P( 3, k2 ) = zero

       IF ( printd ) THEN
         IF ( ABS( data%P( 1, k2m1 ) - e11 ) > ten ** ( - 5 ) .OR.             &
              ABS( data%P( 2, k2m1 ) - e12 ) > ten ** ( - 5 ) .OR.             &
              ABS( data%P( 1, k2 ) - e22 ) > ten ** ( - 5 ) ) THEN
           WRITE( out, "( ' theoretical (P) and computed (C) Hessian:', /,     &
          &               ' P_11, C_11 = ', 2ES12.4, /,                        &
          &               ' P_21, C_21 = ', 2ES12.4, /,                        &
          &               ' P_22, C_22 = ', 2ES12.4 )" )                       &
            data%P( 1, k2m1 ), e11, data%P( 2, k2m1 ), e12,                    &
            data%P( 1, k2 ), e22
          END IF
        END IF

!  ....................-------------------
!  test system of order 2k for termination
!  ....................-------------------

!  compute the projected solution ( P + shift(j) I ) s(j) = delta_0 e_1
!  for j in [1,...,m]

       IF (  termination_test ) THEN   
         j_max = 1

!  exact shifts are required

         IF ( control%exact_shift ) THEN

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

           CALL SYEV( 'V','L', k2, data%Q, data%is_max, data%D,                &
                      data%WORK_syev, data%lwork_syev, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  form c' = Q^T ||c|| e_1

!          data%C( 1 : k2 ) = data%c_norm * data%Q( 1, 1 : k2 )
           data%C( 1 : k2 ) = - data%c_norm * data%Q( 1, 1 : k2 )

!  solve the diagonal trust-region problem 
!    min 1/2 x'^T D x' + c'^T x' : ||x'|| <= radius

           data%control%trs_control%use_initial_multiplier = .TRUE.
           IF ( data%shifted ) THEN
             data%control%trs_control%initial_multiplier                       &
               = shift - data%shift_val
             CALL TRS_solve_diagonal( k2, radius, zero,                        &
                                      data%C, data%D, data%X,                  &
                                      data%control%trs_control,                &
                                      inform%trs_inform,                       &
                                      shift = data%shift_val )
             shift = inform%trs_inform%multiplier + data%shift_val
           ELSE
             data%control%trs_control%initial_multiplier = shift
             CALL TRS_solve_diagonal( k2, radius, zero,                        &
                                      data%C, data%D, data%X,                  &
                                      data%control%trs_control,                &
                                      inform%trs_inform )
             shift = inform%trs_inform%multiplier
           END IF
           inform%multiplier = shift
           inform%obj = inform%trs_inform%obj
           inform%x_norm = inform%trs_inform%x_norm

!  recover x = Q x'

           data%S1( 1 : k2 ) = MATMUL( data%Q( : k2 , : k2 ), data%X( : k2 ) )

!  exact shifts are not required

         ELSE

!  form and factorize the shifted matrix P + shift(j) I

           data%P_shift( 1, : k2 ) = data%P( 1, : k2 ) + shift
           data%P_shift( 2 : 3, : k2 ) = data%P( 2 : 3, : k2 )
           CALL PBTRF( 'L', k2, 2, data%P_shift, ldp, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF

!  solve the shifted system to find the solution s(j)

           data%S2( 1, 1 ) = data%c_norm ; data%S2( 2 : k2, 1 ) = zero
           CALL PBTRS( 'L', k2, 2, 1, data%P_shift, ldp,                       &
                       data%S2, data%is_max, lapack_info )
           IF ( lapack_info < 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 920
           END IF
           data%S1( 1 : k2 ) = data%S2( 1 : k2, 1 )
         END IF

!  record ||r_2k(t)||^2 = ( alpha_k s_2k-1(j))^2 + ( alpha_-k s_2k(j))^2

         error = ABS( gamma * data%S1( k2m1 ) + delta * data%S1( k2 ) )

!  compute ||s|| and the violation

         s_norm = TWO_NORM( data%S1( 1 : k2 ) )
         violation = s_norm - radius
         IF ( printm ) WRITE( out,                                             &
           "(' k = ', I0, ' ||s||, radius, multiplier = ', 3ES11.4 )" )        &
              k, s_norm, radius, shift

!  if ||s|| > radius, update shift. First, solve L w = s ...

         IF ( .NOT. control%exact_shift ) THEN
           inform%obj                                                          &
             = half * ( data%c_norm * data%S1( 1 ) - shift * s_norm ** 2 )
!          IF ( s_norm > radius ) THEN
           IF ( .TRUE. ) THEN
             data%W( : k2 ) = data%S1( : k2 )
             CALL TBSV( 'L', 'N', 'N', k2, 2, data%P_shift, ldp,               &
                        data%W, 1_ip_)

!  ... then use the norm of w_j to improve the shift

             w_norm = TWO_NORM( data%W( 1 : k2 ) )
             shift = shift + ( s_norm - radius ) * s_norm ** 2                 &
                       / ( RADIUS * w_norm ** 2 )
           END IF
         END IF

!  record ||s|| and the current shift

         inform%x_norm = s_norm
         inform%multiplier = shift

!  debug - compare predicted and actual error

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
           R = shift * SOL + C
           IF ( data%shifted ) THEN
             CALL TREK_Hv( one, data%H, SOL, one, R, out, control%error,       &
                           symmetric = .TRUE. )
           ELSE
             CALL TREK_Hv( one, H, SOL, one, R, out, control%error,            &
                           symmetric = .TRUE. )
           END IF
           WRITE( out, "( ' ||r||, est = ', 2ES12.4 )" ) TWO_NORM( R ), error
         END IF

         IF ( printh ) WRITE( out, 2000 ) prefix
         IF ( printi ) WRITE( out, 2010 ) prefix, k, k2,                       &
              s_norm, radius, shift, error, inform%obj
         error = error / data%c_norm

!  check for termination

         IF ( error < data%control%stop_residual .AND.                         &
              violation <= violation_max ) THEN
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
           data%last_shift = shift
           inform%status = GALAHAD_ok ; EXIT
         END IF
       END IF

!  set delta_{-k-1} := ||u|| 

       delta = TWO_NORM( data%U )

!  set v_{-k-1} = u/delta_{-k-1}

       IF ( k < data%k_max ) THEN
         data%DELTA( - k - 1 ) = delta
         data%V( 1 : n, - k - 1 ) = data%U / delta
         data%n_v = data%n_v + 1
         IF ( data%n_v == n ) WRITE( out, 2020 ) prefix
       END IF

!  ------------------------------------
!  end of main backward iteration loop
!  ------------------------------------

     END DO
     inform%error = error

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
           CALL TREK_Hv( one, data%H, data%V( 1 : n, i ), zero, data%U,        &
                         out, control%error, symmetric = .TRUE. )
         ELSE
           CALL TREK_Hv( one, H, data%V( 1 : n, i ), zero, data%U,             &
                         out, control%error, symmetric = .TRUE. )
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
           WRITE( out, "( ' P(', I3, ') = ', 3ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ), &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) ), &
                 ABS( data%P( 3, ii ) - P_calc( ii + 2, ii ) )
         ELSE IF ( ii + 1 <= p_dim ) THEN
           WRITE( out, "( ' P(', I3, ') = ', 2ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) ), &
                 ABS( data%P( 2, ii ) - P_calc( ii + 1, ii ) )
         ELSE
           WRITE( out, "( ' P(', I3, ') = ', ES22.14 )" ) &
             ii, ABS( data%P( 1, ii ) - P_calc( ii, ii ) )
         END IF
       END DO
     END IF

!  successful return

 900 CONTINUE
     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total = inform%time%clock_total + clock_now - clock_start
     RETURN

!  allocation error

 910 CONTINUE
     IF ( out > 0 .AND. control%print_level > 0 )                              &
       WRITE( control%out, "( A, '   **  Allocation error return ', I0,        &
      & ' from TREK ' )" ) prefix, inform%status
     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total = inform%time%clock_total + clock_now - clock_start
     RETURN

!  other error returns

 920 CONTINUE
     IF ( out > 0 .AND. control%print_level > 0 )                              &
       WRITE( control%out, "( A, '   **  Error return ', I0,                   &
      & ' from TREK ' )" ) prefix, inform%status
     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total = inform%time%clock_total + clock_now - clock_start
     RETURN

!  non-executable statements

2000 FORMAT( A, '    k d(K)   ||x||     radius      shift      error         f')
2010 FORMAT( A, 2I5, 4ES11.4, ES12.4 )
2020 FORMAT( A, 8X, ' - Krylov space is full space -' )

!  end of subroutine TREK_solve

     END SUBROUTINE TREK_solve

!-*-*-*-*-   T R E K _ H _ f i n d _ s h i f t   S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE TREK_find_shift( H, shift, lower, status )

!  find a (small) shift so that H + shift I is positive definite

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ),  INTENT( OUT ) :: status
     TYPE ( SMT_type ), INTENT( INOUT ) :: H
     REAL ( KIND = rp_), INTENT( OUT ) :: shift
     REAL ( KIND = rp_), DIMENSION( H%n ), INTENT( OUT ) :: LOWER

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, j, l, h_ne

     status = 0

!  Lower gives the lower bound on each Gershgorin interval

     LOWER( : H%n ) = zero
     SELECT CASE ( SMT_get( H%type ) )
     CASE ( 'COORDINATE' )
       DO l = 1, H%ne
         i =  H%row( l ) ; j = H%col( l )
         IF ( i == j ) THEN
           LOWER( i ) = LOWER( i ) + H%val( l )
         ELSE
           LOWER( i ) = LOWER( i ) - ABS( H%val( l ) )
           LOWER( j ) = LOWER( j ) - ABS( H%val( l ) )
         END IF
       END DO
       h_ne = H%ne
     CASE ( 'SPARSE_BY_ROWS' )
       DO i = 1, H%n
         DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
           j = H%col( l )
           IF ( i == j ) THEN
             LOWER( i ) = LOWER( i ) + H%val( l )
           ELSE
             LOWER( i ) = LOWER( i ) - ABS( H%val( l ) )
             LOWER( j ) = LOWER( j ) - ABS( H%val( l ) )
           END IF
         END DO
       END DO
       h_ne = H%ptr( H%ne + 1 ) - 1
     CASE ( 'DENSE' )
       l = 0
       DO i = 1, H%n
         DO j = 1, i
           l = l + 1
           IF ( i == j ) THEN
             LOWER( i ) = LOWER( i ) + H%val( l )
           ELSE
             LOWER( i ) = LOWER( i ) - ABS( H%val( l ) )
             LOWER( j ) = LOWER( j ) - ABS( H%val( l ) )
           END IF
         END DO
       END DO
       h_ne = l
     END SELECT
!write(6,*) LOWER

!  the required lower bound is minus the lowest lower interval bound

     shift = - MINVAL( LOWER( : H%n ) )
     
!  add a very small perturbation

     shift = shift + h_pert * MAXVAL( ABS( H%val( : h_ne ) ) )

!  end of SUBROUTINE TREK_find_shift

     END SUBROUTINE TREK_find_shift

! -*-*-  G A L A H A D -  T R E K _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE TREK_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine TREK_initialize
!   control see Subroutine TREK_initialize
!   inform  see Subroutine TREK_find

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( TREK_data_type ), INTENT( INOUT ) :: data
      TYPE ( TREK_control_type ), INTENT( IN ) :: control
      TYPE ( TREK_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

      data%allocated_arrays = .FALSE.
      data%k_exit = - 1

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      inform%status = inform%SLS_inform%status
      IF ( inform%SLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'trek: data%SLS'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaing allocated arrays

      array_name = 'trek: data%MAP'
      CALL SPACE_dealloc_array( data%MAP,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%P_shift'
      CALL SPACE_dealloc_array( data%P_shift,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%V'
      CALL SPACE_dealloc_array( data%V,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%S1'
      CALL SPACE_dealloc_array( data%S1,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%S2'
      CALL SPACE_dealloc_array( data%S2,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%Q'
      CALL SPACE_dealloc_array( data%Q,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%D'
      CALL SPACE_dealloc_array( data%D,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%X'
      CALL SPACE_dealloc_array( data%X,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%WORK_syev'
      CALL SPACE_dealloc_array( data%WORK_syev,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%SHIFT'
      CALL SPACE_dealloc_array( data%SHIFT,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%ALPHA'
      CALL SPACE_dealloc_array( data%ALPHA,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%BETA'
      CALL SPACE_dealloc_array( data%BETA,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%DELTA'
      CALL SPACE_dealloc_array( data%DELTA,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%U1'
      CALL SPACE_dealloc_array( data%U1,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%H%ptr'
      CALL SPACE_dealloc_array( data%H%ptr,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%H%col'
      CALL SPACE_dealloc_array( data%H%col,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'trek: data%H%val'
      CALL SPACE_dealloc_array( data%H%val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      IF ( ALLOCATED( data%H%type ) )                                          &
        DEALLOCATE( data%H%type, STAT = inform%alloc_status )

      RETURN

!  End of subroutine TREK_terminate

      END SUBROUTINE TREK_terminate

! -  G A L A H A D -  T R E K _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE TREK_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( TREK_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( TREK_control_type ), INTENT( IN ) :: control
     TYPE ( TREK_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

!    CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL TREK_terminate( data%trek_data, control, inform )

!  deallocate any internal problem arrays

!    array_name = 'trek: data%prob%X'
!    CALL SPACE_dealloc_array( data%prob%X,                                    &
!       inform%status, inform%alloc_status, array_name = array_name,           &
!       bad_alloc = inform%bad_alloc, out = control%error )
!    IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine TREK_full_terminate

     END SUBROUTINE TREK_full_terminate

!  end of module TREK_precision

   END MODULE GALAHAD_TREK_precision

