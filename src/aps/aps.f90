! THIS VERSION: GALAHAD 3.0 - 11/03/2018 AT 11:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ A P S   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally from HSL VF06 June 30th 1997
!   modified for GALAHAD Version 2.5. April 12th 2011
!   extended for the regularization subproblem, March 11th 2018

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_APS_DOUBLE

!      --------------------------------------------
!      | Solve the trust-region subproblem        |
!      |                                          |
!      |    minimize     1/2 <x, H x> + <c, x>    |
!      |    subject to   <x, M x> <= delta^2      |
!      |                                          |
!      | or the regularized quadratic subproblem  |
!      |                                          |
!      |    minimize     1/2 <x, H x> + <c, x>    |
!      |                   + (sigma/p) ||x||_M^p  |
!      |                                          |
!      | in the modified absolute-value norm, M   |
!      --------------------------------------------

      USE GALAHAD_SMT_double
!     USE GALAHAD_SILS_double
      USE GALAHAD_SLS_double
      USE GALAHAD_NORMS_double
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: APS_initialize, APS_solve, APS_resolve, APS_terminate,         &
                SMT_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: aps_flags = 2
      INTEGER, PARAMETER :: SILS_flags = 6
      INTEGER, PARAMETER :: flags = aps_flags + SILS_flags
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = KIND( 1.0E+0 ) ) time, timset

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: APS_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  an initial guess as to the integer workspace required by SBLS     (OBSOLETE)

        INTEGER :: indmin = 1000

!  an initial guess as to the real workspace required by SBLS        (OBSOLETE)

        INTEGER :: valmin = 1000

!   the threshold pivot used by the matrix factorization.
!    See the documentation for SBLS for details                      (OBSOLETE)

        REAL ( KIND = wp ) :: pivot_tol = 0.01_wp

!  smallest allowable value of an eigenvalue of the block diagonal factor of H

        REAL ( KIND = wp ) :: delta_eigen = EPSILON( one )
      END TYPE

      TYPE, PUBLIC :: APS_inform_type

!  return status. See APS_solve for details

        INTEGER :: status = 0

!  amount of integer storage used for the factorization

        INTEGER :: SILSiw = - 1

!  amount of real storage used for the factorization

        INTEGER :: SILSrw = - 1

!  the number of 1 by 1 blocks from the factorization of H that were modified
!   when constructing M

        INTEGER :: mod_1by1 = 0

!  the number of 2 by 2 blocks from the factorization of H that were modified
!   when constructing M

        INTEGER :: mod_2by2 = 0

!  the Lagrange multiplier associated with the constraint

        REAL ( KIND = wp ) :: multiplier = - one

!  the M-norm of the solution

        REAL ( KIND = wp ) :: norm_step = - one

!  information from SLS

        TYPE ( SLS_inform_type ) :: SLS_inform

     END TYPE APS_inform_type

      TYPE, PUBLIC :: APS_data_type
        PRIVATE
        REAL ( KIND = wp ) :: old_delta, old_multiplier, old_f
        INTEGER :: val, ind, nsteps, maxfrt, latop
        LOGICAL :: old_on_boundary, old_convex
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: PERM
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: EVAL, MOD_EVAL
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: EVECT, CS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: D
!       TYPE ( SILS_CONTROL ) :: CNTL
        TYPE ( SLS_control_type ) :: SLS_control
!       TYPE ( SILS_FACTORS ) :: FACTORS
        TYPE ( SLS_data_type ) :: SLS_data
      END TYPE

   CONTAINS

!-*-*-*-*-*-  A P S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

!     SUBROUTINE APS_initialize( data, control )
      SUBROUTINE APS_initialize( data, control, inform )

!      ..............................................
!      .                                            .
!      .  Set initial values for the APS           .
!      .  control parameters                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========

!   data     private internal data
!   control  a structure containing control information. Components are -
!            error       error message output unit
!            out         information message output unit
!            print_level print level. > 0 for output
!            valmin      minimum real storage allocated for factorization
!            indmin      minimum integer storage allocated for factorization

!-----------------------------------------------
!   D u m m y   A r g u m e n t
!-----------------------------------------------

      TYPE ( APS_data_type ), INTENT( OUT ) :: data
      TYPE ( APS_control_type ), INTENT( OUT ) :: control
      TYPE ( APS_inform_type ), INTENT( OUT ) :: inform

!  Initalize SILS components

!     CALL SILS_INITIALIZE( data%FACTORS, data%CNTL )
      CALL SLS_INITIALIZE( data%SLS_data, data%SLS_control, inform%SLS_inform )
      data%SLS_control%scaling = 0

!  Set initial control parameter values

      control%delta_eigen = SQRT( EPSILON( one ) )

!  Ensure that the initial value of the "old" delta is small

      data%old_delta = zero
      data%old_on_boundary = .FALSE. ; data%old_convex = .FALSE.

      RETURN

!  End of subroutine APS_initialize

      END SUBROUTINE APS_initialize

!-*-*-*-*-*-*-*-*-*-*  A P S _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE APS_solve( n, delta, C, H, new_H, f, X, data, control, inform )

!      .................................................
!      .                                               .
!      .  Obtain the modified absolute-value           .
!      .  preconditioner, M, and solve the             .
!      .  trust-region problem                         .
!      .                                               .
!      .     minimize <x,c> + 1/2 <x,Hx>               .
!      .     subject to ||x||_M <= delta               .
!      .                                               .
!      .................................................

!  Arguments:
!  =========
!
!   n        number of unknowns
!   delta    trust-region radius
!   C        the vector c
!   H        the matrix H
!   new_H    indicates that this is the first time a matrix H
!            with the current sparsity structure is to be used
!   f        the value of the quadratic function at the solution.
!            Need not be set on entry. On exit it will contain the value at
!            the best point found
!   X        the required solution vector. Need not be set on entry.
!            On exit, the optimal value
!   data     private internal data
!   control  a structure containing control information. See APS_initialize.
!   inform     a structure containing information. Components are
!             multiplier  the Lagrange multiplier corresponding to the
!                         trust-region constraint
!             SILSiw      integer storage required to hold factors
!             SILSrw      real storage required to hold factors
!             status      exit status
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: delta
      REAL ( KIND = wp ), INTENT( OUT ) :: f
      LOGICAL, INTENT( IN ) :: new_H
      TYPE ( SMT_type ), INTENT( INOUT ) :: H
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE ( APS_data_type ), INTENT( INOUT ) :: data
      TYPE ( APS_control_type ), INTENT( IN ) :: control
      TYPE ( APS_inform_type ), INTENT( OUT ) :: inform

      IF ( n <= 0 ) THEN
         IF ( control%error > 0 )                                              &
              WRITE( control%error, "( ' n = ', I6, ' is not positive ' )" ) n
         inform%status = - 1
         RETURN
      END IF

      IF ( delta <= zero ) THEN
         IF ( control%error > 0 ) WRITE( control%error,                        &
              "( ' The radius ', ES12.4 , ' is not positive ' )" ) delta
         inform%status = - 2
         RETURN
      END IF
      H%n = n

!  Obtain the trust-region norm

      CALL APS_build_preconditioner( n, H, new_H, data, control, inform )
      IF ( inform%status < 0 ) RETURN
      data%old_convex = inform%mod_1by1 == 0 .AND. inform%mod_2by2 == 0

!  Solve the TR problem

      CALL APS_resolve( n, delta, X, f, data, control, inform, C = C )

      RETURN

!  Non-executable statements

!  End of subroutine APS_solve

      END SUBROUTINE APS_solve

!-*-*-*-*-*-*-*-  A P S _ R E S O L V E   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE APS_resolve( n, delta, X, f, data, control, inform, C )

!      .................................................
!      .                                               .
!      .  Solve the trust-region problem               .
!      .                                               .
!      .     minimize <x,c> + 1/2 <x,Hx>               .
!      .     subject to ||x||_M <= delta,              .
!      .                                               .
!      .  where M is the modified absolute value of H  .
!      .                                               .
!      .................................................

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: delta
      REAL ( KIND = wp ), INTENT( OUT ) :: f
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: C
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE ( APS_data_type ), INTENT( INOUT ) :: data
      TYPE ( APS_control_type ), INTENT( IN ) :: control
      TYPE ( APS_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      INTEGER :: i, alloc_stat
      REAL ( KIND = wp ) :: temp
      LOGICAL :: reallocate, oneby1, printe, debug
      CHARACTER ( LEN = 8 ) :: bad_alloc
!     REAL ( KIND = wp ), DIMENSION( n ) :: P

!  Step 1: Initialize
!  ======

      printe = control%error > 0 .AND. control%print_level >= 1
      debug = control%out > 0 .AND. control%print_level >= 10
      inform%status = 0

!  Allocate the arrays for the solution phase

      IF ( PRESENT( C ) ) THEN

         reallocate = .TRUE.
         IF ( ALLOCATED( data%CS ) ) THEN
            IF ( SIZE( data%CS ) < n ) THEN ; DEALLOCATE( data%CS )
            ELSE ; reallocate = .FALSE.
            END IF
         END IF
         IF ( reallocate ) THEN
            ALLOCATE( data%CS( n ), STAT = alloc_status )
            IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%CS' ; GO TO 910
            END IF
         END IF

         IF ( debug ) WRITE( control%out, 2030 ) 'PERM', data%PERM( : n )
         data%CS = C
         IF ( debug ) WRITE( control%out, 2020 ) 'data%CS', data%CS( : n )

!  Step 2: Solve P L c_b = c
!  ======

!        CALL SILS_PART_SOLVE( data%FACTORS, data%CNTL, 'L', data%CS,          &
!                              alloc_stat )
         CALL SLS_part_solve( 'L', data%CS, data%SLS_data, data%SLS_control,   &
                              inform%SLS_inform )

         IF ( inform%SLS_inform%alloc_status /= 0 ) THEN
            IF ( printe ) WRITE( control%error, 2010 )                         &
              inform%SLS_inform%alloc_status
            inform%status = - flags - inform%SLS_inform%alloc_status
            RETURN
         END IF
         IF ( debug ) WRITE( control%out, 2020 ) 'data%CS', data%CS( : n )

!  Step 3: Obtain c_s = Gamma(-1/2) Q^T P^T c_b
!  ======

         data%CS = data%CS( ABS( data%PERM ) )
         IF ( debug ) WRITE( control%out, 2020 ) 'data%CS', data%CS( : n )

         oneby1 = .TRUE.
         DO i = 1, n
            IF ( oneby1 ) THEN
               IF ( i < n ) THEN
                  oneby1 = data%PERM( i ) > 0
               ELSE
                  oneby1 = .TRUE.
               END IF
               IF ( .NOT. oneby1 ) THEN

!  2x2 pivot

                  temp = data%EVECT( i + 1 ) * data%CS( i ) -                  &
                         data%EVECT( i ) * data%CS( i + 1 )
                  data%CS( i ) = data%EVECT( i ) * data%CS( i ) +              &
                                 data%EVECT( i + 1 ) * data%CS( i + 1 )

                  data%CS( i + 1 ) = temp / SQRT( data%MOD_EVAL( i + 1 ) )
               END IF
               data%CS( i ) = data%CS( i ) / SQRT( data%MOD_EVAL( i ) )
            ELSE
               oneby1 = .TRUE.
            END IF
         END DO
      ELSE

!  Special case: the previous problem was convex, and the solution lay on
!  the trust region boundary. The new solution is then simply a rescaled
!  version of the former solution

         IF ( data%old_convex .AND. data%old_on_boundary .AND.                 &
              delta <= data%old_delta ) THEN
            temp = delta / data%old_delta
            f = temp * data%old_f + half * delta * ( delta - data%old_delta )
            X = temp * X
            inform%multiplier = ( one + data%old_multiplier ) / temp - one
            inform%norm_step = delta
            GO TO 900
         END IF

      END IF

!  Step 4: Find x_s by solving the diagonal TR problem
!  ======

      IF ( debug ) WRITE( control%out, 2020 ) 'data%CS', data%CS( : n )
      CALL APS_solve_diagonal_tr( n, ( data%EVAL / data%MOD_EVAL ), data%CS,   &
                                   delta, X, f, control, inform )
      IF ( debug ) WRITE( control%out, 2020 ) 'X', X( : n )


!  Step 5 (alternative): Recover x_r = P Q Gamma(-1/2) x_s
!  ======

      oneby1 = .TRUE.
      DO i = 1, n
         IF ( oneby1 ) THEN
            IF ( i < n ) THEN
               oneby1 = data%PERM( i ) > 0
            ELSE
               oneby1 = .TRUE.
            END IF
            X( i ) = X( i ) / SQRT( data%MOD_EVAL( i ) )
            IF ( .NOT. oneby1 ) THEN

!  2x2 pivot

               temp = X( i + 1 ) / SQRT( data%MOD_EVAL( i + 1 ) )
               X( i + 1 ) = data%EVECT( i + 1 ) * X( i ) -                     &
                            data%EVECT( i ) * temp
               X( i ) = data%EVECT( i ) * X( i ) + data%EVECT( i + 1 ) * temp
            END IF
         ELSE
            oneby1 = .TRUE.
         END IF
      END DO

      X( ABS( data%PERM ) ) = X

!  Step 6: (Alternative) Solve P L(trans) P(trans) x = x_r
!  ======

!     CALL SILS_PART_SOLVE( data%FACTORS, data%CNTL, 'U', X, alloc_status )
      CALL SLS_part_solve( 'U', X, data%SLS_data, data%SLS_control,           &
                           inform%SLS_inform )

      IF ( inform%SLS_inform%alloc_status /= 0 ) THEN
         IF ( printe ) WRITE( control%error, 2010 ) inform%SLS_inform%alloc_status
         inform%status = - flags - inform%SLS_inform%alloc_status
         RETURN
      END IF
      IF ( debug ) WRITE( control%out, 2020 ) 'X', X( : n )

!  Successful return

  900 CONTINUE

      data%old_delta = delta ; data%old_multiplier = inform%multiplier
      data%old_f = f ; data%old_on_boundary = inform%multiplier > zero

      RETURN

!  Unsuccessful returns

  910 CONTINUE
      inform%status = - flags - inform%SLS_inform%alloc_status
      IF ( control%error > 0 .AND. control%print_level >= 1 )                  &
         WRITE( control%error, 2000 ) bad_alloc, inform%SLS_inform%alloc_status
      RETURN

!  Non-executable statements

 2000 FORMAT( ' ** Message from -APS_resolve-', /,                             &
                 ' Allocation error, for ', A8, ', status = ', I6 )
 2010 FORMAT( '   **  Error return ', I3, ' from SILS_part_solve ' )
 2020 FORMAT( A10, /, ( 6ES12.4 ) )
 2030 FORMAT( A10, /, ( 10I8 ) )

      END SUBROUTINE APS_resolve

!-*-*-*-*-   H S L _ A P S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*

      SUBROUTINE APS_terminate( data, control, inform )

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see Subroutine APS_initialize
!   inform    see Subroutine APS_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( APS_data_type ), INTENT( INOUT ) :: data
      TYPE ( APS_control_type ), INTENT( IN ) :: control
      TYPE ( APS_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      INTEGER :: alloc_stat
      LOGICAL :: printe

      inform%status = 0
      printe = control%error > 0 .AND. control%print_level >= 1

!  Deallocate all internal arrays

      IF ( ALLOCATED( data%PERM ) ) THEN
         DEALLOCATE( data%PERM, STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN
            inform%status = - flags - alloc_stat
            IF ( printe ) WRITE( control%error, 2900 ) 'data%PERM'
         END IF
      END IF

      IF ( ALLOCATED( data%D ) ) THEN
         DEALLOCATE( data%D, STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN
            inform%status = - flags - alloc_stat
            IF ( printe ) WRITE( control%error, 2900 ) 'data%D'
         END IF
      END IF

      IF ( ALLOCATED( data%EVAL ) ) THEN
         DEALLOCATE( data%EVAL, STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN
            inform%status = - flags - alloc_stat
            IF ( printe ) WRITE( control%error, 2900 ) 'data%EVAL'
         END IF
      END IF

      IF ( ALLOCATED( data%MOD_EVAL ) ) THEN
         DEALLOCATE( data%MOD_EVAL, STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN
            inform%status = - flags - alloc_stat
            IF ( printe ) WRITE( control%error, 2900 )'data%MOD_EVAL'
         END IF
      END IF

      IF ( ALLOCATED( data%EVECT ) ) THEN
         DEALLOCATE( data%EVECT, STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN
            inform%status = - flags - alloc_stat
            IF ( printe ) WRITE( control%error, 2900 ) 'data%EVECT'
         END IF
      END IF

      IF ( ALLOCATED( data%CS ) ) THEN
         DEALLOCATE( data%CS, STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN
            inform%status = - flags - alloc_stat
            IF ( printe ) WRITE( control%error, 2900 ) 'data%CS'
         END IF
      END IF

!     CALL SILS_FINALIZE( data%FACTORS, data%CNTL, alloc_stat )
      CALL SLS_terminate( data%SLS_data, data%SLS_control, inform )
      IF ( inform%alloc_stat /= 0 ) THEN
         inform%status = - flags - inform%alloc_stat
         IF ( printe ) WRITE( control%error, 2900 ) 'data%SLS_data'
      END IF

      RETURN

!  Non-executable statement

 2900 FORMAT( /, ' ** Message from -APS_terminate-', /,                        &
              '    Deallocation error for array data%', A14 )

!  End of subroutine APS_terminate

      END SUBROUTINE APS_terminate

!-*-  A P S _ B U I L D _ P R E C O N D I T I O N E R  S U B R O U T I N E   -*-

      SUBROUTINE APS_build_preconditioner( n, H, new_H, data, control, inform )

!      .................................................
!      .                                               .
!      .  Obtain the modified absolute-value           .
!      .  preconditioner of the sparse                 .
!      .  symmetric matrix H                           .
!      .                                               .
!      .................................................

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      LOGICAL, INTENT( IN ) :: new_H
      TYPE ( SMT_type ), INTENT( INOUT ) :: H
      TYPE ( APS_data_type ), INTENT( INOUT ) :: data
      TYPE ( APS_control_type ), INTENT( IN ) :: control
      TYPE ( APS_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: zeig, alloc_stat, rank, out
      LOGICAL :: printe, printt, reallocate
      CHARACTER ( LEN = 6 ) :: bad_alloc

!-----------------------------------------------
!   M A 2 7    V a r i a b l e s
!-----------------------------------------------

!     TYPE ( SILS_AINFO ) :: AINFO
!     TYPE ( SILS_FINFO ) :: FINFO

!  Set output levels

      out = control%out
      printe = control%error > 0 .AND. control%print_level >= 1
      printt = out > 0 .AND. control%print_level >= 2

      inform%status = 0

! ::::::::::::::::::::::::::::::::::
!  Analyse the sparsity pattern of H
! ::::::::::::::::::::::::::::::::::

      IF ( new_H ) THEN

!  Initialize the data structures

         data%SLS_control%u = control%pivot_tol
         data%SLS_control%lp = control%error
         data%SLS_control%mp = control%out
         data%SLS_control%wp = control%out
         IF ( control%print_level > 2 ) THEN
            IF ( control%print_level < 10 ) THEN
               data%SLS_control%ldiag = 1
            ELSE
               data%SLS_control%ldiag = 2
            END IF
         END IF

!  Perform the analysis

         IF ( printt ) CALL CPU_TIME( time )
!        CALL SILS_ANALYSE( H, data%FACTORS, data%SLS_CNTL, AINFO )
         CALL SLS_analyse( H, data%SLS_data, data%SLS_control,                &
                           inform%SLS_inform )
         IF ( printt ) THEN
            CALL CPU_TIME( timset )
            WRITE( out, "( ' time(anal) = ', F10.2 )" ) timset - time
         END IF

!  Record the storage requested

         inform%SILSiw = SLS_inform%nirnec ; inform%SILSrw = SLS_inform%nrlnec

!  Check for error returns

         IF ( SLS_INFORM%flag < 0 ) THEN
            IF ( printe ) WRITE( control%error, 2020 ) SLS_inform%flag
            inform%status = - aps_flags + SLS_inform%flag
            RETURN
         ELSE IF ( SLS_INFORM%flag > 0 ) THEN
            IF ( printe ) WRITE( control%error, 2030 ) SLS_inform%flag
         END IF

!  Assign initial lengths for the FACTOR arrays

         data%val = MAX( 2 * inform%SILSrw, control%valmin )
         data%ind = MAX( 2 * inform%SILSiw, control%indmin )
         data%SLS_control%la = data%val ; data%SLS_control%liw = data%ind

      END IF

! ::::::::::::
!  Factorize H
! ::::::::::::

      IF ( printt ) CALL CPU_TIME( time )
!     CALL SILS_FACTORIZE( H, data%FACTORS, data%SLS_CONTROL, FINFO )
      CALL SLS_FACTORIZE( H, data%SLS_data, data%SLS_control, SLS_inform )
      IF ( printt ) THEN
        CALL CPU_TIME( timset )
        WRITE( out, "( ' time(fact) = ', F10.2 )" ) timset - time
      END IF

!  Record the storage required

      inform%SILSiw = SLS_inform%nirbdu ; inform%SILSrw = SLS_inform%nrlbdu

!  Test that the factorization succeeded

      IF ( SLS_inform%flag < 0 ) THEN
         IF ( printe ) WRITE( control%error, 2040 ) SLS_inform%flag
         inform%status = - aps_flags + SLS_inform%flag
         RETURN

!  Record warning conditions

      ELSE IF ( SLS_INFORM%flag > 0 ) THEN
         IF ( printt ) WRITE( control%out, 2050 ) SLS_inform%flag
         IF ( SLS_inform%flag == 4 ) THEN
            rank = SLS_inform%rank
            IF ( printt ) WRITE( control%out, 2010 ) n - rank
         ELSE
            rank = n
         END IF
      ELSE
         rank = n
      END IF
      zeig = n - rank

!  Allocate further arrays

      reallocate = .TRUE.
      IF ( ALLOCATED( data%PERM ) ) THEN
         IF ( SIZE( data%PERM ) < n ) THEN ; DEALLOCATE( data%PERM )
         ELSE ; reallocate = .FALSE.
         END IF
      END IF
      IF ( reallocate ) THEN
         ALLOCATE( data%PERM( n ), STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN ; bad_alloc = 'PERM' ; GO TO 910
         END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%EVAL ) ) THEN
         IF ( SIZE( data%EVAL ) < n ) THEN ; DEALLOCATE( data%EVAL )
         ELSE ; reallocate = .FALSE.
         END IF
      END IF
      IF ( reallocate ) THEN
         ALLOCATE( data%EVAL( n ), STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN ; bad_alloc = 'EVAL' ; GO TO 910
         END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%MOD_EVAL ) ) THEN
         IF ( SIZE( data%MOD_EVAL ) < n ) THEN ; DEALLOCATE( data%MOD_EVAL )
         ELSE ; reallocate = .FALSE.
         END IF
      END IF
      IF ( reallocate ) THEN
         ALLOCATE( data%MOD_EVAL( n ), STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN ; bad_alloc = 'data%MOD_EVAL' ; GO TO 910
         END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%EVECT ) ) THEN
         IF ( SIZE( data%EVECT ) < n ) THEN ; DEALLOCATE( data%EVECT )
         ELSE ; reallocate = .FALSE.
         END IF
      END IF
      IF ( reallocate ) THEN
         ALLOCATE( data%EVECT( n ), STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN ; bad_alloc = 'data%EVECT' ; GO TO 910
         END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%D ) ) THEN
         IF ( SIZE( data%D, 1 ) /= 2  .OR. SIZE( data%D, 2 ) < n ) THEN
            DEALLOCATE( data%D )
         ELSE ; reallocate = .FALSE.
         END IF
      END IF
      IF ( reallocate ) THEN
         ALLOCATE( data%D( 2, n ), STAT = alloc_stat )
         IF ( alloc_stat /= 0 ) THEN ; bad_alloc = 'D' ; GO TO 910
         END IF
      END IF

!  ::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Modify the factorization to produce the preconditioner
!  ::::::::::::::::::::::::::::::::::::::::::::::::::::::

      CALL APS_modify_factors( n, control%delta_eigen,                         &
                               inform%mod_1by1, inform%mod_2by2, rank,         &
                               data%SLS_data, inform%SLS_inform,               &
                               data%PERM( : n ),                               &
                               data%EVAL( : n ), data%MOD_EVAL( : n ),         &
                               data%EVECT( : n ), data%D( : 2, : n ) )
      RETURN

!  Unsuccessful returns

  910 CONTINUE
      inform%status = - flags - alloc_stat
      IF ( printe ) WRITE( control%error, 2000 ) bad_alloc, alloc_stat
      RETURN

!  Non-executable statements

 2000 FORMAT( ' ** Message from -APS_build_preconditioner-', /,                &
                 ' Allocation error, for ', A6, ', status = ', I6 )
 2010 FORMAT( I6, ' zero eigenvalues ' )
 2020 FORMAT( '   **  Error return ', I3, ' from SILS_analyse ' )
 2030 FORMAT( '   **  Warning ', I3, ' from SILS_analyse ' )
 2040 FORMAT( '   **  Error return ', I3, ' from SILS_factorize ' )
 2050 FORMAT( '   **  Warning ', I3, ' from SILS_factorize ' )

      END SUBROUTINE APS_build_preconditioner

!-*-*-  A P S _ S O L V E _ D I A G O N A L _ T R   S U B R O U T I N E   -*-*

      SUBROUTINE APS_solve_diagonal_tr( n, D, C, delta, X, f, control, inform )

!      .................................................
!      .                                               .
!      .  Solve the diagonal trust-region problem      .
!      .                                               .
!      .     minimize f = <x,c> + 1/2 <x,Dx>           .
!      .     subject to ||x||_2 <= delta              .
!      .                                               .
!      .................................................

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: delta
      REAL ( KIND = wp ), INTENT( OUT ) :: f
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: D, C
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE( APS_control_type ), INTENT( IN ) :: control
      TYPE( APS_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      INTEGER :: i, leftmost
      REAL ( KIND = wp ) :: alpha, xdplusx, dmultiplier, tol
      LOGICAL :: printd

!  Step 0: Set convergence tolerance
!  ======

      tol = ten * EPSILON( one ) * n
      printd = control%out > 0 .AND. control%print_level >= 3

      IF ( COUNT( D /= one ) /= 0 ) THEN
!     IF ( COUNT( ABS( D ) /= one ) /= 0 ) THEN

         inform%multiplier = - MINVAL( D )

         IF ( inform%multiplier < zero ) THEN

!  Step 1: Strictly convex problem
!  ======

            inform%multiplier = zero
            X = - C / D
            inform%norm_step = TWO_NORM( X )

!  Step 1a: Interior solution
!  ======

            IF ( printd ) WRITE( control%out, 2000 ) inform%multiplier,        &
              inform%norm_step, delta, inform%norm_step - delta
            IF ( inform%norm_step <= delta ) GO TO 900

         ELSE

!  Step 2: Nonconvex problem
!  ======

            IF ( COUNT( D + inform%multiplier == zero .AND.                    &
                 C /= zero ) == 0) THEN

!  Step 2a: Potential hard case
!  ======

               WHERE ( D + inform%multiplier /= zero )
                  X = - C / ( D + inform%multiplier )
               ELSEWHERE
                  X = zero
               END WHERE
               inform%norm_step = TWO_NORM( X )
               IF ( printd ) WRITE( control%out, 2000 ) inform%multiplier,     &
                    inform%norm_step, delta, inform%norm_step - delta

               IF ( inform%norm_step <= delta ) THEN

!  Step 2a(i): Hard case
!  ======

                  leftmost = COUNT( D + inform%multiplier == zero )
                  alpha = SQRT( ( delta * delta - inform%norm_step *           &
                          inform%norm_step ) / leftmost )
                  WHERE ( D + inform%multiplier == zero ) X = X + alpha
                  inform%norm_step = TWO_NORM( X )
                  IF ( printd ) WRITE( control%out, 2000 ) inform%multiplier,  &
                       inform%norm_step, delta, inform%norm_step - delta
                  GO TO 900
               END IF

            ELSE

!  Step 2b: Not hard case
!  ======

               IF ( printd ) WRITE( control%out, 2010 ) inform%multiplier, delta
               xdplusx = zero
               DO i = 1, n
                  IF ( D( i ) + inform%multiplier == zero )                    &
                       xdplusx = xdplusx + C( i ) ** 2
               END DO

!  Compute an improved multiplier

               inform%multiplier = inform%multiplier + SQRT( xdplusx ) / delta
               X = - C / ( D + inform%multiplier )
               inform%norm_step = TWO_NORM( X )
               IF ( printd ) WRITE( control%out, 2000 ) inform%multiplier,     &
                    inform%norm_step, delta, inform%norm_step - delta

            END IF
         END IF

!  Compute <x,(D+multiplierI)(inv)x>

         xdplusx = zero
         DO i = 1, n
           IF ( D( i ) + inform%multiplier /= zero )                           &
             xdplusx = xdplusx + X( i ) ** 2 / ( D( i ) + inform%multiplier )
         END DO

!  Step 3: Check for convergence
!  ======

  300    CONTINUE
         IF ( ABS( inform%norm_step - delta ) <= tol * delta ) GO TO 900

!  Step 4: Update the estimate of multiplier
!  ======

         dmultiplier = ( ( inform%norm_step - delta ) / delta ) *              &
                     ( inform%norm_step * inform%norm_step / xdplusx )
         IF ( ABS( dmultiplier ) <= tol ) GO TO 900
         inform%multiplier = inform%multiplier + dmultiplier

!  Step 5: Update x and <x,(D+multiplier)^+x>
!  ======

         X = - C / ( D + inform%multiplier )
         inform%norm_step = TWO_NORM( X )
         xdplusx = DOT_PRODUCT( X, X / ( D + inform%multiplier ) )

         IF ( printd ) WRITE( control%out, 2000 )                              &
           inform%multiplier, inform%norm_step, delta, inform%norm_step - delta
         GO TO 300

      ELSE

!  Special case: D = +/-1
!  ============

         IF ( COUNT( D /= one ) == 0 ) THEN

!  Case a: D = I
!  ------

            inform%norm_step = TWO_NORM( C )
            IF ( inform%norm_step <= delta ) THEN
               X = - C
               inform%multiplier = zero
            ELSE
               X = - ( delta / inform%norm_step ) * C
               inform%multiplier = inform%norm_step / delta - one
               inform%norm_step = delta
            END IF
            IF ( printd ) WRITE( control%out, 2000 ) inform%multiplier,        &
                 inform%norm_step, delta, inform%norm_step - delta
            GO TO 900
         ELSE

!  Case b: D = +/-I
!  ------

!  Covered by general case (a special piece of code could be inserted here)

         END IF
      END IF

!  Exit

  900 CONTINUE

!  Compute the value of the model

      f = half * ( DOT_PRODUCT( C, X ) - inform%multiplier * delta * delta )
      RETURN

!  Non-executable statements

 2000 FORMAT( ' multiplier, step, delta, error', 4ES12.4 )
 2010 FORMAT( ' multiplier, step, delta, error', 2( ES12.4, '   infinity ') )

      END SUBROUTINE APS_solve_diagonal_tr

!-*-*-*-*-  A P S _ M O D I F Y _ F A C T O R S   S U B R O U T I N E   -*-*-*

      SUBROUTINE APS_modify_factors( n, deigen, mod1, mod2, rank, data,        &
                                     inform, PERM, EVAL, MOD_EVAL, EVECT, D )

!      .................................................
!      .                                               .
!      .  Solve the trust-region problem               .
!      .                                               .
!      .  minimize <s,g> + 1/2 <s,Hs>                  .
!      .  subject to ||s||_M <= delta,                 .
!      .                                               .
!      .  where M is the modified absolute value of H  .
!      .                                               .
!      .................................................

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!   Based on the Gill-Murray-Ponceleon-Saunders code for modifying the negative
!   eigen-components obtained when factorizing a symmetric indefinite
!   matrix using the Harwell code SILS. (See SOL 90-8, P.19-21)
!
!   Also extracts eigenvalues and eigenvectors, and the modified eigenvalues
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n, rank
      INTEGER, INTENT( OUT ) :: mod1, mod2
      REAL ( KIND = wp ), INTENT( IN ) :: deigen
      INTEGER, INTENT( OUT ), DIMENSION( n ) :: PERM
!     TYPE ( SILS_FACTORS ), INTENT( INOUT ) :: FACTORS
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: EVAL, MOD_EVAL
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: EVECT
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, n ) :: D

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      INTEGER :: i
      REAL ( KIND = wp ) :: alpha, beta, gamma, tau
      REAL ( KIND = wp ) :: t, c , s, e1, e2, eigen
      LOGICAL :: oneby1

!     CALL SILS_ENQUIRE( FACTORS, PIVOTS = PERM, D = D )
      CALL SLS_ENQUIRE( data, inform, PIVOTS = PERM, D = D )
      D( 1, rank + 1 : n ) = zero

!  mod1 and mod2 are the number of negative eigenvalues which arise
!  from small or negative 1x1 and 2x2 block pivots

      mod1 = 0 ; mod2 = 0

!  Loop over all the block pivots

      oneby1 = .TRUE.
      DO i = 1, n

!  Decide if the current block is a 1x1 or 2x2 pivot block

         IF ( oneby1 ) THEN
            IF ( i < n ) THEN
               oneby1 = PERM( i ) > 0
            ELSE
               oneby1 = .TRUE.
            END IF

            alpha = D( 1, i )

!  =========
!  1x1 block
!  =========

            IF ( oneby1 ) THEN

!  Record the eigenvalue

               IF ( alpha /= zero ) THEN
                  eigen = one / alpha
               ELSE
                  eigen = zero
               END IF
               EVAL( i ) = eigen

               IF ( eigen < - deigen ) THEN

!  Negative 1x1 block
!  ------------------

                  mod1 = mod1 + 1
                  D( 1, i ) = - alpha

!  Record the modification

                  MOD_EVAL( i ) = - eigen

                ELSE IF ( eigen < deigen ) THEN

!  Small 1x1 block
!  ---------------

                  mod1 = mod1 + 1
                  D( 1, i ) = one / deigen

!  Record the modification

                  MOD_EVAL( i ) = deigen

               ELSE

!  Positive 1x1 block
!  ------------------
!  Record the modification

                  MOD_EVAL( i ) = eigen

                END IF

!  Record the eigenvector

               EVECT( i ) = one

!  =========
!  2x2 block
!  =========

            ELSE

               beta = D( 2, i )
               gamma = D( 1, i + 1 )

!  2x2 block: (  alpha  beta   ) = (  c  s  ) (  e1     ) (  c  s  )
!             (  beta   gamma  )   (  s -c  ) (     e2  ) (  s -c  )

               IF ( alpha * gamma < beta ** 2 ) THEN
                  tau = ( gamma - alpha ) / ( two * beta )
                  t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) )
                  IF ( tau < zero ) t = - t
                  c = one / SQRT( one + t ** 2 ) ; s = t * c
                  e1 = alpha + beta * t ; e2 = gamma - beta * t

!  Record the first eigenvalue

                  eigen = one / e1
                  EVAL( i ) = eigen

!  Change e1 and e2 to their modified values and then multiply the
!  three 2 * 2 matrices to get the modified alpha, beta and gamma

                  IF ( eigen < - deigen ) THEN

!  Negative first eigenvalue
!  -------------------------

                     mod2 = mod2 + 1

!  Record the modification

                     MOD_EVAL( i ) = - eigen
                     e1 = - e1

                  ELSE IF ( eigen < deigen ) THEN

!  Small first eigenvalue
!  ----------------------

                     mod2 = mod2 + 1

!  Record the modification

                     MOD_EVAL( i ) = deigen
                     e1 = one / deigen

                  ELSE

!  Positive first eigenvalue
!  -------------------------
!  Record the modification

                     MOD_EVAL( i ) = eigen

                   END IF

!  Record the second eigenvalue

                  eigen = one / e2
                  EVAL( i + 1 ) = eigen

                  IF ( eigen < - deigen ) THEN

!  Negative second eigenvalue
!  --------------------------

                     mod2 = mod2 + 1

!  Record the modification

                     MOD_EVAL( i + 1 ) = - eigen
                     e2 = - e2

                  ELSE IF ( eigen < deigen ) THEN

!  Small second eigenvalue
!  -----------------------

                     mod2 = mod2 + 1

!  Record the modification

                     MOD_EVAL( i + 1 ) = deigen
                     e2 = one / deigen

                  ELSE

!  Positive second eigenvalue
!  --------------------------
!  Record its modification

                     MOD_EVAL( i + 1 ) = eigen

                  END IF


!  Record the modified block

                  D( 1, i ) = c ** 2 * e1 + s ** 2 * e2
                  D( 2, i ) = c * s * ( e1 - e2 )
                  D( 1, i + 1 ) = s ** 2 * e1 + c ** 2 * e2

               ELSE

!  Positive 2 by 2 block
!  ---------------------

                  IF ( beta /= zero ) THEN
                     tau = ( gamma - alpha ) / ( two * beta )
                     t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) )
                     IF ( tau < zero ) t = - t
                     c = one / SQRT( one + t ** 2 ) ;  s = t * c
                     e1 = alpha + beta * t ; e2 = gamma - beta * t
                  ELSE
                     c = one ; s = zero
                     e1 = alpha ; e2 = gamma
                  END IF

!  Record the eigenvalue and its modification

                  EVAL( i ) = one / e1
                  MOD_EVAL( i ) = EVAL( i )
                  EVAL( i + 1 ) = one / e2
                  MOD_EVAL( i + 1 ) = one / EVAL( i + 1 )

               END IF

!  Record the eigenvector

               EVECT( i ) = c
               EVECT( i + 1 ) = s

            END IF
         ELSE
            oneby1 = .TRUE.
         END IF
      END DO

!  Register the (possibly modified) diagonal blocks

!     CALL SILS_ALTER_D( FACTORS, D, i )
      CALL SLS_alter_D( data, D, inform )

      RETURN

!  End of subroutine APS_modify_factors

      END SUBROUTINE APS_modify_factors

!  End of module GALAHAD_APS

    END MODULE GALAHAD_APS_DOUBLE
