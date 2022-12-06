! THIS VERSION: GALAHAD 2.6 - 12/03/2014 AT 13:30 GMT.

!-*--*-*-*-*-  L A N C E L O T  -B-  D R C H E   M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   Initial version January 31st 1995
!   update released with GALAHAD Version 2.0. May 11th 2006

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  January 31st 1995

   MODULE LANCELOT_DRCHE_double

     USE LANCELOT_types_double, ONLY : LANCELOT_problem_type

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: DRCHE_save_type, DRCHE_check_element_derivatives

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenp2 = 100.0_wp

!   The DRCHE_save_type derived type

    TYPE :: DRCHE_save_type
      INTEGER :: itest, ielf, j, kk, lg, lh, lend, nelvar, ninvar, ntestl
      LOGICAL :: intre, warning
      REAL ( KIND = wp ) :: epsqrt
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: XT
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: XINT
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: FTUVAL
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: WINV
    END TYPE DRCHE_save_type

      CONTAINS

!-  L A N C E L O T -B- DRCHE_check_element_derivatives S U B R O U T I N E -

     SUBROUTINE DRCHE_check_element_derivatives(                               &
                      prob  , ICALCF, ncalcf, X, FUVALS, lfuval, IELVAR_temp,  &
                      X_temp, nelmax, ninmax, relpr , second, ITESTL, iprint,  &
                      iout  , RANGE , status, S    , ELFUN  )

!  Given a partially separable function, check the analytical gradients
!  (and Hessians if required) against approximations by differences at the
!  given point X

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_problem_type ), INTENT( INOUT ) :: prob
     INTEGER, INTENT( IN ) :: lfuval, nelmax, ninmax, iprint, iout
     INTEGER, INTENT( INOUT ) :: status, ncalcf
     REAL ( KIND = wp ), INTENT( IN ) :: relpr
     LOGICAL, INTENT( IN ) :: second
     INTEGER, INTENT( IN ), DIMENSION( prob%nel     ) :: ITESTL
     INTEGER, INTENT( OUT ),                                                   &
              DIMENSION( prob%ISTAEV( prob%nel + 1 ) - 1 ) :: IELVAR_temp
     INTEGER, INTENT( INOUT ), DIMENSION( prob%nel     ) :: ICALCF
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( prob%n  ) :: X
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nelmax ) :: X_temp
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( lfuval ) :: FUVALS
     TYPE ( DRCHE_save_type ), INTENT( INOUT ) :: S

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

!  Interface block for RANGE

       SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,       &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN  ), DIMENSION ( lw1 ) :: W1
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE

!  Interface block for ELFUN 

       SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, ifflag, ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
       INTEGER, INTENT( IN ) :: INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
       INTEGER, INTENT( IN ) :: ICALCF(LCALCF)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(LXVALU)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(LEPVLU)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(LFUVAL)
       END SUBROUTINE ELFUN 

     END INTERFACE

!-----------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!-----------------------------------------------------

     OPTIONAL :: ELFUN 

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: internal_el

     internal_el = PRESENT( ELFUN  )

! Initial entry: set up data

     IF ( status == 0 ) THEN 

!  If the element functions are to be evaluated internally, check that
!  the user has supplied appropriate information

       IF ( internal_el ) THEN
         IF ( ALLOCATED( prob%ISTEPA ) .AND. ALLOCATED( prob%EPVALU ) ) THEN
           IF ( SIZE( prob%ISTEPA ) < prob%nel + 1 ) THEN
             status = 10 ; RETURN ; END IF
           IF ( SIZE( prob%EPVALU ) < prob%ISTEPA( prob%nel + 1 ) - 1) THEN
             status = 10 ; RETURN ; END IF
         ELSE
           status = 10 ; RETURN
         END IF
       END IF
     END IF

!  ========================================================
!  Call the main subroutine to perform the bulk of the work
!  ========================================================

!  Internal element evaluations will be performed
!  ----------------------------------------------

     IF ( internal_el ) THEN
       CALL DRCHE_check_main( prob%n, prob%nel, lfuval,                        &
                              prob%ISTAEV, prob%ISTADH,                        &
                              prob%IELVAR, prob%INTVAR, prob%ITYPEE,           &
                              prob%INTREP, ICALCF, ncalcf, X, FUVALS,          &
                              IELVAR_temp, X_temp, nelmax, ninmax, relpr,      &
                              second, ITESTL, iprint, iout, RANGE , status, S, &
                              ELFUN  = ELFUN , ISTEPA = prob%ISTEPA,           &
                              EPVALU = prob%EPVALU )

!  Element evaluations will be performed via reverse communication
!  ---------------------------------------------------------------

     ELSE
       CALL DRCHE_check_main( prob%n, prob%nel, lfuval,                        &
                              prob%ISTAEV, prob%ISTADH,                        &
                              prob%IELVAR, prob%INTVAR, prob%ITYPEE,           &
                              prob%INTREP, ICALCF, ncalcf, X, FUVALS,          &
                              IELVAR_temp, X_temp, nelmax, ninmax, relpr,      &
                              second, ITESTL, iprint, iout, RANGE , status, S )
     END IF

     RETURN

!  End of subroutine DRCHE_check_element_derivatives

     END SUBROUTINE DRCHE_check_element_derivatives

!-*-*-*-*-  L A N C E L O T -B- DRCHE_check_main S U B R O U T I N E -*-*-*-*-*-

     SUBROUTINE DRCHE_check_main(                                              &
                      n, nel, lfuval,                              &
                      ISTAEV, ISTADH, IELVAR, INTVAR, ITYPEE, INTREP,          &
                      ICALCF, ncalcf, X     , FUVALS, IELVAR_temp, X_temp,     &
                      nelmax, ninmax, relpr , second,                          &
                      ITESTL, iprint, iout  , RANGE , status, S,               &
                      ELFUN , ISTEPA, EPVALU )                     

!  Given a partially separable function, check the analytical gradients
!  (and Hessians if required) against approximations by differences at the
!  given point X

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nel, lfuval
     INTEGER, INTENT( IN ) :: nelmax, ninmax, iprint, iout
     INTEGER, INTENT( INOUT ) :: status, ncalcf
     REAL ( KIND = wp ), INTENT( IN ) :: relpr
     LOGICAL, INTENT( IN ) :: second
     INTEGER, INTENT( IN ), DIMENSION( nel     ) :: ITESTL
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTAEV
     INTEGER, INTENT( IN ), DIMENSION( ISTAEV( nel + 1 ) - 1 ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     INTEGER, INTENT( OUT ),                                                   &
              DIMENSION( ISTAEV( nel + 1 ) - 1 ) :: IELVAR_temp
     INTEGER, INTENT( INOUT ), DIMENSION( nel     ) :: ICALCF
     INTEGER, INTENT( INOUT ), DIMENSION( nel + 1 ) :: ISTADH, INTVAR
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n  ) :: X
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nelmax ) :: X_temp
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( lfuval ) :: FUVALS
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     TYPE ( DRCHE_save_type ), INTENT( INOUT ) :: S

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

!  Interface block for RANGE

       SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                          lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN  ), DIMENSION ( lw1 ) :: W1
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE

!  Interface block for ELFUN 

       SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, ifflag, ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
       INTEGER, INTENT( IN ) :: INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
       INTEGER, INTENT( IN ) :: ICALCF(LCALCF)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(LXVALU)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(LEPVLU)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(LFUVAL)
       END SUBROUTINE ELFUN 

     END INTERFACE

!-----------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!-----------------------------------------------------

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( nel + 1 ) :: ISTEPA
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( : ) :: EPVALU
     OPTIONAL :: ELFUN 

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i , ii, ifstat, inform, ip1, l, k, lfxi, lgxi, lhxi
     INTEGER :: nel1, nsizeh
     REAL ( KIND = wp ) :: comp, gtol, temp
     LOGICAL :: external_el

     external_el = .NOT. PRESENT( ELFUN  )

!  Branch to the interior of the code if a re-entry is being made

     SELECT CASE ( status )
     CASE ( - 1 ) ; GO TO 100
     CASE ( - 2 ) ; GO TO 240
     CASE ( - 3 ) ; GO TO 250
     END SELECT
     
     IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2000 )
     S%warning = .FALSE.
     S%epsqrt = SQRT( relpr )

!  Set the starting addresses for the partitions within FUVALS

     lfxi = 0 ; lgxi = lfxi + nel

!  Set up the starting addresses for the element gradients with respect to
!  their internal variables

     nel1 = nel + 1
     INTVAR( nel1 ) = 0
     k = INTVAR( 1 )
     INTVAR( 1 ) = lgxi + 1
     DO i = 1, nel
       ip1 = i + 1
       l = INTVAR( ip1 )
       INTVAR( ip1 ) = INTVAR( i ) + k
       k = l
     END DO

!  Ensure that all the element functions are evaluated at the initial point

     S%ntestl = ncalcf
     ICALCF( : ncalcf ) = ITESTL( : ncalcf )
     lhxi = INTVAR( nel1 ) - 1
     S%ninvar = INTVAR( nel1 ) - INTVAR( 1 )

!  Set up the starting addresses for the element Hessians with respect to
!  their internal variables

     k = lhxi + 1
     IF ( second ) THEN
       DO i = 1, nel
         ISTADH( i ) = k
         nsizeh = INTVAR( i + 1 ) - INTVAR( i )
         k = k + nsizeh * ( nsizeh + 1 ) / 2
       END DO
     END IF
     S%lend = k - 1

!  Allocate workspace arrays

     ALLOCATE( S%XT( nelmax ) ) ; ALLOCATE( S%XINT( ninmax ) )
     ALLOCATE( S%FTUVAL( S%lend ) ) ; ALLOCATE( S%WINV( ninmax, nelmax ) )

!  Initialize FUVALS as zero

     FUVALS = zero

!  If necessary, return to the calling program to obtain the element function
!  and derivative values at the initial point

     status = - 1
     IF ( external_el ) RETURN
     CALL ELFUN ( FUVALS, X     , EPVALU, ncalcf, ITYPEE, ISTAEV,              &
                  IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel, nel + 1,        &
                  ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1, nel + 1, nel,       &
                  lfuval, n, ISTEPA( nel + 1 ) - 1, 1, ifstat )
     IF ( second ) THEN
       CALL ELFUN ( FUVALS, X     , EPVALU, ncalcf, ITYPEE, ISTAEV,            &
                    IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel, nel + 1,      &
                    ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1, nel + 1, nel,     &
                    lfuval, n, ISTEPA( nel + 1 ) - 1, 3, ifstat )
     ELSE
       CALL ELFUN ( FUVALS, X     , EPVALU, ncalcf, ITYPEE, ISTAEV,            &
                    IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel, nel + 1,      &
                    ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1, nel + 1, nel,     &
                    lfuval, n, ISTEPA( nel + 1 ) - 1, 2, ifstat )
     END IF

!  Copy FUVALS into FTUVAL

 100 CONTINUE
     S%FTUVAL = FUVALS( : S%lend )

!  Check the analytical gradient ( in internal representation )

     status = - 2

!  Mock do loop to allow reverse communication

     S%itest = 0 ; ncalcf = 1

 200 CONTINUE
     S%itest = S%itest + 1
     IF ( S%itest > S%ntestl ) GO TO 290
     S%ielf = ITESTL( S%itest )
     S%ninvar = INTVAR( S%ielf + 1 ) - INTVAR( S%ielf )
     S%nelvar = ISTAEV( S%ielf + 1 ) - ISTAEV( S%ielf )
     S%kk = INTVAR( S%ielf )
     S%intre = INTREP( S%ielf )
     ICALCF( 1 ) = S%ielf

!  Assign temporary indices to the variables. Store the elemental variables
!  in X_temp

     ii = ISTAEV( S%ielf )
     DO i = 1, S%nelvar
       IELVAR_temp( ii ) = i
       X_temp( i ) = X( IELVAR( ii ) )
       ii = ii + 1
     END DO
     IF ( S%intre ) THEN

!  Compute the values of the internal variables

       CALL RANGE ( S%ielf, .FALSE., X_temp, S%XINT, S%nelvar, S%ninvar,       &
                    ITYPEE( S%ielf ), S%nelvar, S%ninvar )
       CALL DRCHE_generalized_inverse( S%ielf, S%WINV , ninmax, nel, S%nelvar, &
                                       S%ninvar, inform, iout, ITYPEE, RANGE )
       IF ( inform == 1 ) THEN
         IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2080 ) S%ielf
         GO TO 280
       END IF
     END IF
     IF ( second ) S%lh = ISTADH( S%ielf )

!  Mock do loop to allow reverse communication

     S%j = 0

 220 CONTINUE
     S%j = S%j + 1
     IF ( S%j > S%ninvar ) GO TO 280

!  Check the K-th component of the IELF-th gradient

     IF ( S%intre ) THEN
       S%XINT( S%j ) = S%XINT( S%j ) + S%epsqrt

!  Put the elemental variables in X_temp

       S%XT( : S%nelvar ) = X_temp( : S%nelvar )
       DO i = 1, S%nelvar
!        X_temp( i ) = DOT_PRODUCT( S%WINV( : S%ninvar, i ),                   &
!                                   S%XINT( : S%ninvar ) )
         temp = zero
         DO ii = 1, S%ninvar
           temp = temp + S%WINV( ii, i ) * S%XINT( ii )
         END DO
         X_temp( i ) = temp
       END DO
     ELSE
       S%XT( 1 ) = X_temp( S%j )
       X_temp( S%j ) = X_temp( S%j ) + S%epsqrt
     END IF

!  Evaluate the IELF-th element function at the perturbed point

     status = - 2
     IF ( external_el ) RETURN
     CALL ELFUN ( FUVALS, X_temp, EPVALU, ncalcf, ITYPEE, ISTAEV,              &
                  IELVAR_temp, INTVAR, ISTADH, ISTEPA, ICALCF, nel, nel + 1,   &
                  ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1, nel + 1, nel,       &
                  lfuval, nelmax, ISTEPA( nel + 1 ) - 1, 1, ifstat )

!  Estimate the K-th component of the gradient and test it w.r.t. its
!  analytical value

 240 CONTINUE
     comp = ( FUVALS( S%ielf ) - S%FTUVAL( S%ielf ) ) / S%epsqrt

!  Print the components for comparison

!    IF ( iout > 0 .AND. iprint > 0 )                                          &
!      WRITE( iout, 2020 ) S%j, S%FTUVAL( S%kk ), S%j, comp

!  Test agreement between analytical and approx. values

     gtol = tenp2 * S%epsqrt * MAX( one, ABS( S%FTUVAL( S%kk ) ) )
     IF ( iprint >= 10 .OR. ABS( S%FTUVAL( S%kk ) - comp ) > gtol ) THEN
       IF ( ABS( S%FTUVAL( S%kk ) - comp ) > gtol ) THEN
         S%warning = .TRUE.
         IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2050 )
       END IF
       IF ( iout > 0 .AND. iprint > 0 ) THEN
         WRITE( iout, 2010 ) S%ielf
         WRITE( iout, 2020 ) S%j, S%FTUVAL( S%kk ), S%j, comp
       END IF
     END IF
     S%kk = S%kk + 1

!  Check the analytical Hessian ( in internal representation )

     IF ( second ) THEN

!  Compute the IELF-th perturbed gradient in internal representation

        status = - 3
        IF ( external_el ) RETURN
        CALL ELFUN ( FUVALS, X_temp, EPVALU, ncalcf, ITYPEE, ISTAEV,           &
                     IELVAR_temp, INTVAR, ISTADH, ISTEPA, ICALCF, nel, nel + 1,&
                     ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1, nel + 1, nel,    &
                     lfuval, nelmax, ISTEPA( nel + 1 ) - 1, 2, ifstat )
     END IF

 250 CONTINUE
     IF ( second ) THEN

!  Estimate each component of the Hessian's column and test it w.r.t. its
!  analytical value

       S%lg = INTVAR( S%ielf )
       DO k = 1, S%j
         comp = ( FUVALS( S%lg ) - S%FTUVAL( S%lg ) ) / S%epsqrt

!  Print the components for comparison

!        IF ( iout > 0 .AND. iprint > 0 )                                    &
!          WRITE( IOUT, 2040 ) k, S%j, S%FTUVAL( S%lh ), k, S%j, comp

!  Test agreement between analytical and approximate values

         gtol = tenp2 * S%epsqrt * MAX( one, ABS( S%FTUVAL( S%lh ) ) )
         IF ( ABS( S%FTUVAL( S%lh ) - comp ) > gtol .OR. iprint >= 10 ) THEN
           IF ( ABS( S%FTUVAL( S%lh ) - comp ) > gtol ) THEN
             S%warning = .TRUE.
             IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2060 )
           END IF
           IF ( iout > 0 .AND. iprint > 0 ) THEN
             WRITE( iout, 2030 ) S%ielf
             WRITE( iout, 2040 ) k, S%j, S%FTUVAL( S%lh ), k, S%j, comp
           END IF
         END IF
         S%lg = S%lg + 1 ; S%lh = S%lh + 1
       END DO
     END IF

!  Reset the point X

     IF ( S%intre ) THEN
       S%XINT( S%j ) = S%XINT( S%j ) - S%epsqrt
       X_temp( : S%nelvar ) = S%XT( : S%nelvar )
     ELSE
       X_temp( S%j ) = S%XT( 1 )
     END IF
     GO TO 220

 280 CONTINUE
     GO TO 200

!  Prepare to exit. Reset the pointers to number of internal variables

 290 CONTINUE
     DO i = 1, nel
       INTVAR( i ) = INTVAR( i + 1 ) - INTVAR( i )
     END DO

!  Reset FUVALS to FTUVAL

     FUVALS( : S%lend ) = S%FTUVAL
     IF ( S%warning ) THEN
       status = 1
       IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2090 )
     ELSE
       status = 0
       IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2070 )
     END IF

!  Deallocate workspace arrays

     IF ( ALLOCATED( S%XT ) ) DEALLOCATE( S%XT )
     IF ( ALLOCATED( S%XINT ) ) DEALLOCATE( S%XINT )
     IF ( ALLOCATED( S%FTUVAL ) ) DEALLOCATE( S%FTUVAL )
     IF ( ALLOCATED( S%WINV ) ) DEALLOCATE( S%WINV )

     RETURN

!  Non-executable statements

 2000  FORMAT( /, ' ********* Checking element derivatives **********',        &
               /, ' *                                               *', / )
 2010  FORMAT( ' Gradient of element function ', I4, ' :', /, 42( '-' ) )
 2020  FORMAT( ' Anal. grad.( ', I4, ' )      = ', ES14.6,                     &
               ' Approx.( ', I4, '    )      = ', ES14.6, / )
 2030  FORMAT( ' Hessian of element function ', I4, ' :', /, 41( '-' ) )
 2040  FORMAT( ' Anal. hess.( ', I4, ',', I4, ' ) = ', ES14.6,                 &
               ' Appr. hess.( ', I4, ',', I4, ' ) = ', ES14.6, / )
 2050  FORMAT( ' Possible mistake in the computation of the gradient ' )
 2060  FORMAT( ' Possible mistake in the computation of the Hessian  ' )
 2070  FORMAT( ' *                                               * ', /,       &
               ' *********** Derivatives checked O.k. ************ ' )
 2080  FORMAT( ' The range transformation for element ', I5,                   &
               ' is',' singular ' )
 2090  FORMAT( ' *                                               * ', /,       &
               ' ******** Derivatives checked - warnings ********* ' )

!  End of subroutine DRCHE_check_main

     END SUBROUTINE DRCHE_check_main

!-*-  L A N C E L O T  -B- DRCHE_generalized_inverse S U B R O U T I N E -*-*

     SUBROUTINE DRCHE_generalized_inverse(                                     &
         ielf, A, la, nel, nelvar, ninvar, inform, iout, ITYPEE, RANGE )

!  To find the inverse transformation to the "gather" used by RANGE.
!  Form the (Moore-Penrose) generalized inverse of A using the
!  storage-miserly method of Powell (AERE report R-6072)

!  Transform A to lower triangular form by a sequence of elementary
!  transformations with row and column interchanges for stability

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN  ) :: ielf, la, nel, nelvar, ninvar, iout
     INTEGER, INTENT( OUT ) :: inform
     INTEGER, INTENT( IN ), DIMENSION ( nel ) :: ITYPEE
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( la, nelvar ) :: A

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,       &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN  ), DIMENSION ( lw1 ) :: W1
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i , ir, j , k , kp1, ii, l
     REAL ( KIND = wp ) :: sizea, bsq, rmax, sigma , temp, epsmch

!-----------------------------------------------
!   A u t o m a t i c  A r r a y s
!-----------------------------------------------

     INTEGER, DIMENSION( ninvar + nelvar ) :: IW_drche
     REAL ( KIND = wp ), DIMENSION( ninvar ) :: W_IN

     epsmch = EPSILON( one )

!  Form A

     W_IN = zero
     DO i = 1, ninvar
       W_IN( i ) = one
       CALL RANGE ( ielf, .TRUE., W_IN, A( i, : nelvar ), nelvar, ninvar,      &
                    ITYPEE( ielf ), ninvar, nelvar )
!      sizea = SQRT( SUM( A( i, : nelvar ) ** 2 ) )
       sizea = zero ; DO j = 1, nelvar ; sizea = sizea + A( i, j ) ** 2 ; END DO
       sizea = SQRT( sizea )
       IF ( sizea < epsmch ) THEN
         inform = 1 ; RETURN
       END IF
       W_IN( i ) = zero
     END DO

!  Record all row and column interchanges in IW_drche

     DO i = 1, ninvar
       IW_drche( i ) = i
     END DO

     DO i = 1, nelvar
       IW_drche( ninvar + i ) = i
     END DO

!  Perform the K-th elementary transformation

     DO k = 1, ninvar
       kp1 = k + 1

!  Find the largest row

       rmax = zero
       DO i = k, ninvar
         temp = zero ; DO j = k, nelvar ; temp = temp + A( i, j ) ** 2 ; END DO
         IF ( temp > rmax ) THEN
           ir = i
           rmax = temp
         END IF
       END DO

!  If the matrix is not of full rank, stop

       IF ( rmax == zero ) THEN
         IF ( iout > 0 ) WRITE( iout, 2000 ) ninvar - k
         STOP
       END IF

!  If the current row is not the largest, swop it with the largest

       IF ( ir > k ) THEN
         l = IW_drche( k )
         IW_drche( k ) = IW_drche( ir )
         IW_drche( ir ) = l
         DO j = 1, nelvar
           temp = A( k, j )
           A( k, j ) = A( ir, j )
           A( ir, j ) = temp
         END DO
       END IF

!  Find largest element in the pivotal row

       rmax = zero
       temp = zero
       DO j = k, nelvar
         temp = temp + A( k, j ) ** 2
         IF ( rmax < ABS( A( k, j ) ) ) THEN
           ir = j
           rmax = ABS( A( k, j ) )
         END IF
       END DO

!  If the current column is not the largest, swap it with the largest

       IF ( ir > k ) THEN
         i = ninvar + k
         j = ninvar + ir
         l = IW_drche( i )
         IW_drche( i ) = IW_drche( j )
         IW_drche( j ) = l
         DO i = 1, ninvar
           rmax = A( i, k )
           A( i, k ) = A( i, ir )
           A( i, ir ) = rmax
         END DO
       END IF

!  Replace the pivotal row by the Housholder transformation vector

       sigma = SQRT( temp )
       bsq = SQRT( temp + sigma * ABS( A( k, k ) ) )
       W_IN( k ) = SIGN( sigma + ABS( A( k, k ) ), A( k, k ) ) / bsq
       A( k, k ) = - SIGN( sigma, A( k, k ) )
       IF ( kp1 <= nelvar ) THEN
         A( k, kp1 : nelvar ) = A( k, kp1 : nelvar ) / bsq

!  Apply the transformation to the remaining rows of A

         DO i = kp1, ninvar
!          temp = W_IN( k ) * A( i, k ) +                                      &
!            DOT_PRODUCT( A( k, kp1: nelvar ), A( i, kp1: nelvar ) )
           temp = W_IN( k ) * A( i, k )
           DO j = kp1, nelvar
              temp = temp + A( k, j ) * A( i, j )
           END DO
           A( i, k ) = A( i, k ) - temp * W_IN( k )
           A( i, kp1 : nelvar ) =                                              &
             A( i, kp1 : nelvar ) - temp * A( k, kp1 : nelvar )
         END DO
       END IF
     END DO

!  The reduction of A is complete. Build the generalized inverse. Firstly,
!  apply the first elementary transformation

     temp = W_IN( ninvar ) / A( ninvar, ninvar )
     A( ninvar, ninvar + 1 : nelvar )                                          &
       = - temp * A( ninvar, ninvar + 1 : nelvar )
     A( ninvar, ninvar )                                                       &
       = one / A( ninvar, ninvar ) - temp * W_IN( ninvar )

!  Now apply the remaining NINVAR - 1 transformations

     DO k = ninvar - 1, 1, - 1
       kp1 = k + 1

!  First transform the last NINVAR - K rows

       DO i = kp1, ninvar
!        temp = DOT_PRODUCT( A( k, kp1 : nelvar ), A( i, kp1 : nelvar ) )
         temp = zero
         DO ii = kp1, nelvar
           temp = temp + A( k, ii ) * A( i, ii )
         END DO
         A( i, kp1 : nelvar ) =                                                &
           A( i, kp1 : nelvar ) - temp * A( k, kp1 : nelvar )
         W_IN( i ) = - temp * W_IN( k )
       END DO

!  Then calculate the new K-th row

       DO j = kp1, nelvar
!        temp = - W_IN( k ) * A( k, j ) -                                      &
!          DOT_PRODUCT( A( kp1 : ninvar, k ), A( kp1 : ninvar, j ) )
         temp = - W_IN( k ) * A( k, j ) 
         DO ii = kp1, ninvar
            temp = temp - A( ii, k ) * A( ii, j )
         END DO
         A( k, j ) = temp / A( k, k )
       END DO

!  Update the K-th column

!      temp = one - W_IN( k ) ** 2 -                                           &
!        DOT_PRODUCT( A( kp1: ninvar , k ), W_IN( kp1: ninvar ) )
       temp = one - W_IN( k ) ** 2
       DO ii = kp1, ninvar
          temp = temp - A( ii, k ) * W_IN( ii )
       END DO
       A( kp1 : ninvar, k ) = W_IN( kp1 : ninvar )
       A( k, k ) = temp / A( k, k )
     END DO

!  Undo the row interchanges

     DO i = 1, ninvar
 410   CONTINUE
       ir = IW_drche( i )
       IF ( i < ir ) THEN
         IW_drche( i ) = IW_drche( ir )
         IW_drche( ir ) = ir

!  Swap rows I and IR

         DO j = 1, nelvar
           temp = A( i, j )
           A( i, j ) = A( ir, j )
           A( ir, j ) = temp
         END DO
         GO TO 410
       END IF
     END DO

!  Undo the column interchanges

     DO j = 1, nelvar
 440   CONTINUE
       i = ninvar + j
       ir = IW_drche( i )
       IF ( j < ir ) THEN
         k = ninvar + ir
         IW_drche( i ) = IW_drche( k )
         IW_drche( k ) = ir

!  Swap columns J and IR

         DO i = 1, ninvar
           temp = A( i, j )
           A( i, j ) = A( i, ir )
           A( i, ir ) = temp
         END DO
         GO TO 440
       END IF
     END DO

    inform = 0
    RETURN

!  Non-executable statements

 2000  FORMAT( /, ' *** Error message from DRCHE_generalized_inverse *** ',   &
               I3, ' reduced rows found to be zero ' )

!  End of subroutine DRCHE_generalized_inverse

     END SUBROUTINE DRCHE_generalized_inverse

!  End of module LANCELOT_DRCHE

   END MODULE LANCELOT_DRCHE_double




