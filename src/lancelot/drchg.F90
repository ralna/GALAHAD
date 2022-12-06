! THIS VERSION: GALAHAD 2.6 - 12/03/2014 AT 13:30 GMT.

!-*-*-*-*-*-  L A N C E L O T  -B-  D R C H G   M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   February 6th 1995 as runlanb
!   March 14th 2003 as uselanb
!   update released with GALAHAD Version 2.0. May 11th 2006

   MODULE LANCELOT_DRCHG_double

     USE LANCELOT_types_double, ONLY: LANCELOT_problem_type

     IMPLICIT NONE

     PRIVATE
     PUBLIC ::  DRCHG_save_type, DRCHG_check_group_derivatives

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenp2 = 100.0_wp

!   The DRCHG_save_type derived type

     TYPE :: DRCHG_save_type
      REAL ( KIND = wp ) :: epsqrt
      LOGICAL :: warning
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: XT
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: GTVALS
     END TYPE DRCHG_save_type

   CONTAINS

!-  L A N C E L O T -B- DRCHG_check_group_derivatives  S U B R O U T I N E -

     SUBROUTINE DRCHG_check_group_derivatives(                                 &
                      prob, X, GVALS , ITESTG, ntestg, relpr, iprint, iout,    &
                      status, S, GROUP )

!  Given a vector of functionals GVALS, check their analytical derivatives
!  against approximations by differences at the given point X

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_problem_type ), INTENT( INOUT ) :: prob
     INTEGER, INTENT( IN    ) :: ntestg, iprint, iout
     INTEGER, INTENT( INOUT ) :: status
     REAL ( KIND = wp ), INTENT( IN ) :: relpr
     INTEGER, INTENT( IN    ), DIMENSION( prob%ng ) :: ITESTG
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( prob%ng ) :: X
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( prob%ng, 3 ) :: GVALS
     TYPE ( DRCHG_save_type ), INTENT( INOUT ) :: S

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

!  Interface block for GROUP

       SUBROUTINE GROUP ( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg,              &
                          ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa,              &
                          lcalcg, lfvalu, lgpvlu, derivs, igstat )
       INTEGER, INTENT( IN ) :: lgvalu, ncalcg
       INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
       INTEGER, INTENT( OUT ) :: igstat
       LOGICAL, INTENT( IN ) :: derivs
       INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
       INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
       INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lfvalu ) :: FVALUE
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lgpvlu ) :: GPVALU
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ),                        &
                                       DIMENSION ( lgvalu, 3 ) :: GVALUE
       END SUBROUTINE GROUP

     END INTERFACE

!-----------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!-----------------------------------------------------

     OPTIONAL :: GROUP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: internal_gr

     internal_gr = PRESENT( GROUP )

! Initial entry: check data

     IF ( status == 0 ) THEN

!  If the group functions are to be evaluated internally, check that
!  the user has supplied appropriate information

       IF ( internal_gr ) THEN
         IF ( ALLOCATED( prob%ISTGPA ) .AND. ALLOCATED( prob%ITYPEG ) .AND.  &
              ALLOCATED( prob%GPVALU ) ) THEN
           IF ( SIZE( prob%ISTGPA ) < prob%ng + 1 .OR.                         &
                SIZE( prob%ITYPEG ) < prob%ng ) THEN
             status = 11 ; RETURN ; END IF
           IF ( SIZE( prob%GPVALU ) < prob%ISTGPA( prob%ng + 1 ) - 1 ) THEN
             status = 11 ; RETURN ; END IF
         ELSE
           status = 11 ; RETURN
         END IF
       END IF
     END IF

!  ========================================================
!  Call the main subroutine to perform the bulk of the work
!  ========================================================

!  Internal group evaluations will be performed
!  --------------------------------------------

     IF ( internal_gr ) THEN
       CALL DRCHG_check_main( prob%ng, X, GVALS , ITESTG, ntestg, relpr,       &
                              iprint, iout  , status, S,                       &
                              GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
                              ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )

!  Element and group evaluations will be performed via reverse communication
!  -------------------------------------------------------------------------

     ELSE
       CALL DRCHG_check_main( prob%ng, X, GVALS , ITESTG, ntestg, relpr,       &
                              iprint, iout, status, S )
     END IF

     RETURN

!  End of subroutine DRCHG_check_group_derivatives

     END SUBROUTINE DRCHG_check_group_derivatives

!-*-*-*-*-  L A N C E L O T -B- DRCHG_check_main  S U B R O U T I N E -*-*-*-*-

     SUBROUTINE DRCHG_check_main(                                              &
                      ng, X , GVALS , ITESTG, ntestg, relpr, iprint, iout,     &
                      status, S, GROUP , ISTGPA, ITYPEG, GPVALU )

!  Given a vector of functionals GVALS, check their analytical derivatives
!  against approximations by differences at the given point X

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN    ) :: ng    , ntestg, iprint, iout
     INTEGER, INTENT( INOUT ) :: status
     REAL ( KIND = wp ), INTENT( IN ) :: relpr
     INTEGER, INTENT( IN    ), DIMENSION( ng ) :: ITESTG
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ng ) :: X
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ng, 3 ) :: GVALS
     TYPE ( DRCHG_save_type ), INTENT( INOUT ) :: S

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

!  Interface block for GROUP

       SUBROUTINE GROUP ( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg,              &
                          ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa,              &
                          lcalcg, lfvalu, lgpvlu, derivs, igstat )
       INTEGER, INTENT( IN ) :: lgvalu, ncalcg
       INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
       INTEGER, INTENT( OUT ) :: igstat
       LOGICAL, INTENT( IN ) :: derivs
       INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
       INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
       INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lfvalu ) :: FVALUE
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lgpvlu ) :: GPVALU
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ),                        &
                                       DIMENSION ( lgvalu, 3 ) :: GVALUE
       END SUBROUTINE GROUP

     END INTERFACE

!-----------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!-----------------------------------------------------

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng + 1 ) :: ISTGPA
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: ITYPEG
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( : ) :: GPVALU
     OPTIONAL :: GROUP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, igstat
     REAL ( KIND = wp ) :: comp, gtol
     LOGICAL :: external_gr

     external_gr = .NOT. PRESENT( GROUP )

!  Branch to the interior of the code if a re-entry is being made

     SELECT CASE ( status )
     CASE ( - 1 ) ; GO TO 100
     CASE ( - 2 ) ; GO TO 200
     CASE ( - 3 ) ; GO TO 300
     END SELECT

     IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2000 )
     S%warning = .FALSE.
     S%epsqrt = SQRT( relpr )

!  Allocate XT and GTVALS

     ALLOCATE( S%XT( ng ) ) ; ALLOCATE( S%GTVALS( ng , 3 ) )

!  Obtain the values of the functions and their derivatives at the point X

     status = - 1
     IF ( external_gr ) RETURN
     CALL GROUP ( GVALS, ng, X, GPVALU, ntestg, ITYPEG, ISTGPA, ITESTG,        &
                  ng, ng + 1, ng, ng, ISTGPA( ng + 1 ) - 1, .FALSE., igstat )
     CALL GROUP ( GVALS, ng, X, GPVALU, ntestg, ITYPEG, ISTGPA, ITESTG,        &
                  ng, ng + 1, ng, ng, ISTGPA( ng + 1 ) - 1,  .TRUE., igstat )
 100 CONTINUE

!  Copy the components of GVALS for GROUP  that are to be tested into GTVALS

     S%GTVALS( ITESTG( : ntestg ), : ) = GVALS( ITESTG( : ntestg ), : )

!  Copy the components of X for groups that are to be tested into XT and
!  perturb X by a small quantity

     S%XT( ITESTG( : ntestg ) ) = X( ITESTG( : ntestg ) )
     X( ITESTG( : ntestg ) ) = X( ITESTG( : ntestg ) ) + S%epsqrt

!  Evaluate the required groups at the perturbed point

     status = - 2
     IF ( external_gr ) RETURN
     CALL GROUP ( GVALS, ng, X, GPVALU, ntestg, ITYPEG, ISTGPA, ITESTG,        &
                  ng, ng + 1, ng, ng, ISTGPA( ng + 1 ) - 1, .FALSE., igstat )
 200 CONTINUE

!  Estimate the first derivative of the I-th group and test it w.r.t. its
!  analytical value

     DO j = 1, ntestg
       i = ITESTG( j )
       comp = ( GVALS( i, 1 ) - S%GTVALS( i, 1 ) ) / S%epsqrt

!  Test agreement between analytical and approxroximate values

       gtol = tenp2 * S%epsqrt * MAX( one, ABS( S%GTVALS( i, 2 ) ) )
       IF ( iprint >= 10 .OR. ABS( S%GTVALS( i, 2 ) - comp ) > gtol ) THEN
         IF ( ABS( S%GTVALS( i, 2 ) - comp ) > gtol ) THEN
           S%warning = .TRUE.
           IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2040 )
         END IF
         IF ( iout > 0 .AND. iprint > 0 ) THEN
           WRITE( iout, 2010 ) i
           WRITE( iout, 2020 ) S%GTVALS( i, 2 ), comp
         END IF
       END IF
     END DO

!  Evaluate the required group derivatives at the perturbed point

     status = - 3
     IF ( external_gr ) RETURN
     CALL GROUP ( GVALS, ng, X, GPVALU, ntestg, ITYPEG, ISTGPA, ITESTG,        &
                  ng, ng + 1, ng, ng, ISTGPA( ng + 1 ) - 1, .TRUE., igstat )
 300 CONTINUE

!  Estimate the second derivative of the I-th group and test it w.r.t. its
!  analytical value

     DO j = 1, ntestg
       i = ITESTG( j )
       comp = ( GVALS( i, 2 ) - S%GTVALS( i, 2 ) ) / S%epsqrt

!  Test agreement between analytical and approximate values

       gtol = tenp2 * S%epsqrt * MAX( one, ABS( S%GTVALS( i, 3 ) ) )
       IF ( iprint >= 10 .OR. ABS( S%GTVALS( i, 3 ) - comp ) > gtol ) THEN
         IF ( ABS( S%GTVALS( i, 3 ) - comp ) > gtol ) THEN
           S%warning = .TRUE.
           IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2050 )
         END IF
         IF ( iout > 0 .AND. iprint > 0 ) THEN
           WRITE( iout, 2030 ) i
           WRITE( iout, 2020 ) S%GTVALS( i,3 ), comp
         END IF
       END IF
     END DO

!  Reset the components of GVALS and X for groups that have been tested

     GVALS( ITESTG( : ntestg ), : ) = S%GTVALS( ITESTG( : ntestg ), : )
     X( ITESTG( : ntestg ) ) = S%XT( ITESTG( : ntestg ) )

     IF ( S%warning ) THEN
       status = 1
       IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2060 )
     ELSE
       status = 0
       IF ( iout > 0 .AND. iprint > 0 ) WRITE( iout, 2070 )
     END IF

!  Deallocate XT and GTVALS

     IF ( ALLOCATED( S%XT ) ) DEALLOCATE( S%XT )
     IF ( ALLOCATED( S%GTVALS ) ) DEALLOCATE( S%GTVALS )

     RETURN

!  Non-executable statements

 2000  FORMAT( /, ' *********** Checking group derivatives **********',        &
               /, ' *                                               *', / )
 2010  FORMAT( ' 1st derivative of group function ', I4, ' :',                 &
               /, 42( '-' ) )
 2020  FORMAT( ' Anal. deriv. = ', ES14.6, ' Approx. deriv. = ', ES14.6, / )
 2030  FORMAT( ' 2nd derivative of group function ', I4, ' :',                 &
               /, 42( '-' ) )
 2040  FORMAT( ' Possible mistake in the computation of the 1st',              &
               ' derivative' )
 2050  FORMAT( ' Possible mistake in the computation of the 2nd',              &
               ' derivative' )
 2060  FORMAT( ' *                                               * ', /,       &
               ' *********** Derivatives checked - warnings ****** ' )
 2070  FORMAT( ' *                                               * ', /,       &
               ' *********** Derivatives checked O.K. ************ ' )

!  End of subroutine DRCHG_check_main

     END SUBROUTINE DRCHG_check_main

!  End of module LANCELOT_DRCHG

   END MODULE LANCELOT_DRCHG_double

