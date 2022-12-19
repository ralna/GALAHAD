! THIS VERSION: GALAHAD 4.1 - 2022-12-18 AT 13:00 GMT.
#include "galahad_modules.h"
   SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp, lw1, lw2 )
   USE GALAHAD_PRECISION
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: lw1, lw2
   LOGICAL, INTENT( IN ) :: transp
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lw1 ) :: W1
   REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( lw2 ) :: W2
   SELECT CASE ( ieltyp )
   CASE ( 1 ) ! Element type 1 has a non-trivial transformation
     IF ( transp ) THEN
       W2( 1 ) = W1( 1 )
       W2( 2 ) = W1( 2 )
       W2( 3 ) = W1( 2 )
     ELSE
       W2( 1 ) = W1( 1 )
       W2( 2 ) = W1( 2 ) + W1( 3 )
    END IF
   CASE DEFAULT ! Element 2 has a trivial transformation - no action required
   END SELECT
   RETURN
   END SUBROUTINE RANGE
   
   SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,          &
                      IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,          &
                      lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,          &
                      lfuval, lxvalu, lepvlu, ifflag, ifstat )
   USE GALAHAD_PRECISION
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: lelvar, lntvar, lstadh, lstepa
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: lcalcf, lfuval, lxvalu, lepvlu
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ifstat
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: ITYPEE( LTYPEE ), ISTAEV( LSTAEV )
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: IELVAR( LELVAR ), INTVAR( LNTVAR )
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: ISTADH( LSTADH ), ISTEPA( LSTEPA )
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: ICALCF( LCALCF )
   REAL ( KIND = rp_ ), INTENT( IN ) :: XVALUE( LXVALU )
   REAL ( KIND = rp_ ), INTENT( IN ) :: EPVALU( LEPVLU )
   REAL ( KIND = rp_ ), INTENT( INOUT ) :: FUVALS( LFUVAL )
   INTEGER ( KIND = ip_ ) :: ielemn, ieltyp, ihstrt, ilstrt, igstrt, ipstrt
   INTEGER ( KIND = ip_ ) :: jcalcf
   REAL ( KIND = rp_ ) :: v1, v2, u1, u2, u3, cs, sn
   ifstat = 0
   DO jcalcf = 1, ncalcf
     ielemn = ICALCF( jcalcf ) 
     ilstrt = ISTAEV( ielemn ) - 1
     igstrt = INTVAR( ielemn ) - 1
     ipstrt = ISTEPA( ielemn ) - 1
     IF ( ifflag == 3 ) ihstrt = ISTADH( ielemn ) - 1
     ieltyp = ITYPEE( ielemn )
     SELECT CASE ( ieltyp )
     CASE ( 1 )
       u1 = XVALUE( IELVAR( ilstrt + 1 ) )
       u2 = XVALUE( IELVAR( ilstrt + 2 ) )
       u3 = XVALUE( IELVAR( ilstrt + 3 ) )
       v1 = u1
       v2 = u2 + u3
       cs = COS( v2 )
       sn = SIN( v2 )
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = v1 * sn
       ELSE
         FUVALS( igstrt + 1 ) = sn
         FUVALS( igstrt + 2 ) = v1 * cs
         IF ( ifflag == 3 ) THEN
           FUVALS( ihstrt + 1 ) = 0.0_rp_
           FUVALS( ihstrt + 2 ) = cs
           FUVALS( ihstrt + 3 ) = - v1 * sn
         END IF
       END IF
     CASE ( 2 )
       u1 = XVALUE( IELVAR( ilstrt + 1 ) )
       u2 = XVALUE( IELVAR( ilstrt + 2 ) )
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = u1 * u2
       ELSE
         FUVALS( igstrt + 1 ) = u2
         FUVALS( igstrt + 2 ) = u1
         IF ( ifflag == 3 ) THEN
           FUVALS( ihstrt + 1 ) = 0.0_rp_
           FUVALS( ihstrt + 2 ) = 1.0_rp_
           FUVALS( ihstrt + 3 ) = 0.0_rp_
         END IF
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE ELFUN 
   
   SUBROUTINE GROUP ( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg,                  &
                      ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa,                  &
                      lcalcg, lfvalu, lgpvlu, derivs, igstat )              
   USE GALAHAD_PRECISION
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: lgvalu, ncalcg, ltypeg, lstgpa
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: lcalcg, lfvalu, lgpvlu          
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: igstat
   LOGICAL, INTENT( IN ) :: derivs                                          
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ltypeg ) :: ITYPEG
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( lstgpa ) :: ISTGPA
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( lcalcg ) :: ICALCG
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lfvalu ) :: FVALUE           
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lgpvlu ) :: GPVALU           
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( lgvalu, 3 ) :: GVALUE
   INTEGER ( KIND = ip_ ) :: igrtyp, igroup, ipstrt, jcalcg
   REAL ( KIND = rp_ ) :: alpha, alpha2
   igstat = 0
   DO jcalcg = 1, ncalcg
     igroup = ICALCG( jcalcg )
     igrtyp = ITYPEG( igroup )
     IF ( igrtyp == 0 ) CYCLE ! skip if the group is trivial
     ipstrt = ISTGPA( igroup ) - 1
     SELECT CASE ( igrtyp )
     CASE ( 1 )
       alpha = FVALUE( igroup )
       IF ( .NOT. derivs ) THEN
         GVALUE( igroup, 1 ) = alpha * alpha
       ELSE
         GVALUE( igroup, 2 ) = 2.0_rp_ * alpha
         GVALUE( igroup, 3 ) = 2.0_rp_
       END IF
     CASE ( 2 )
       alpha  = FVALUE( igroup )
       alpha2 = alpha * alpha
       IF ( .NOT. derivs ) THEN
         GVALUE( igroup, 1 ) = alpha2 * alpha2
       ELSE
         GVALUE( igroup, 2 ) = 4.0_rp_ * alpha2 * alpha
         GVALUE( igroup, 3 ) = 12.0_rp_ * alpha2
       END IF
     CASE ( 3 )
       alpha  = FVALUE( igroup )
       IF ( .NOT. derivs ) THEN
         GVALUE( igroup, 1 ) = COS( alpha )
       ELSE
         GVALUE( igroup, 2 ) = - SIN( alpha )
         GVALUE( igroup, 3 ) = - COS( alpha )
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE GROUP

   PROGRAM LANCELOT_example
   USE GALAHAD_PRECISION
   USE LANCELOT_steering_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( LANCELOT_control_type ) :: control
   TYPE ( LANCELOT_inform_type ) :: info
   TYPE ( LANCELOT_data_type ) :: data
   TYPE ( LANCELOT_problem_type ) :: prob
   INTEGER ( KIND = ip_ ) :: i, lfuval
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, ng = 6, nel = 3
   INTEGER ( KIND = ip_ ), PARAMETER :: nnza = 4, nvrels = 7
   INTEGER ( KIND = ip_ ), PARAMETER :: ntotel = 3, ngpvlu = 0, nepvlu = 0
   INTEGER ( KIND = ip_ ) :: IVAR( n ), ICALCF( nel ), ICALCG( ng )
   REAL ( KIND = rp_ ) :: Q( n ), XT( n ), DGRAD( n ), FVALUE( ng )
   REAL ( KIND = rp_ ) :: GVALUE( ng, 3 )
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: FUVALS
   EXTERNAL RANGE, ELFUN, GROUP
! make space for problem data
   prob%n = n ; prob%ng = ng ; prob%nel = nel 
   ALLOCATE( prob%ISTADG( prob%ng + 1 ), prob%ISTGPA( prob%ng + 1 ) )
   ALLOCATE( prob%ISTADA( prob%ng + 1 ), prob%ISTAEV( prob%nel + 1 ) )
   ALLOCATE( prob%ISTEPA( prob%nel + 1 ), prob%ITYPEG( prob%ng ) )
   ALLOCATE( prob%KNDOFG( prob%ng ), prob%ITYPEE( prob%nel ) )
   ALLOCATE( prob%INTVAR( prob%nel + 1 ) )
   ALLOCATE( prob%IELING( ntotel ), prob%IELVAR( nvrels ), prob%ICNA( nnza ) )
   ALLOCATE( prob%ISTADH( prob%nel + 1 ), prob%A( nnza ) )
   ALLOCATE( prob%B( prob%ng ), prob%BL( prob%n ), prob%BU( prob%n ) )
   ALLOCATE( prob%X( prob%n ), prob%Y( prob%ng ), prob%C( prob%ng ) )
   ALLOCATE( prob%GPVALU( ngpvlu ), prob%EPVALU( nepvlu ) )
   ALLOCATE( prob%ESCALE( ntotel ), prob%GSCALE( prob%ng ) )
   ALLOCATE( prob%VSCALE( prob%n ) )
   ALLOCATE( prob%INTREP( prob%nel ), prob%GXEQX( prob%ng ) )
   ALLOCATE( prob%GNAMES( prob%ng ), prob%VNAMES( prob%n ) )
! set problem data
   prob%ISTADG = (/ 1, 1, 2, 3, 3, 4, 4 /)
   prob%IELVAR = (/ 2, 1, 3, 2, 3, 1, 2 /)
   prob%ISTAEV = (/ 1, 4, 6, 8 /) ; prob%INTVAR( : nel ) = (/ 2,  2, 2 /)
   prob%IELING = (/ 1, 2, 3 /) ; prob%ICNA = (/ 1, 2, 1, 2 /)
   prob%ISTADA = (/ 1, 2, 2, 2, 3, 3, 5 /)
   prob%KNDOFG = (/ 1, 1, 1, 1, 1, 2 /)
   prob%ITYPEG = (/ 1, 0, 2, 0, 1, 3 /) ; prob%ITYPEE = (/ 1, 2, 2 /)
   prob%ISTGPA = (/ 0, 0, 0, 0, 0, 0, 0 /) ; prob%ISTEPA = (/ 0, 0, 0, 0 /)
   prob%A = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 2.0_rp_ /)
   prob%B = (/ 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
   prob%BL = (/ - infinity, -1.0_rp_, 1.0_rp_ /)
   prob%BU = (/ infinity, 1.0_rp_, 2.0_rp_ /)
   prob%GSCALE = (/ 1.0_rp_, 1.0_rp_, 3.0_rp_, 1.0_rp_, 2.0_rp_, 1.0_rp_ /)
   prob%ESCALE = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   prob%VSCALE = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
   prob%X = (/ 0.0_rp_, 0.0_rp_, 1.5_rp_ /) ;  prob%Y( 6 ) = 0.0_rp_
   prob%INTREP = (/ .TRUE., .FALSE., .FALSE. /)
   prob%GXEQX = (/ .FALSE., .TRUE., .FALSE., .TRUE., .FALSE., .FALSE. /)
   prob%GNAMES = (/ 'Obj 1     ', 'Obj 2     ', 'Obj 3     ', 'Obj 4     ', &
                    'Obj 5     ', 'Constraint' /)
   prob%VNAMES = (/ 'x 1', 'x 2', 'x 3' /)
! allocate space for FUVALS
   lfuval = prob%nel + 2 * prob%n
   DO i = 1, prob%nel
     lfuval = lfuval + ( prob%INTVAR( i ) * ( prob%INTVAR( i ) + 3 ) ) / 2
   END DO
   ALLOCATE( FUVALS( lfuval ) )
! problem data complete
   CALL LANCELOT_initialize( data, control )     ! Initialize control parameters
   control%maxit = 100 ; control%out = 6 ! ; control%print_level = 1
   control%stopg = 0.00001_rp_ ; control%stopc = 0.00001_rp_
   control%linear_solver = 1
   control%exact_gcp = .FALSE.
   info%status = 0
! solve the problem
   CALL LANCELOT_solve(                                                        &
       prob, RANGE, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,        &
       IVAR, Q, DGRAD, control, info, data, ELFUN  = ELFUN , GROUP = GROUP )
!  DO 
!    CALL LANCELOT_solve(                                                 &
!        prob, RANGE, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG, &
!        IVAR, Q, DGRAD , control, info, data )
!    IF ( info%status >= 0 ) EXIT
!    IF ( info%status == - 1 .OR. info%status == - 3 .OR. info%status == - 7 ) &
!      THEN
!      CALL ELFUN ( FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,         &
!          prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,    &
!          ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,    &
!          prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n, &
!          prob%ISTEPA( prob%nel + 1 ) - 1, 1, i )
!      IF ( i /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
!    END IF  
!    IF ( info%status == - 1 .OR. info%status == - 5 .OR. info%status == - 6 ) &
!      THEN
!      CALL ELFUN ( FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,         &
!          prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,    &
!          ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,    &
!          prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n, &
!          prob%ISTEPA( prob%nel + 1 ) - 1, 3, i )
!      IF ( i /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
!    END IF  
!    IF ( info%status == - 2 .OR. info%status == - 4 ) THEN
!      CALL GROUP ( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,         &
!          prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,    &
!          prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .FALSE., i )
!      IF ( i /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
!    END IF  
!    IF ( info%status == - 2 .OR. info%status == - 5 ) THEN
!      CALL GROUP( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,          &
!          prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,    &
!          prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .TRUE., i )
!      IF ( i /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
!    END IF  
!  END DO
! act on return status
   IF ( info%status == 0 ) THEN                  !  Successful return
     WRITE( 6, "( I6, ' iterations. Optimal objective value =', &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" ) &
     info%iter, info%obj, prob%X
   ELSE                                          !  Error returns
     WRITE( 6, "( ' LANCELOT_solve exit status = ', I6 ) " ) info%status
   END IF
   CALL LANCELOT_terminate( data, control, info ) !  delete internal workspace
   DEALLOCATE( prob%GNAMES, prob%VNAMES )         !  delete problem space
   DEALLOCATE( prob%VSCALE, prob%ESCALE, prob%GSCALE, prob%INTREP )
   DEALLOCATE( prob%GPVALU, prob%EPVALU, prob%ITYPEG, prob%GXEQX )
   DEALLOCATE( prob%X, prob%Y, prob%C, prob%A, prob%B, prob%BL, prob%BU )
   DEALLOCATE( prob%ISTADH, prob%IELING, prob%IELVAR, prob%ICNA )
   DEALLOCATE( prob%INTVAR, prob%KNDOFG, prob%ITYPEE, prob%ISTEPA )
   DEALLOCATE( prob%ISTADA, prob%ISTAEV, prob%ISTADG, prob%ISTGPA )
   DEALLOCATE( FUVALS )
   END PROGRAM LANCELOT_example

