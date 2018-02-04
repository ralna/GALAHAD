! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
!  =============================================================================
!  --------------------------- TEST PROBLEM 1 ----------------------------------
!  =============================================================================

   SUBROUTINE RANGE1( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp, lw1, lw2 )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
   LOGICAL, INTENT( IN ) :: transp
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
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
   END SUBROUTINE RANGE1

   SUBROUTINE ELFUN1( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV, &
                      IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee, &
                      lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf, &
                      lfuval, lxvalu, lepvlu, ifflag, ifstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
   INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
   INTEGER, INTENT( OUT ) :: ifstat
   INTEGER, INTENT( IN ) :: ITYPEE( ltypee ), ISTAEV( lstaev ), IELVAR( lelvar )
   INTEGER, INTENT( IN ) :: INTVAR( lntvar ), ISTADH( lstadh ), ISTEPA( lstepa )
   INTEGER, INTENT( IN ) :: ICALCF( lcalcf )
   REAL ( KIND = wp ), INTENT( IN ) :: XVALUE( lxvalu )
   REAL ( KIND = wp ), INTENT( IN ) :: EPVALU( lepvlu )
   REAL ( KIND = wp ), INTENT( INOUT ) :: FUVALS( lfuval )
   INTEGER :: ielemn, ieltyp, ihstrt, ilstrt, igstrt, ipstrt, jcalcf
   REAL ( KIND = wp ) :: v1, v2, u1, u2, u3, cs, sn
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
           FUVALS( ihstrt + 1 ) = 0.0_wp
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
           FUVALS( ihstrt + 1 ) = 0.0_wp
           FUVALS( ihstrt + 2 ) = 1.0_wp
           FUVALS( ihstrt + 3 ) = 0.0_wp
         END IF
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE ELFUN1

   SUBROUTINE ELFUN1_flexible(                                        &
                      FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV, &
                      IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee, &
                      lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf, &
                      lfuval, lxvalu, lepvlu, llders, ifflag, ELDERS, &
                      ifstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
   INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
   INTEGER, INTENT( IN ) :: llders
   INTEGER, INTENT( OUT ) :: ifstat
   INTEGER, INTENT( IN ) :: ITYPEE( ltypee ), ISTAEV( lstaev ), IELVAR( lelvar )
   INTEGER, INTENT( IN ) :: INTVAR( lntvar ), ISTADH( lstadh ), ISTEPA( lstepa )
   INTEGER, INTENT( IN ) :: ICALCF( lcalcf ), ELDERS( 2, llders )
   REAL ( KIND = wp ), INTENT( IN ) :: XVALUE( lxvalu )
   REAL ( KIND = wp ), INTENT( IN ) :: EPVALU( lepvlu )
   REAL ( KIND = wp ), INTENT( INOUT ) :: FUVALS( lfuval )
   INTEGER :: ielemn, ieltyp, ihstrt, ilstrt, igstrt, ipstrt, jcalcf
   REAL ( KIND = wp ) :: v1, v2, u1, u2, u3, cs, sn
   ifstat = 0
   DO jcalcf = 1, ncalcf
     ielemn = ICALCF( jcalcf )
     ilstrt = ISTAEV( ielemn ) - 1
     igstrt = INTVAR( ielemn ) - 1
     ipstrt = ISTEPA( ielemn ) - 1
     IF ( ELDERS( 2, ielemn ) <= 0 ) ihstrt = ISTADH( ielemn ) - 1
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
         IF ( ELDERS( 1, ielemn ) <= 0 ) THEN
           FUVALS( igstrt + 1 ) = sn
           FUVALS( igstrt + 2 ) = v1 * cs
           IF ( ELDERS( 2, ielemn ) <= 0 ) THEN
             FUVALS( ihstrt + 1 ) = 0.0_wp
             FUVALS( ihstrt + 2 ) = cs
             FUVALS( ihstrt + 3 ) = - v1 * sn
           END IF
         END IF
       END IF
     CASE ( 2 )
       u1 = XVALUE( IELVAR( ilstrt + 1 ) )
       u2 = XVALUE( IELVAR( ilstrt + 2 ) )
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = u1 * u2
       ELSE
         IF ( ELDERS( 1, ielemn ) <= 0 ) THEN
           FUVALS( igstrt + 1 ) = u2
           FUVALS( igstrt + 2 ) = u1
           IF ( ELDERS( 2, ielemn ) <= 0 ) THEN
             FUVALS( ihstrt + 1 ) = 0.0_wp
             FUVALS( ihstrt + 2 ) = 1.0_wp
             FUVALS( ihstrt + 3 ) = 0.0_wp
           END IF
         END IF
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE ELFUN1_flexible

   SUBROUTINE GROUP1( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg, &
                      ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa, &
                      lcalcg, lfvalu, lgpvlu, derivs, igstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: lgvalu, ncalcg
   INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
   INTEGER, INTENT( OUT ) :: igstat
   LOGICAL, INTENT( IN ) :: derivs
   INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
   INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
   INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lfvalu ) :: FVALUE
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lgpvlu ) :: GPVALU
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( lgvalu, 3 ) :: GVALUE
   INTEGER :: igrtyp, igroup, ipstrt, jcalcg
   REAL ( KIND = wp ) :: alpha, alpha2
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
         GVALUE( igroup, 2 ) = 2.0_wp * alpha
         GVALUE( igroup, 3 ) = 2.0_wp
       END IF
     CASE ( 2 )
       alpha  = FVALUE( igroup )
       alpha2 = alpha * alpha
       IF ( .NOT. derivs ) THEN
         GVALUE( igroup, 1 ) = alpha2 * alpha2
       ELSE
         GVALUE( igroup, 2 ) = 4.0_wp * alpha2 * alpha
         GVALUE( igroup, 3 ) = 12.0_wp * alpha2
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
   END SUBROUTINE GROUP1

!  =============================================================================
!  --------------------------- TEST PROBLEM 2 ----------------------------------
!  =============================================================================

   SUBROUTINE RANGE2( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp, lw1, lw2 )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
   LOGICAL, INTENT( IN ) :: transp
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
   SELECT CASE ( ieltyp )
   CASE ( 2 ) ! Element type 2 has a non-trivial transformation
     IF ( TRANSP ) THEN
       W2( 1 ) =   W1( 1 )
       W2( 2 ) =   W1( 1 )
     ELSE
       W2( 1 ) =   W1( 1 ) + W1( 2 )
     END IF
   CASE DEFAULT ! Other elements have trivial transformations - no action needed
   END SELECT
   END SUBROUTINE RANGE2

   SUBROUTINE ELFUN2( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV, &
                      IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee, &
                      lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf, &
                      lfuval, lxvalu, lepvlu, ifflag, ifstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
   INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
   INTEGER, INTENT( OUT ) :: ifstat
   INTEGER, INTENT( IN ) :: ITYPEE( ltypee ), ISTAEV( lstaev ), IELVAR( lelvar )
   INTEGER, INTENT( IN ) :: INTVAR( lntvar ), ISTADH( lstadh ), ISTEPA( lstepa )
   INTEGER, INTENT( IN ) :: ICALCF( lcalcf )
   REAL ( KIND = wp ), INTENT( IN ) :: XVALUE( lxvalu )
   REAL ( KIND = wp ), INTENT( IN ) :: EPVALU( lepvlu )
   REAL ( KIND = wp ), INTENT( INOUT ) :: FUVALS( lfuval )
   INTEGER :: ielemn, ieltyp, ihstrt, ilstrt, igstrt, ipstrt, jcalcf
   REAL ( KIND = wp ) :: x, y, z, p, sinx, cosx, xx, yy
   ifstat = 0
   DO JCALCF = 1, NCALCF
     ielemn = ICALCF( jcalcf )
     ilstrt = ISTAEV( ielemn ) - 1
     igstrt = INTVAR( ielemn ) - 1
     ipstrt = ISTEPA( ielemn ) - 1
     IF ( ifflag == 3 ) ihstrt = ISTADH( ielemn ) - 1
     ieltyp = ITYPEE( ielemn )
     SELECT CASE ( ieltyp )
     CASE ( 1 )
       x = XVALUE( IELVAR( ilstrt + 1 ) )
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = x * x
       ELSE
         FUVALS( igstrt + 1 ) = x + x
         IF ( ifflag == 3 ) FUVALS( ihstrt + 1 ) = 2.0
       END IF
     CASE ( 2 )
       y = XVALUE( IELVAR( ilstrt + 1 ) )
       z = XVALUE( IELVAR( ilstrt + 2 ) )
       p = EPVALU( ipstrt + 1 )
       x = y + z
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn )= p * x * x
       ELSE
         FUVALS( igstrt + 1 ) = p * ( x + x )
         IF ( ifflag == 3 ) FUVALS( ihstrt + 1 ) = 2.0 * p
        END IF
     CASE ( 3 )
       x    = XVALUE( IELVAR( ilstrt + 1 ) )
       sinx = SIN( x )
       cosx = COS( x )
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = sinx * sinx
       ELSE
         FUVALS( igstrt + 1 ) = 2.0 * sinx * cosx
         IF ( ifflag == 3 )                                                    &
           FUVALS( ihstrt + 1 ) = 2.0 * ( cosx * cosx - sinx * sinx )
        END IF
     CASE ( 4 )
       x  = XVALUE( IELVAR( ilstrt + 1 ) )
       y  = XVALUE( IELVAR( ilstrt + 2 ) )
       xx = x * x
       yy = y * y
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = xx * yy
       ELSE
         FUVALS( igstrt + 1 ) = 2.0 * x * yy
         FUVALS( igstrt + 2 ) = 2.0 * xx * y
         IF ( ifflag == 3 ) THEN
           FUVALS( ihstrt + 1 ) = 2.0 * yy
           FUVALS( ihstrt + 2 ) = 4.0 * x * y
           FUVALS( ihstrt + 3 ) = 2.0 * xx
         END IF
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE ELFUN2

   SUBROUTINE ELFUN2E( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV, &
                      IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee, &
                      lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf, &
                      lfuval, lxvalu, lepvlu, ifflag, ifstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
   INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
   INTEGER, INTENT( OUT ) :: ifstat
   INTEGER, INTENT( IN ) :: ITYPEE( ltypee ), ISTAEV( lstaev ), IELVAR( lelvar )
   INTEGER, INTENT( IN ) :: INTVAR( lntvar ), ISTADH( lstadh ), ISTEPA( lstepa )
   INTEGER, INTENT( IN ) :: ICALCF( lcalcf )
   REAL ( KIND = wp ), INTENT( IN ) :: XVALUE( lxvalu )
   REAL ( KIND = wp ), INTENT( IN ) :: EPVALU( lepvlu )
   REAL ( KIND = wp ), INTENT( INOUT ) :: FUVALS( lfuval )
   INTEGER :: ielemn, ieltyp, ihstrt, ilstrt, igstrt, ipstrt, jcalcf
   REAL ( KIND = wp ) :: x, y, z, p, sinx, cosx, xx, yy
   ifstat = 0
   DO JCALCF = 1, NCALCF
     ielemn = ICALCF( jcalcf )
     ilstrt = ISTAEV( ielemn ) - 1
     igstrt = INTVAR( ielemn ) - 1
     ipstrt = ISTEPA( ielemn ) - 1
     IF ( ifflag == 3 ) ihstrt = ISTADH( ielemn ) - 1
     ieltyp = ITYPEE( ielemn )
     SELECT CASE ( ieltyp )
     CASE ( 1 )
       x = XVALUE( IELVAR( ilstrt + 1 ) )
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = x * x
       ELSE
         FUVALS( igstrt + 1 ) = x + x
         IF ( ifflag == 3 ) FUVALS( ihstrt + 1 ) = 2.0
       END IF
     CASE ( 2 )
       y = XVALUE( IELVAR( ilstrt + 1 ) )
       z = XVALUE( IELVAR( ilstrt + 2 ) )
       p = EPVALU( ipstrt + 1 )
       x = y + z
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn )= p * x * x
       ELSE
         FUVALS( igstrt + 1 ) = p * ( x + x )
         IF ( ifflag == 3 ) FUVALS( ihstrt + 1 ) = 2.0 * p
        END IF
     CASE ( 3 )
       x    = XVALUE( IELVAR( ilstrt + 1 ) )
       sinx = SIN( x )
       cosx = COS( x )
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = sinx * sinx
       ELSE
         FUVALS( igstrt + 1 ) = 2.0 * sinx * cosx
         IF ( ifflag == 3 )                                                    &
           FUVALS( ihstrt + 1 ) = 2.0 * ( cosx * cosx - sinx * sinx )
        END IF
     CASE ( 4 )
       x  = XVALUE( IELVAR( ilstrt + 1 ) )
       y  = XVALUE( IELVAR( ilstrt + 2 ) )
       xx = x * x
       yy = y * y
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = xx * yy
       ELSE
         FUVALS( igstrt + 1 ) = 2.0 * x * yy
         FUVALS( igstrt + 2 ) = 2.0 * xx * y
         IF ( ifflag == 3 ) THEN
           FUVALS( ihstrt + 1 ) = 2.0 * yy
           FUVALS( ihstrt + 2 ) = 4.0 * x * y
           FUVALS( ihstrt + 3 ) = 2.0 * xx
         END IF
       END IF
     END SELECT
   END DO
   ifstat = 1
   RETURN
   END SUBROUTINE ELFUN2E

   SUBROUTINE GROUP2( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg, &
                      ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa, &
                      lcalcg, lfvalu, lgpvlu, derivs, igstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: lgvalu, ncalcg
   INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
   INTEGER, INTENT( OUT ) :: igstat
   LOGICAL, INTENT( IN ) :: derivs
   INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
   INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
   INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lfvalu ) :: FVALUE
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lgpvlu ) :: GPVALU
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( lgvalu, 3 ) :: GVALUE
   INTEGER :: igrtyp, igroup, ipstrt, jcalcg
   REAL ( KIND = wp ) :: gvar  , p
   igstat = 0
   DO jcalcg = 1, ncalcg
     igroup = ICALCG( jcalcg )
     igrtyp = ITYPEG( igroup )
     IF ( igrtyp == 0 ) CYCLE ! skip if the group is trivial
     ipstrt = ISTGPA( igroup ) - 1
     SELECT CASE ( igrtyp )
     CASE ( 1 )
       gvar  = FVALUE( igroup )
       IF ( .NOT. derivs ) THEN
        GVALUE( igroup, 1 ) = gvar
       ELSE
        GVALUE( igroup, 2 ) = 1.0
        GVALUE( igroup, 3 ) = 0.0
       END IF
     CASE ( 2 )
       gvar = FVALUE( igroup )
       p    = GPVALU( ipstrt + 1 )
       IF ( .NOT. derivs ) THEN
        GVALUE( igroup, 1 ) = p * gvar * gvar
       ELSE
        GVALUE( igroup, 2 ) = p * ( gvar + gvar )
        GVALUE( igroup, 3 ) = 2.0 * p
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE GROUP2

   SUBROUTINE GROUP2E( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg, &
                       ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa, &
                       lcalcg, lfvalu, lgpvlu, derivs, igstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: lgvalu, ncalcg
   INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
   INTEGER, INTENT( OUT ) :: igstat
   LOGICAL, INTENT( IN ) :: derivs
   INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
   INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
   INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lfvalu ) :: FVALUE
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lgpvlu ) :: GPVALU
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( lgvalu, 3 ) :: GVALUE
   INTEGER :: igrtyp, igroup, ipstrt, jcalcg
   REAL ( KIND = wp ) :: gvar  , p
   igstat = 0
   DO jcalcg = 1, ncalcg
     igroup = ICALCG( jcalcg )
     igrtyp = ITYPEG( igroup )
     IF ( igrtyp == 0 ) CYCLE ! skip if the group is trivial
     ipstrt = ISTGPA( igroup ) - 1
     SELECT CASE ( igrtyp )
     CASE ( 1 )
       gvar  = FVALUE( igroup )
       IF ( .NOT. derivs ) THEN
        GVALUE( igroup, 1 ) = gvar
       ELSE
        GVALUE( igroup, 2 ) = 0.5
        GVALUE( igroup, 3 ) = 0.0
       END IF
     CASE ( 2 )
       gvar = FVALUE( igroup )
       p    = GPVALU( ipstrt + 1 )
       IF ( .NOT. derivs ) THEN
        GVALUE( igroup, 1 ) = p * gvar * gvar
       ELSE
        GVALUE( igroup, 2 ) = - p * ( gvar + gvar )
        GVALUE( igroup, 3 ) = 2.0 * p
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE GROUP2E

!  =============================================================================
!  --------------------------- TEST PROBLEM 3 ----------------------------------
!  =============================================================================

   SUBROUTINE RANGE3( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp, lw1, lw2 )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
   LOGICAL, INTENT( IN ) :: transp
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
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
   END SUBROUTINE RANGE3

   SUBROUTINE ELFUN3( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV, &
                      IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee, &
                      lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf, &
                      lfuval, lxvalu, lepvlu, ifflag, ifstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
   INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
   INTEGER, INTENT( OUT ) :: ifstat
   INTEGER, INTENT( IN ) :: ITYPEE( ltypee ), ISTAEV( lstaev ), IELVAR( lelvar )
   INTEGER, INTENT( IN ) :: INTVAR( lntvar ), ISTADH( lstadh ), ISTEPA( lstepa )
   INTEGER, INTENT( IN ) :: ICALCF( lcalcf )
   REAL ( KIND = wp ), INTENT( IN ) :: XVALUE( lxvalu )
   REAL ( KIND = wp ), INTENT( IN ) :: EPVALU( lepvlu )
   REAL ( KIND = wp ), INTENT( INOUT ) :: FUVALS( lfuval )
   INTEGER :: ielemn, ieltyp, ihstrt, ilstrt, igstrt, ipstrt, jcalcf
   REAL ( KIND = wp ) :: v1, v2, u1, u2, u3
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
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = v1 + v2
       ELSE
         FUVALS( igstrt + 1 ) = 1.0_wp
         FUVALS( igstrt + 2 ) = 1.0_wp
         IF ( ifflag == 3 ) THEN
           FUVALS( ihstrt + 1 ) = 0.0_wp
           FUVALS( ihstrt + 2 ) = 0.0_wp
           FUVALS( ihstrt + 3 ) = 0.0_wp
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
           FUVALS( ihstrt + 1 ) = 0.0_wp
           FUVALS( ihstrt + 2 ) = 1.0_wp
           FUVALS( ihstrt + 3 ) = 0.0_wp
         END IF
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE ELFUN3

   SUBROUTINE ELFUN3_flexible(                                        &
                      FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV, &
                      IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee, &
                      lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf, &
                      lfuval, lxvalu, lepvlu, llders, ifflag, ELDERS, &
                      ifstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
   INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
   INTEGER, INTENT( IN ) :: llders
   INTEGER, INTENT( OUT ) :: ifstat
   INTEGER, INTENT( IN ) :: ITYPEE( ltypee ), ISTAEV( lstaev ), IELVAR( lelvar )
   INTEGER, INTENT( IN ) :: INTVAR( lntvar ), ISTADH( lstadh ), ISTEPA( lstepa )
   INTEGER, INTENT( IN ) :: ICALCF( lcalcf ), ELDERS( 2, llders )
   REAL ( KIND = wp ), INTENT( IN ) :: XVALUE( lxvalu )
   REAL ( KIND = wp ), INTENT( IN ) :: EPVALU( lepvlu )
   REAL ( KIND = wp ), INTENT( INOUT ) :: FUVALS( lfuval )
   INTEGER :: ielemn, ieltyp, ihstrt, ilstrt, igstrt, ipstrt, jcalcf
   REAL ( KIND = wp ) :: v1, v2, u1, u2, u3
   ifstat = 0
   DO jcalcf = 1, ncalcf
     ielemn = ICALCF( jcalcf )
     ilstrt = ISTAEV( ielemn ) - 1
     igstrt = INTVAR( ielemn ) - 1
     ipstrt = ISTEPA( ielemn ) - 1
     IF ( ELDERS( 2, ielemn ) <= 0 ) ihstrt = ISTADH( ielemn ) - 1
     ieltyp = ITYPEE( ielemn )
     SELECT CASE ( ieltyp )
     CASE ( 1 )
       u1 = XVALUE( IELVAR( ilstrt + 1 ) )
       u2 = XVALUE( IELVAR( ilstrt + 2 ) )
       u3 = XVALUE( IELVAR( ilstrt + 3 ) )
       v1 = u1
       v2 = u2 + u3
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = v1 + v2
       ELSE
         IF ( ELDERS( 1, ielemn ) <= 0 ) THEN
           FUVALS( igstrt + 1 ) = 1.0_wp
           FUVALS( igstrt + 2 ) = 1.0_wp
           IF ( ELDERS( 2, ielemn ) <= 0 ) THEN
             FUVALS( ihstrt + 1 ) = 0.0_wp
             FUVALS( ihstrt + 2 ) = 0.0_wp
             FUVALS( ihstrt + 3 ) = 0.0_wp
           END IF
         END IF
       END IF
     CASE ( 2 )
       u1 = XVALUE( IELVAR( ilstrt + 1 ) )
       u2 = XVALUE( IELVAR( ilstrt + 2 ) )
       IF ( ifflag == 1 ) THEN
         FUVALS( ielemn ) = u1 * u2
       ELSE
         IF ( ELDERS( 1, ielemn ) <= 0 ) THEN
           FUVALS( igstrt + 1 ) = u2
           FUVALS( igstrt + 2 ) = u1
           IF ( ELDERS( 2, ielemn ) <= 0 ) THEN
             FUVALS( ihstrt + 1 ) = 0.0_wp
             FUVALS( ihstrt + 2 ) = 1.0_wp
             FUVALS( ihstrt + 3 ) = 0.0_wp
           END IF
         END IF
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE ELFUN3_flexible

   SUBROUTINE GROUP3( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg, &
                      ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa, &
                      lcalcg, lfvalu, lgpvlu, derivs, igstat )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, INTENT( IN ) :: lgvalu, ncalcg
   INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
   INTEGER, INTENT( OUT ) :: igstat
   LOGICAL, INTENT( IN ) :: derivs
   INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
   INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
   INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lfvalu ) :: FVALUE
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lgpvlu ) :: GPVALU
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( lgvalu, 3 ) :: GVALUE
   INTEGER :: igrtyp, igroup, ipstrt, jcalcg
   REAL ( KIND = wp ) :: alpha
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
         GVALUE( igroup, 2 ) = 2.0_wp * alpha
         GVALUE( igroup, 3 ) = 2.0_wp
       END IF
     CASE ( 2 )
       alpha  = FVALUE( igroup )
       IF ( .NOT. derivs ) THEN
         GVALUE( igroup, 1 ) = alpha
       ELSE
         GVALUE( igroup, 2 ) = 1.0_wp
         GVALUE( igroup, 3 ) = 0.0_wp
       END IF
     CASE ( 3 )
       alpha  = FVALUE( igroup )
       IF ( .NOT. derivs ) THEN
         GVALUE( igroup, 1 ) = 2.0_wp * alpha
       ELSE
         GVALUE( igroup, 2 ) = 2.0_wp
         GVALUE( igroup, 3 ) = 0.0_wp
       END IF
     END SELECT
   END DO
   RETURN
   END SUBROUTINE GROUP3

!  =============================================================================
!  ----------------------- LANCELOT B * TEST DECK ------------------------------
!  =============================================================================

   PROGRAM LANCELOT_test_deck
   USE LANCELOT_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( LANCELOT_control_type ) :: control
   TYPE ( LANCELOT_inform_type ) :: info
   TYPE ( LANCELOT_data_type ) :: data
   TYPE ( LANCELOT_problem_type ) :: prob
   INTEGER :: i, lfuval, class, run, alloc_status, istat
   INTEGER, PARAMETER :: n1 = 3, ng1 = 6, nel1 = 3, nnza1 = 4, nvrels1 = 7
   INTEGER, PARAMETER :: ntotel1 = 3, ngpvlu1 = 0, nepvlu1 = 0
   INTEGER, PARAMETER :: n2 = 4, ng2 = 13, nel2 = 10, nnza2 = 4, nvrels2 = 14
   INTEGER, PARAMETER :: ntotel2 = 14, ngpvlu2 = 6, nepvlu2 = 2
   LOGICAL, PARAMETER :: noma27 = .FALSE.
   LOGICAL, PARAMETER :: noma61 = .FALSE.
   LOGICAL, PARAMETER :: noicfs = .FALSE.
!  LOGICAL, PARAMETER :: noma27 = .TRUE.
!  LOGICAL, PARAMETER :: noma61 = .TRUE.
!  LOGICAL, PARAMETER :: noicfs = .TRUE.
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: IVAR, ICALCF, ICALCG
   INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: ELDERS
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Q, XT, DGRAD, FVALUE
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: GVALUE
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: FUVALS
   EXTERNAL RANGE1, ELFUN1, GROUP1, ELFUN1_flexible
   EXTERNAL RANGE2, ELFUN2, GROUP2, ELFUN2E, GROUP2E
   EXTERNAL RANGE3, ELFUN3, GROUP3

! make space for problem data

   prob%n = n2 ; prob%ng = ng2 ; prob%nel = nel2
   ALLOCATE( prob%ISTADG( prob%ng + 1 ), prob%ISTGPA( prob%ng + 1 ) )
   ALLOCATE( prob%ISTADA( prob%ng + 1 ), prob%ISTAEV( prob%nel + 1 ) )
   ALLOCATE( prob%ISTEPA( prob%nel + 1 ), prob%ITYPEG( prob%ng ) )
   ALLOCATE( prob%KNDOFG( prob%ng ), prob%ITYPEE( prob%nel ) )
   ALLOCATE( prob%INTVAR( prob%nel + 1 ), prob%ICNA( nnza2 ) )
   ALLOCATE( prob%IELING( ntotel2 ), prob%IELVAR( nvrels2 ) )
   ALLOCATE( prob%ISTADH( prob%nel + 1 ), prob%A( nnza2 ) )
   ALLOCATE( prob%B( prob%ng ), prob%BL( prob%n ), prob%BU( prob%n ) )
   ALLOCATE( prob%X( prob%n ), prob%Y( prob%ng ), prob%C( prob%ng ) )
   ALLOCATE( prob%GPVALU( ngpvlu2 ), prob%EPVALU( nepvlu2 ) )
   ALLOCATE( prob%ESCALE( ntotel2 ), prob%GSCALE( prob%ng ) )
   ALLOCATE( prob%VSCALE( prob%n ) )
   ALLOCATE( prob%INTREP( prob%nel ), prob%GXEQX( prob%ng ) )
   ALLOCATE( prob%GNAMES( prob%ng ), prob%VNAMES( prob%n ) )

   ALLOCATE( IVAR( prob%n ), ICALCF( prob%nel ), ICALCG( prob%ng ) )
   ALLOCATE( Q( prob%n ), XT( prob%n ), DGRAD( prob%n ), FVALUE( prob%ng ) )
   ALLOCATE( GVALUE( prob%ng, 3 ), ELDERS( 2, prob%nel ) )

!  set general problem data

   prob%ISTADG = (/ 1, 1, 1, 2, 4, 6, 7, 7, 7, 8, 10, 12, 13, 15 /)
   prob%ISTGPA = (/ 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7 /)
   prob%ISTADA = (/ 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5 /)
   prob%ISTAEV = (/ 1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 15 /)
   prob%ISTEPA = (/ 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3 /)
   prob%ITYPEG = (/ 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1 /)
   prob%KNDOFG = (/ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2 /)
   prob%ITYPEE = (/ 1, 1, 2, 3, 4, 1, 1, 2, 3, 4 /)
   prob%IELING = (/ 1, 2, 3, 4, 5, 4, 6, 7, 8, 9, 10, 9, 1, 2 /)
   prob%IELVAR = (/ 1, 2, 3, 4, 3, 1, 2, 2, 3, 4, 1, 4, 2, 3 /)
   prob%ICNA   = (/ 3, 4, 4, 1 /)
   prob%A      = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
   prob%B      = (/ 0.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 3.0_wp, 0.0_wp, 0.0_wp, &
  &                 1.0_wp, 0.0_wp, 0.0_wp, 4.0_wp, 0.0_wp, 1.0_wp /)
   prob%BL     = (/ -infinity, 1.0_wp, -infinity, 2.0_wp /)
   prob%BU     = (/ infinity, infinity, 1.0_wp, 2.0_wp /)
   prob%GPVALU = 1.0_wp
   prob%EPVALU = 1.0_wp
   prob%ESCALE = 1.0_wp
   prob%GSCALE = 1.0_wp
   prob%VSCALE = 1.0_wp
   prob%INTREP = (/ .FALSE., .FALSE., .TRUE., .FALSE., .FALSE., .FALSE.,       &
  &                 .FALSE., .TRUE., .FALSE., .FALSE. /)
   prob%GXEQX  = .FALSE.
   prob%GNAMES = (/ 'FT1       ', 'FT2       ', 'FT3       ', 'FT4       ',    &
                    'FT5       ', 'FT6       ', 'FNT1      ', 'FNT2      ',    &
                    'FT3       ', 'FT4       ', 'FNT5      ', 'FNT6      ',    &
                    'C1        ' /)
   prob%VNAMES = (/ 'X1 ', 'X2 ', 'X3 ', 'X4 ' /)

!  Test error returns

   WRITE( 6, "( /, ' Testing error retuns ...', / )" )

!  DO run = 1, 0
   DO run = 1, 17

! set problem data

     prob%INTVAR( : prob%nel ) = (/ 1, 1, 1, 1, 2, 1, 1, 1, 1, 2 /)
     prob%X = 0.0_wp ; prob%Y = 0.0_wp

! allocate space for FUVALS

     lfuval = prob%nel + 2 * prob%n
     DO i = 1, prob%nel
       lfuval = lfuval + ( prob%INTVAR( i ) * ( prob%INTVAR( i ) + 3 ) ) / 2
     END DO
     ALLOCATE( FUVALS( lfuval ) )

!  problem data complete

     CALL LANCELOT_initialize( data, control )  ! Initialize control parameters
!    control%maxit = 100 ; control%out = 6 ; control%print_level = 1
!    control%maxit = 100 ; control%out = 6 ; control%print_level = 6
     control%stopg = 0.00001_wp ; control%stopc = 0.00001_wp
     control%second_derivatives = 0 ; control%exact_gcp = .FALSE.
     info%status = 0

     IF ( run == 1 ) THEN
       control%maxit = 1
     ELSE IF ( run == 2 ) THEN
       control%initial_radius = EPSILON( 1.0_wp ) ** 2
       control%print_level = - 1
     ELSE IF ( run == 3 ) THEN
       control%stopg = 0.0000000000000001_wp
       prob%KNDOFG( 13 ) = 1
     ELSE IF ( run == 4 ) THEN
       prob%KNDOFG( 13 ) = 2
       DEALLOCATE( FUVALS, STAT = alloc_status )
       CYCLE
     ELSE IF ( run == 5 ) THEN
       DEALLOCATE( FUVALS, STAT = alloc_status )
       CYCLE
     ELSE IF ( run == 6 ) THEN
       DEALLOCATE( FUVALS, STAT = alloc_status )
       CYCLE
     ELSE IF ( run == 7 ) THEN
       prob%KNDOFG( 1 ) = 7
     ELSE IF ( run == 8 ) THEN
       prob%KNDOFG( 1 ) = 1
       prob%B( 13 ) = - 1.0_wp
     ELSE IF ( run == 9 ) THEN
       prob%B( 13 ) = 1.0_wp
       DEALLOCATE( prob%Y )
     ELSE IF ( run == 10 ) THEN
       DEALLOCATE( prob%ISTEPA )
     ELSE IF ( run == 11 ) THEN
       DEALLOCATE( prob%ISTGPA )
     ELSE IF ( run == 12 ) THEN
       DEALLOCATE( FUVALS, STAT = alloc_status )
       CYCLE
     ELSE IF ( run == 14 ) THEN
     ELSE IF ( run == 15 ) THEN
       prob%n = - 1
     END IF

!  solve the problem

     IF ( run == 2 ) THEN
       CALL LANCELOT_solve(                                                    &
           prob, RANGE2, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,   &
           IVAR, Q, DGRAD, control, info, data, ELFUN = ELFUN2, GROUP = GROUP2E)
     ELSE IF ( run == 13 ) THEN
       CALL LANCELOT_solve(                                                    &
           prob, RANGE2, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,   &
          IVAR, Q, DGRAD, control, info, data, ELFUN = ELFUN2E, GROUP = GROUP2)
     ELSE IF ( run == 14 ) THEN
       DO
         CALL LANCELOT_solve(                                                  &
             prob, RANGE2, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG, &
             IVAR, Q, DGRAD , control, info, data, ELFUN = ELFUN2 )
         IF ( info%status >= 0 ) EXIT
         IF ( info%status == - 2 .OR. info%status == - 4 ) THEN
           OPEN( control%alive_unit, FILE = control%alive_file,                &
                 FORM = 'FORMATTED', STATUS = 'OLD' )
           CLOSE( UNIT = control%alive_unit, IOSTAT = i, STATUS = 'DELETE' )
           CALL GROUP2( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .FALSE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 2 .OR. info%status == - 5 ) THEN
           CALL GROUP2( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .TRUE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
       END DO
     ELSE IF ( run == 16 ) THEN
       ELDERS( 1, : ) = - 1 ; ELDERS( 2, : ) = - 1
       CALL LANCELOT_solve(                                                    &
           prob, RANGE2, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,   &
           IVAR, Q, DGRAD, control, info, data, ELFUN = ELFUN2E,               &
           GROUP = GROUP2, ELDERS = ELDERS )
     ELSE IF ( run == 17 ) THEN
       CALL LANCELOT_solve(                                                    &
           prob, RANGE2, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,   &
           IVAR, Q, DGRAD, control, info, data, GROUP = GROUP2,                &
           ELFUN_flexible = ELFUN1_flexible )
     ELSE
       CALL LANCELOT_solve(                                                    &
           prob, RANGE2, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,   &
           IVAR, Q, DGRAD, control, info, data, ELFUN = ELFUN2, GROUP = GROUP2 )
     END IF

! act on return status

     IF ( info%status == 0 ) THEN                  !  Successful return
       WRITE( 6, "( ' run ', I2, I6, ' iterations. Optimal objective value =', &
      &       ES12.4 )" ) run, info%iter, info%obj
     ELSE                                          !  Error returns
       WRITE( 6, "( ' run ', I2, ' LANCELOT_solve exit status = ', I6 ) " )    &
         run, info%status
     END IF
     CALL LANCELOT_terminate( data, control, info ) !delete internal workspace
     DEALLOCATE( FUVALS, STAT = alloc_status )
     IF ( run == 9 ) THEN
       ALLOCATE( prob%Y( prob%ng ) )
     ELSE IF ( run == 10 ) THEN
       ALLOCATE( prob%ISTEPA( prob%nel + 1 ) )
       prob%ISTEPA = (/ 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3 /)
     ELSE IF ( run == 11 ) THEN
       ALLOCATE( prob%ISTGPA( prob%ng + 1 ) )
       prob%ISTGPA = (/ 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7 /)
     ELSE IF ( run == 15 ) THEN
       prob%n = n2
     END IF
   END DO

!  Test general options

   WRITE( 6, "( /, ' Testing general options ...', / )" )

!  DO run = 1, 0
   DO run = 1, 26

     IF ( run == 5 .AND. noma61 ) THEN
       WRITE( 6, "( A, I2, A )" ) ' Skipping run ', run,                      &
                           ' since MA61 is not available'
       CYCLE
     END IF
     IF ( ( run == 4 .OR. run == 6 .OR. run == 7 .OR.                         &
            run == 10 .OR. run == 11) .AND. noma27 ) THEN
       WRITE( 6, "( A, I2, A )" ) ' Skipping run ', run,                      &
                           ' since MA27 is not available'
       CYCLE
     END IF
     IF ( run == 9 .AND. noicfs ) THEN
       WRITE( 6, "( A, I2, A )" ) ' Skipping run ', run,                      &
                           ' since ICFS is not available'
       CYCLE
     END IF

! set problem data

     prob%INTVAR( : prob%nel ) = (/ 1, 1, 1, 1, 2, 1, 1, 1, 1, 2 /)
     prob%X = 0.0_wp ; prob%Y = 0.0_wp

! allocate space for FUVALS

     lfuval = prob%nel + 2 * prob%n
     DO i = 1, prob%nel
       lfuval = lfuval + ( prob%INTVAR( i ) * ( prob%INTVAR( i ) + 3 ) ) / 2
     END DO
     ALLOCATE( FUVALS( lfuval ) )

!  problem data complete

     CALL LANCELOT_initialize( data, control )  ! Initialize control parameters
!    control%maxit = 100 ; control%out = 6 ; control%print_level = 1
!    control%maxit = 100 ; control%out = 6 ; control%print_level = 6
     control%stopg = EPSILON( 1.0_wp ) ** 0.333
     control%stopc = control%stopg
     control%second_derivatives = 0 ; control%exact_gcp = .FALSE.
     info%status = 0

     IF ( run == 1 ) THEN
       control%linear_solver = 1
     ELSE IF ( run == 2 ) THEN
       control%linear_solver = 2
     ELSE IF ( run == 3 ) THEN
       control%linear_solver = 3
     ELSE IF ( run == 4 ) THEN
       control%linear_solver = 4
     ELSE IF ( run == 5 ) THEN
       control%linear_solver = 5
     ELSE IF ( run == 6 ) THEN
       control%linear_solver = 6
     ELSE IF ( run == 7 ) THEN
       control%linear_solver = 7
     ELSE IF ( run == 8 ) THEN
       control%linear_solver = 8
     ELSE IF ( run == 9 ) THEN
       control%linear_solver = 9
     ELSE IF ( run == 10 ) THEN
       control%linear_solver = 11
     ELSE IF ( run == 11 ) THEN
       control%linear_solver = 12
     ELSE IF ( run == 12 ) THEN
       control%more_toraldo = 5
     ELSE IF ( run == 13 ) THEN
       control%non_monotone = 0
     ELSE IF ( run == 14 ) THEN
       control%first_derivatives = 1
     ELSE IF ( run == 15 ) THEN
       control%first_derivatives = 2
     ELSE IF ( run == 16 ) THEN
       control%second_derivatives = 1
     ELSE IF ( run == 17 ) THEN
       control%second_derivatives = 2
     ELSE IF ( run == 18 ) THEN
       control%second_derivatives = 3
     ELSE IF ( run == 19 ) THEN
       control%second_derivatives = 4
     ELSE IF ( run == 20 ) THEN
       control%initial_radius = 1.0_wp
     ELSE IF ( run == 21 ) THEN
       control%maximum_radius = 1.0_wp
     ELSE IF ( run == 22 ) THEN
       control%two_norm_tr = .TRUE.
     ELSE IF ( run == 23 ) THEN
       control%exact_gcp = .FALSE.
     ELSE IF ( run == 24 ) THEN
       control%accurate_bqp = .TRUE.
     ELSE IF ( run == 25 ) THEN
       control%structured_tr = .TRUE.
     ELSE IF ( run == 26 ) THEN
       control%print_max = .TRUE.
     END IF

!  solve the problem

 100 CONTINUE
     CALL LANCELOT_solve(                                                      &
         prob, RANGE2, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,     &
         IVAR, Q, DGRAD, control, info, data, ELFUN = ELFUN2, GROUP = GROUP2 )
     IF ( run == 3 ) THEN
       IF ( info%status == - 8 .OR. info%status == - 9 .OR.                    &
            info%status == - 10 ) THEN
!        Q( IVAR( : info%nvar ) ) = DGRAD( IVAR( : info%nvar ) )
         Q( IVAR( : info%nvar ) ) = DGRAD( : info%nvar )
         GO TO 100
       END IF
     ENDIF

! act on return status

     IF ( info%status == 0 ) THEN                  !  Successful return
       WRITE( 6, "( ' run ', I2, I6, ' iterations. Optimal objective value =', &
      &       ES12.4 )" ) run, info%iter, info%obj
     ELSE                                          !  Error returns
       WRITE( 6, "( ' run ', I2, ' LANCELOT_solve exit status = ', I6 ) " )    &
         run, info%status
     END IF
     CALL LANCELOT_terminate( data, control, info ) !delete internal workspace
     DEALLOCATE( FUVALS, STAT = alloc_status )
   END DO

   IF ( ALLOCATED( prob%KNDOFG ) ) DEALLOCATE( prob%KNDOFG )
   DEALLOCATE( prob%ISTADG, prob%ISTGPA, prob%ISTADA, prob%ISTAEV )
   DEALLOCATE( prob%ISTEPA, prob%ITYPEG, prob%ITYPEE, prob%INTVAR )
   DEALLOCATE( prob%IELING, prob%IELVAR, prob%ICNA, prob%ISTADH )
   DEALLOCATE( prob%A, prob%B, prob%BL, prob%BU, prob%X, prob%Y, prob%C )
   DEALLOCATE( prob%GPVALU, prob%EPVALU, prob%ESCALE, prob%GSCALE )
   DEALLOCATE( prob%VSCALE, prob%INTREP, prob%GXEQX, prob%GNAMES, prob%VNAMES )
   DEALLOCATE( IVAR, ICALCF, ICALCG, Q, XT, DGRAD, FVALUE, GVALUE, ELDERS )

! make space for problem data

   prob%n = n1 ; prob%ng = ng1 ; prob%nel = nel1
   ALLOCATE( prob%ISTADG( prob%ng + 1 ), prob%ISTGPA( prob%ng + 1 ) )
   ALLOCATE( prob%ISTADA( prob%ng + 1 ), prob%ISTAEV( prob%nel + 1 ) )
   ALLOCATE( prob%ISTEPA( prob%nel + 1 ), prob%ITYPEG( prob%ng ) )
   ALLOCATE( prob%KNDOFG( prob%ng ), prob%ITYPEE( prob%nel ) )
   ALLOCATE( prob%INTVAR( prob%nel + 1 ), prob%ICNA( nnza1 ) )
   ALLOCATE( prob%IELING( ntotel1 ), prob%IELVAR( nvrels1 ) )
   ALLOCATE( prob%ISTADH( prob%nel + 1 ), prob%A( nnza1 ) )
   ALLOCATE( prob%B( prob%ng ), prob%BL( prob%n ), prob%BU( prob%n ) )
   ALLOCATE( prob%X( prob%n ), prob%Y( prob%ng ), prob%C( prob%ng ) )
   ALLOCATE( prob%GPVALU( ngpvlu1 ), prob%EPVALU( nepvlu1 ) )
   ALLOCATE( prob%ESCALE( ntotel1 ), prob%GSCALE( prob%ng ) )
   ALLOCATE( prob%VSCALE( prob%n ) )
   ALLOCATE( prob%INTREP( prob%nel ), prob%GXEQX( prob%ng ) )
   ALLOCATE( prob%GNAMES( prob%ng ), prob%VNAMES( prob%n ) )

   ALLOCATE( IVAR( prob%n ), ICALCF( prob%nel ), ICALCG( prob%ng ) )
   ALLOCATE( Q( prob%n ), XT( prob%n ), DGRAD( prob%n ), FVALUE( prob%ng ) )
   ALLOCATE( GVALUE( prob%ng, 3 ) )
   ALLOCATE( ELDERS( 2, prob%nel ) )

!  set general problem data

   prob%ISTADG = (/ 1, 1, 2, 3, 3, 4, 4 /)
   prob%IELVAR = (/ 2, 1, 3, 2, 3, 1, 2 /) ; prob%ISTAEV = (/ 1, 4, 6, 8 /)
   prob%IELING = (/ 1, 2, 3 /) ; prob%KNDOFG = (/ 1, 1, 1, 1, 1, 2 /)
   prob%ISTADA = (/ 1, 2, 2, 2, 3, 3, 5 /) ; prob%ICNA = (/ 1, 2, 1, 2 /)
   prob%ITYPEG = (/ 1, 0, 2, 0, 1, 3 /) ; prob%ITYPEE = (/ 1, 2, 2 /)
   prob%ISTGPA = (/ 0, 0, 0, 0, 0, 0, 0 /) ; prob%ISTEPA = (/ 0, 0, 0, 0 /)
   ELDERS( 1, : ) = (/ 1, 1, 0 /)
   prob%A = (/ 1.0_wp, 1.0_wp, 1.0_wp, 2.0_wp /)
   prob%B = (/ 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
   prob%BL = (/ - infinity, -1.0_wp, 1.0_wp /)
   prob%BU = (/ infinity, 1.0_wp, 2.0_wp /)
   prob%GSCALE = (/ 1.0_wp, 1.0_wp, 3.0_wp, 1.0_wp, 2.0_wp, 1.0_wp /)
   prob%ESCALE = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
   prob%VSCALE = (/ 1.0_wp, 1.0_wp, 1.0_wp /)
   prob%INTREP = (/ .TRUE., .FALSE., .FALSE. /)
   prob%GXEQX = (/ .FALSE., .TRUE., .FALSE., .TRUE., .FALSE., .FALSE. /)
   prob%GNAMES = (/ 'Obj 1     ', 'Obj 2     ', 'Obj 3     ', 'Obj 4     ',    &
                    'Obj 5     ', 'Constraint' /)
   prob%VNAMES = (/ 'x 1', 'x 2', 'x 3' /)

!  DO class = 1, 0
   DO class = 1, 4

   IF ( class == 1 ) THEN
     WRITE( 6, "( /,                                                           &
    &  ' Testing optional arguments (general problems) ...', / )" )
   ELSE IF ( class == 2 ) THEN
     WRITE( 6, "( /,                                                           &
    &  ' Testing optional arguments (unconstrained problems) ...', / )" )
     prob%KNDOFG( 6 ) = 1
   ELSE IF ( class == 3 ) THEN
     WRITE( 6, "( /,                                                           &
    &  ' Testing optional arguments (subset problems) ...', / )" )
     prob%KNDOFG( 1 ) = 0
     prob%KNDOFG( 6 ) = 2
!    prob%KNDOFG = 0
   ELSE
     stop
     WRITE( 6, "( /,                                                           &
    &  ' Testing optional arguments (least-squares problems) ...', / )" )
     prob%ng = 1
     IF ( .NOT. ALLOCATED( prob%KNDOFG ) ) ALLOCATE( prob%KNDOFG( prob%ng ) )
     IF ( ALLOCATED( GVALUE ) ) DEALLOCATE( GVALUE )
     ALLOCATE( GVALUE( prob%ng, 3 ) )

     prob%GXEQX( 1 ) = .FALSE.
     prob%GNAMES( 1 ) = 'Constraint'
     prob%KNDOFG( 1 ) = 1
     prob%B( 1 ) = 1.0_wp
     prob%Y( 1 ) = 0.0_wp
     prob%ISTADA( 1 ) = 1 ; prob%ISTADA( 2 ) = 3
     prob%ICNA( 1 ) = 1 ; prob%ICNA( 2 ) = 2
     prob%A( 1 ) = 1.0_wp ; prob%A( 2 ) = 2.0_wp
     prob%ISTADG( 1 ) = 1; prob%ISTADG( 2 ) = 2
     prob%IELING( 1 ) = 2
   END IF

!  Test optional arguments (general problems)

!  DO run = 1, 0
   DO run = 1, 8

! set problem data

     prob%INTVAR( : prob%nel ) = (/ 2,  2, 2 /)
     prob%X = (/ 0.0_wp, 0.0_wp, 1.5_wp /) ;  prob%Y( 6 ) = 0.0_wp

! allocate space for FUVALS

     lfuval = prob%nel + 2 * prob%n
     DO i = 1, prob%nel
       lfuval = lfuval + ( prob%INTVAR( i ) * ( prob%INTVAR( i ) + 3 ) ) / 2
     END DO
     ALLOCATE( FUVALS( lfuval ) )

!  problem data complete

     CALL LANCELOT_initialize( data, control )  ! Initialize control parameters
     IF ( class == 3 ) THEN
       control%maxit = 100 ; control%out = 6 ! ; control%print_level = 1
     END IF
!    control%maxit = 100 ; control%out = 6 ! ; control%print_level = 1
!    control%maxit = 100 ; control%out = 6 ; control%print_level = 6
!    control%stopg = 10.0_wp * SQRT( EPSILON( 1.0_wp ) )
     control%stopc = control%stopg
!    control%linear_solver = 1
     control%second_derivatives = 0
     control%more_toraldo = 5
!    control%exact_gcp = .FALSE.
     info%status = 0

!  solve the problem

!  internal evaluation of elements and groups

     IF ( run == 1 ) THEN
       CALL LANCELOT_solve(                                                    &
!          prob, RANGE3, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,   &
!          IVAR, Q, DGRAD, control, info, data, ELFUN = ELFUN3, GROUP = GROUP3 )
           prob, RANGE1, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,   &
           IVAR, Q, DGRAD, control, info, data, ELFUN = ELFUN1, GROUP = GROUP1 )

!  internal evaluation of elements, external of groups

     ELSE IF ( run == 2 ) THEN
       DO
         CALL LANCELOT_solve(                                                  &
             prob, RANGE1, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG, &
             IVAR, Q, DGRAD , control, info, data, ELFUN = ELFUN1 )
         IF ( info%status >= 0 ) EXIT
         IF ( info%status == - 2 .OR. info%status == - 4 ) THEN
           CALL GROUP1( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .FALSE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 2 .OR. info%status == - 5 ) THEN
           CALL GROUP1( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .TRUE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
       END DO

!  internal evaluation of groups, external for elements

     ELSE IF ( run == 3 ) THEN
       DO
         CALL LANCELOT_solve(                                                  &
             prob, RANGE1, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG, &
             IVAR, Q, DGRAD , control, info, data, GROUP = GROUP1 )
         IF ( info%status >= 0 ) EXIT
         IF ( info%status == - 1 .OR. info%status == - 3 .OR.                  &
              info%status == - 7 ) THEN
           CALL ELFUN1( FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,     &
               prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,&
               ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,&
               prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval,     &
               prob%n, prob%ISTEPA( prob%nel + 1 ) - 1, 1, istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 1 .OR. info%status == - 5 .OR.                  &
              info%status == - 6 ) THEN
           CALL ELFUN1( FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,     &
               prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,&
               ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,&
               prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval,     &
               prob%n, prob%ISTEPA( prob%nel + 1 ) - 1, 3, istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
       END DO

!  external evaluation of elements and groups

     ELSE IF ( run == 4 ) THEN
       DO
         CALL LANCELOT_solve(                                                  &
             prob, RANGE1, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG, &
             IVAR, Q, DGRAD , control, info, data )
         IF ( info%status >= 0 ) EXIT
         IF ( info%status == - 1 .OR. info%status == - 3 .OR.                  &
              info%status == - 7 ) THEN
           CALL ELFUN1( FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,     &
               prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,&
               ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,&
               prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval,     &
               prob%n, prob%ISTEPA( prob%nel + 1 ) - 1, 1, istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 1 .OR. info%status == - 5 .OR.                  &
              info%status == - 6 ) THEN
           CALL ELFUN1( FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,     &
               prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,&
               ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,&
               prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval,     &
               prob%n, prob%ISTEPA( prob%nel + 1 ) - 1, 3, istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 2 .OR. info%status == - 4 ) THEN
           CALL GROUP1( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .FALSE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 2 .OR. info%status == - 5 ) THEN
           CALL GROUP1( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .TRUE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
       END DO

!  internal evaluation of elements and groups

     ELSE IF ( run == 5 ) THEN
       ELDERS( 2, : ) = (/ 0, 1, 2 /)
       CALL LANCELOT_solve(                                                    &
           prob, RANGE1, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG,   &
           IVAR, Q, DGRAD, control, info, data, GROUP = GROUP1,                &
           ELFUN_flexible = ELFUN1_flexible, ELDERS = ELDERS )

!  internal evaluation of elements, external of groups

     ELSE IF ( run == 6 ) THEN
       ELDERS( 2, : ) = (/ 0, 3, 4 /)
       DO
         CALL LANCELOT_solve(                                                  &
             prob, RANGE1, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG, &
             IVAR, Q, DGRAD , control, info, data,                             &
             ELFUN_flexible = ELFUN1_flexible, ELDERS = ELDERS )
         IF ( info%status >= 0 ) EXIT
         IF ( info%status == - 2 .OR. info%status == - 4 ) THEN
           CALL GROUP1( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .FALSE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 2 .OR. info%status == - 5 ) THEN
           CALL GROUP1( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .TRUE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
       END DO

!  internal evaluation of groups, external for elements

     ELSE IF ( run == 7 ) THEN
       ELDERS( 2, : ) = (/ 0, 1, 2 /)
       DO
         CALL LANCELOT_solve(                                                  &
             prob, RANGE1, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG, &
             IVAR, Q, DGRAD , control, info, data, GROUP = GROUP1,             &
             ELDERS = ELDERS )
         IF ( info%status >= 0 ) EXIT
         IF ( info%status == - 1 .OR. info%status == - 3 .OR.                  &
              info%status == - 7 ) THEN
           CALL ELFUN1_flexible(                                               &
               FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,              &
               prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,&
               ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,&
               prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval,     &
               prob%n, prob%ISTEPA( prob%nel + 1 ) - 1, prob%nel, 1,           &
               ELDERS, istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 1 .OR. info%status == - 5 .OR.                  &
              info%status == - 6 ) THEN
           CALL ELFUN1_flexible(                                               &
               FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,              &
               prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,&
               ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,&
               prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval,     &
               prob%n, prob%ISTEPA( prob%nel + 1 ) - 1, prob%nel, 3,           &
               ELDERS, istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
       END DO

!  external evaluation of elements and groups

     ELSE IF ( run == 8 ) THEN
       ELDERS( 2, : ) = (/ 0, 3, 4 /)
       DO
         CALL LANCELOT_solve(                                                  &
             prob, RANGE1, GVALUE, FVALUE, XT, FUVALS, lfuval, ICALCF, ICALCG, &
             IVAR, Q, DGRAD , control, info, data,                             &
             ELDERS = ELDERS )
         IF ( info%status >= 0 ) EXIT
         IF ( info%status == - 1 .OR. info%status == - 3 .OR.                  &
              info%status == - 7 ) THEN
           CALL ELFUN1_flexible(                                               &
               FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,              &
               prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,&
               ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,&
               prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval,     &
               prob%n, prob%ISTEPA( prob%nel + 1 ) - 1, prob%nel, 1,           &
               ELDERS, istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 1 .OR. info%status == - 5 .OR.                  &
              info%status == - 6 ) THEN
           CALL ELFUN1_flexible(                                               &
               FUVALS, XT, prob%EPVALU, info%ncalcf, prob%ITYPEE,              &
               prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH, prob%ISTEPA,&
               ICALCF, prob%nel, prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1,&
               prob%nel + 1, prob%nel + 1, prob%nel + 1, prob%nel, lfuval,     &
               prob%n, prob%ISTEPA( prob%nel + 1 ) - 1, prob%nel, 3,           &
               ELDERS, istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 2 .OR. info%status == - 4 ) THEN
           CALL GROUP1( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .FALSE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
         IF ( info%status == - 2 .OR. info%status == - 5 ) THEN
           CALL GROUP1( GVALUE, prob%ng, FVALUE, prob%GPVALU, info%ncalcg,     &
               prob%ITYPEG, prob%ISTGPA, ICALCG, prob%ng, prob%ng + 1, prob%ng,&
               prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .TRUE., istat )
           IF ( istat /= 0 ) THEN ; info%status = - 11 ; CYCLE ; END IF
         END IF
       END DO
     END IF

! act on return status

     IF ( info%status == 0 ) THEN                  !  Successful return
       WRITE( 6, "( ' run ', I2, I6, ' iterations. Optimal objective value =', &
      &       ES12.4 )" ) run, info%iter, info%obj
     ELSE                                          !  Error returns
       WRITE( 6, "( ' run ', I2, ' LANCELOT_solve exit status = ', I6 ) " )    &
         run, info%status
     END IF
     CALL LANCELOT_terminate( data, control, info ) !delete internal workspace
     DEALLOCATE( FUVALS, STAT = alloc_status )
   END DO

   END DO

   END PROGRAM LANCELOT_test_deck

