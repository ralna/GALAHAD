! THIS VERSION: GALAHAD 2.6 - 23/06/2013 AT 13:00 GMT.

!-*-*-*-*-*-*-*-  L A N C E L O T  -B-   PRECN   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   ( based on Conn-Gould-Toint fortran 77 version LANCELOT A, ~1992 )
!   originally released pre GALAHAD Version 1.0. February 3rd 1995
!   update released with GALAHAD Version 2.0. February 16th 2005

   MODULE LANCELOT_PRECN_double

!   USE GLOBAL_ma27e,ONLY:SA%INFO(5),SA%INFO(6),SA%INFO(7),SA%INFO(9),SA%INFO(2)
     USE LANCELOT_EXTEND_double, ONLY: EXTEND_arrays
!NOT95USE GALAHAD_CPU_time
     USE LANCELOT_BAND_double
     USE GALAHAD_SMT_double
     USE GALAHAD_SILS_double
     USE GALAHAD_SCU_double, ONLY : SCU_matrix_type, SCU_data_type,            &
       SCU_info_type, SCU_restart_m_eq_0, SCU_solve, SCU_append
     USE LANCELOT_HSL_routines, ONLY : MA61_initialize
     USE LANCELOT_ASMBL_double
     USE LANCELOT_MDCHL_double
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: PRECN_save_type, PRECN_use_preconditioner

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     INTEGER, PARAMETER :: liwmin = 1, lwmin = 1
     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp

!  ================================
!  The PRECN_save_type derived type
!  ================================

     TYPE :: PRECN_save_type
       INTEGER :: liw, lw, nsemiw, nupdat, liccgg, nextra, nz01, iaj
       REAL ( KIND = KIND( 1.0E0 ) ) :: tfactr, t1stsl, tupdat, tsolve
       INTEGER :: ICNTL_iccg( 5 ), KEEP_iccg( 12 ), INFO_iccg( 10 )
       REAL ( KIND = wp ) :: CNTL_iccg( 3 )
     END TYPE PRECN_save_type

   CONTAINS

!-*-  L A N C E L O T  -B-  PRECN_use_preconditioner S U B R O U T I N E -*-*

     SUBROUTINE PRECN_use_preconditioner(                                      &
         ifactr, munks, use_band, seprec, icfs, n, ng, nel, ntotel, nnza,      &
         maxsel, nadd, nvargp, nfree, nfixed, buffer, refact, nvar, IVAR,      &
         ISTADH, ICNA, ISTADA, INTVAR, IELVAR, nvrels, IELING, ISTADG, ISTAEV, &
         IFREE, A, GUVALS, lnguvl, HUVALS, lnhuvl, GVALS2, GVALS3, GRAD, Q,    &
         GSCALE, ESCALE, GXEQX, INTREP, RANGE , icfact, ciccg, nsemib, ratio,  &
         iprint, error , out  , status, alloc_status, bad_alloc,               &
         ITYPEE, DIAG, OFFDIA, IW, IKEEP, IW1, IVUSE,                          &
         H_col_ptr, L_col_ptr, W, W1, RHS, RHS2, P2, G, ISTAGV, ISVGRP,        &
         lirnh, ljcnh, lh, lirnh_min, ljcnh_min, lh_min,                       &
         ROW_start, POS_in_H, USED, FILLED, lrowst, lpos, lused, lfilled,      &
         IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                               &
         matrix, SILS_data, SILS_cntl, SILS_infoa, SILS_infof, SILS_infos,     &
         S, SCU_matrix, SCU_data, SCU_info, SA, skipg, KNDOFG )

!  Form the product Q = M ** -1 GRAD, where M is a preconditioner for the
!  linear system H * x = b and H is the Hessian of a group partially separable
!  function

!  This is achieved in three stages:
!  1) Assemble the matrix H.
!  2) Find and factorize the positive definite matrix M.
!  3) Solve M * Q = GRAD

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nel, maxsel, nvar  , nvrels, nsemib
     INTEGER, INTENT( IN ) :: ng    , lnguvl, lnhuvl, ntotel, nvargp, error
     INTEGER, INTENT( IN ) :: iprint, out   , nnza  , nadd  , icfact, buffer
     INTEGER, INTENT( OUT ) :: status, alloc_status
     INTEGER, INTENT( INOUT ) :: ifactr, nfixed, nfree
     REAL ( KIND = wp ), INTENT( INOUT ) :: ciccg
     REAL ( KIND = wp ), INTENT( OUT ) :: ratio
     LOGICAL, INTENT( IN ) ::  munks, use_band, seprec, icfs, skipg
     LOGICAL, INTENT( OUT ) ::  refact
     CHARACTER ( LEN = 80 ), INTENT( OUT ) :: bad_alloc
     INTEGER, INTENT( IN ), DIMENSION( n ) :: IVAR
     INTEGER, INTENT( IN ), DIMENSION( nnza ) :: ICNA
     INTEGER, INTENT( IN ), DIMENSION( ng  + 1 ) :: ISTADA, ISTADG
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: INTVAR, ISTAEV, ISTADH
     INTEGER, INTENT( IN ), DIMENSION( ntotel ) :: IELING
     INTEGER, INTENT( IN ), DIMENSION( nvrels ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ITYPEE
     INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IFREE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nnza ) :: A
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lnguvl ) :: GUVALS
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lnhuvl ) :: HUVALS
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GVALS2
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GVALS3
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: GRAD
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GSCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ntotel ) :: ESCALE
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: Q
     LOGICAL, INTENT( IN ), DIMENSION( ng  ) :: GXEQX
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     TYPE ( PRECN_save_type ), INTENT( INOUT ) :: S
     TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_matrix
     TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
     TYPE ( SCU_info_type ), INTENT( INOUT ) :: SCU_info
     TYPE ( SMT_type ), INTENT( INOUT ) :: matrix
     TYPE ( SILS_factors ), INTENT( INOUT ) :: SILS_data
     TYPE ( SILS_control ), INTENT( INOUT ) :: SILS_cntl
     TYPE ( SILS_ainfo ), INTENT( INOUT ) :: SILS_infoa
     TYPE ( SILS_finfo ), INTENT( INOUT ) :: SILS_infof
     TYPE ( SILS_sinfo ), INTENT( INOUT ) :: SILS_infos
!    TYPE ( ASSL_save_type ), INTENT( INOUT ) :: S_ASSL
     TYPE ( ASMBL_save_type ), INTENT( INOUT ) :: SA

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG

!---------------------------------------------------------------
!   D u m m y   A r g u m e n t s   f o r   W o r k s p a c e
!--------------------------------------------------------------

     INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: IKEEP, IW1
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, IVUSE
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_col_ptr, L_col_ptr
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) ::                        &
       W, RHS, RHS2, P2, G , DIAG
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: W1, OFFDIA

     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV

     INTEGER, INTENT( INOUT ) :: lirnh, ljcnh, lh, lirnh_min, ljcnh_min, lh_min
     INTEGER, INTENT( INOUT ) :: lrowst, lpos, lused, lfilled
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROW_start
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: POS_in_H
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: USED
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: FILLED
     INTEGER, INTENT( OUT ), DIMENSION( : ) :: IW_asmbl
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_ws
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: H_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: H_in

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,       &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: neg1, scu_status, i, neg2, maxsbw
     INTEGER :: ndiag, max_sc, ntotal, j, band_status
     INTEGER :: nnzh, mlh, ulh, iai, nz0, nlh, n_stat
     REAL ( KIND = KIND( 1.0E0 ) ) :: tt, t , time
     REAL ( KIND = wp ) :: pertur
     LOGICAL :: prnter, reallocate
     CHARACTER ( LEN = 60 ) :: task

!-----------------------------------------------
!   A l l o c a t a b l e   A r r a y s
!-----------------------------------------------

!    pronel = iprint == 2 .AND. out > 0
     prnter = iprint >= 5 .AND. out > 0
     max_sc = SCU_matrix%m_max

!  -----------------------------------------
!  Stage 1a - form H. This stage needs only
!  be performed when the integer ifactr is 1
!  -----------------------------------------

 100 CONTINUE
     IF ( ifactr == 1 ) THEN
       IF ( iprint >= 200 .AND. out > 0 ) WRITE( out, 2210 )
       CALL CPU_TIME( t )
       S%nupdat = 0
       status = 0

!  Define the integer work space needed for ASMBL_assemble_hessian. Ensure
!  that there is sufficient space

       IF ( .NOT. use_band ) THEN
         lirnh = MAX( lirnh, lirnh_min )
         ljcnh = MAX( ljcnh, ljcnh_min ) ; lh = MAX( lh, lh_min )
       END IF

!  Assemble the Hessian restricted to the variables IVAR( I ), I = 1,.., NVAR.
!  Remove the nonzeros which lie outside a band with semi-bandwidth nsemib

       CALL CPU_TIME( tt )
       IF ( use_band ) THEN
         S%nsemiw = MIN( nsemib, MAX( nvar - 1, 0 ) )

!  Allocate space to hold the band matrix

         reallocate = .TRUE.
         IF ( ALLOCATED( DIAG ) ) THEN
           IF ( SIZE( DIAG ) < nvar ) THEN ; DEALLOCATE( DIAG )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( DIAG( nvar ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'DIAG' ; GO TO 980
           END IF
         END IF

         reallocate = .TRUE.
         IF ( ALLOCATED( OFFDIA ) ) THEN
           IF ( SIZE( OFFDIA, 1 ) /= S%nsemiw .OR.                             &
                SIZE( OFFDIA, 2 ) < nvar ) THEN ; DEALLOCATE( OFFDIA )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( OFFDIA( S%nsemiw, nvar ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'OFFDIA'; GO TO 980
           END IF
         END IF

         CALL ASMBL_assemble_hessian(                                          &
             n, ng, nel, ntotel, nvrels, nnza, maxsel, nvargp, nvar, IVAR,     &
             ISTADH, ICNA, ISTADA, INTVAR, IELVAR, IELING, ISTADG,             &
             ISTAEV, ISTAGV, ISVGRP, A, GUVALS, lnguvl, HUVALS, lnhuvl,        &
             GVALS2, GVALS3, GSCALE, ESCALE, GXEQX, ITYPEE, INTREP, RANGE,     &
             iprint, error, out, use_band, .TRUE., .FALSE.,                    &
             S%nsemiw, status, alloc_status, bad_alloc,                        &
             lirnh, ljcnh, lh, matrix%row, matrix%col, matrix%val,             &
             ROW_start, POS_in_H, USED, FILLED, lrowst, lpos, lused, lfilled,  &
             IW_asmbl, W_ws, W_el, W_in, H_el, H_in, skipg,                    &
             maxsbw = maxsbw, DIAG = DIAG, OFFDIA = OFFDIA, KNDOFG = KNDOFG )
       ELSE

!  Set starting addresses for partitions of the integer workspace

         S%nsemiw = nsemib
         CALL ASMBL_assemble_hessian(                                          &
             n, ng, nel, ntotel, nvrels, nnza, maxsel, nvargp, nvar, IVAR,     &
             ISTADH, ICNA, ISTADA, INTVAR, IELVAR, IELING, ISTADG,             &
             ISTAEV, ISTAGV, ISVGRP, A, GUVALS, lnguvl, HUVALS, lnhuvl,        &
             GVALS2, GVALS3, GSCALE, ESCALE, GXEQX, ITYPEE, INTREP, RANGE,     &
             iprint, error, out, use_band, .TRUE., .FALSE.,                    &
             S%nsemiw, status, alloc_status, bad_alloc,                        &
             lirnh, ljcnh, lh, matrix%row, matrix%col, matrix%val,             &
             ROW_start, POS_in_H, USED, FILLED, lrowst, lpos, lused, lfilled,  &
             IW_asmbl, W_ws, W_el, W_in, H_el, H_in, skipg,                    &
             nnzh = nnzh, KNDOFG = KNDOFG )
       END IF
       IF ( iprint >= 200 .AND. out > 0 ) THEN
         CALL CPU_TIME( time ) ; WRITE( out, 2220 ) time - tt
       END IF

!  Check that there is sufficient integer workspace

       IF ( status /= 0 ) RETURN

!  ------------------------------------------------------
!  Stage 2 - Form and factorize M. This stage needs only
!  be performed when the integer ifactr is 1.
!  ------------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Munksgaard's preconditioner, ICCG, is to be used
!  - - - - - - - - - - - - - - - - - - - - - - - - - - -

       IF ( munks ) THEN

!  Initialize data for ICCG

         CALL MA61_initialize( S%ICNTL_iccg, S%CNTL_iccg, S%KEEP_iccg )
         IF ( iprint >= 1000 ) THEN
           S%ICNTL_iccg( 1 ) = 6 ; S%ICNTL_iccg( 2 ) = 6
         ELSE
           S%ICNTL_iccg( 1 ) = 0 ; S%ICNTL_iccg( 2 ) = 0
         END IF

!  Decide how much room is available for the incomplete factorization.
!  nextra gives the amount of workspace above the minimum required
!  for the factorization which is to be allowed for fill-in

         S%nextra = nnzh
         iai = nnzh + S%nextra
         S%iaj = 2 * nnzh + S%nextra

!  Extend the arrays matrix%row, %col and %val to accomodate this extra room

         IF ( iai > lirnh ) THEN
           nlh = iai ; ulh = nnzh; mlh = nnzh
           CALL EXTEND_arrays( matrix%row, lirnh, ulh, nlh, mlh, buffer,       &
                               status, alloc_status )
           IF ( status /= 0 ) THEN
             bad_alloc = 'matrix%row' ; GO TO 990 ; END IF
           lirnh = nlh ; iai = lirnh
         END IF
         IF ( S%iaj > ljcnh ) THEN
           nlh = S%iaj ; ulh = nnzh; mlh = 2 * nnzh
           CALL EXTEND_arrays( matrix%col, ljcnh, ulh, nlh, mlh, buffer,       &
                               status, alloc_status )
           IF ( status /= 0 ) THEN
             bad_alloc = 'matrix%col' ; GO TO 990 ; END IF
           ljcnh = nlh
         END IF
         IF ( S%iaj > lh ) THEN
           nlh = S%iaj ; ulh = nnzh; mlh = 2 * nnzh
           CALL EXTEND_arrays( matrix%val, lh, ulh, nlh, mlh, buffer,          &
                               status, alloc_status )
           IF ( status /= 0 ) THEN ; bad_alloc = 'H' ; GO TO 990 ; END IF
           lh = nlh ; S%iaj = MIN( ljcnh, lh )
         END IF

!  Allocate workspace arrays for MA61

         reallocate = .TRUE.
         IF ( ALLOCATED( IKEEP ) ) THEN
           IF ( SIZE( IKEEP, 1 ) /= nvar .OR. SIZE( IKEEP, 2 ) < 4 ) THEN
             DEALLOCATE( IKEEP ) ; ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( IKEEP( nvar, 4 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'IKEEP' ; GO TO 980; END IF
         END IF

         reallocate = .TRUE.
         IF ( ALLOCATED( IW1 ) ) THEN
           IF ( SIZE( IW1, 1 ) /= nvar .OR. SIZE( IW1, 2 ) < 4 ) THEN
             DEALLOCATE( IW1 ) ; ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( IW1( nvar, 4 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'IW1' ; GO TO 980 ; END IF
         END IF

         reallocate = .TRUE.
         IF ( ALLOCATED( W1 ) ) THEN
           IF ( SIZE( W1, 1 ) /= nvar .OR. SIZE( W1, 2 ) < 3 ) THEN
             DEALLOCATE( W1 ) ; ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( W1( nvar, 3 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'W1' ; GO TO 980 ; END IF
         END IF

!  Form and factorize Munksgaard's preconditioner

         CALL MDCHL_iccga( nvar, nnzh, matrix%val, matrix%row, matrix%col,     &
                           iai, S%iaj, IKEEP, IW1, W1, ciccg,                  &
                           S%ICNTL_iccg, S%CNTL_iccg, S%INFO_iccg, S%KEEP_iccg )
         IF ( prnter .OR. out > 0 .AND. iprint == 2 ) THEN
           IF ( prnter ) THEN
             WRITE( out, 2140 ) nvar, nnzh, S%INFO_iccg( 2 )
           ELSE
             WRITE( out, 2150 ) nvar, nnzh, S%INFO_iccg( 2 )
           END IF
         END IF
         IF ( ( S%INFO_iccg( 1 ) < 0 .AND. out > 0 ) .OR.                      &
              ( S%INFO_iccg( 1 ) > 0 .AND. prnter ) )                          &
           WRITE( out, 2160 ) S%INFO_iccg( 1 )

!  Compress the vector IW to remove unused locations

         nz0 = nnzh - S%INFO_iccg( 5 )
         S%nz01 = nz0 + 1
         S%liccgg = S%iaj - nz0

!  Record the relative fill-in

         IF ( nnzh > 0 ) THEN
           ratio = DBLE( FLOAT( S%liccgg ) ) / DBLE( FLOAT( nnzh ) )
         ELSE
           ratio = one
         END IF

!  - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Lin and More's preconditioner, ICFS, is to be used
!  - - - - - - - - - - - - - - - - - - - - - - - - - -

       ELSE IF ( icfs ) THEN

!  Allocate workspace arrays for ICFS

         reallocate = .TRUE.
         IF ( ALLOCATED( IW ) ) THEN
            IF ( SIZE( IW ) < 3 * nvar ) THEN ; DEALLOCATE( IW )
            ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( IW( 3 * nvar ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'IW' ; GO TO 980 ; END IF
         END IF

         reallocate = .TRUE.
         IF ( ALLOCATED( H_col_ptr ) ) THEN
           IF ( SIZE( H_col_ptr ) < nvar + 1 ) THEN
             DEALLOCATE( H_col_ptr )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( H_col_ptr( nvar + 1 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'H_col_ptr' ; GO TO 980 ; END IF
         END IF

         reallocate = .TRUE.
         IF ( ALLOCATED( L_col_ptr ) ) THEN
           IF ( SIZE( L_col_ptr ) < nvar + 1 ) THEN
             DEALLOCATE( L_col_ptr )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( L_col_ptr( nvar + 1 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'L_col_ptr' ; GO TO 980 ; END IF
         END IF

         reallocate = .TRUE.
         IF ( ALLOCATED( DIAG ) ) THEN
           IF ( SIZE( DIAG ) < nvar ) THEN
             DEALLOCATE( DIAG, STAT = alloc_status )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( DIAG( nvar ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'DIAG' ; GO TO 980 ; END IF
         END IF

         reallocate = .TRUE.
         IF ( ALLOCATED( W ) ) THEN
           IF ( SIZE( W ) < nvar ) THEN
             DEALLOCATE( W, STAT = alloc_status )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN
           ALLOCATE( W( nvar ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'W' ; GO TO 980 ; END IF
         END IF

!  Check to see if H contains a diagonal entry in each row

         IW( : nvar ) = 0
         DO j = 1, nnzh
          i = matrix%row( j )
          IF ( i == matrix%col( j ) ) IW( i ) = IW( i ) + 1
         END DO
         ndiag = COUNT( IW( : nvar ) == 0 )
         nz0 = nnzh + ndiag

!  Decide how much room is available for the incomplete factorization.
!  nextra gives the amount of workspace above the minimum required
!  for the factorization which is to be allowed for fill-in

         S%nextra = nz0 + nvar * icfact
         ntotal = nz0 + S%nextra

!  Extend the arrays matrix%row, %col and val to accomodate this extra room

         IF ( ntotal > lirnh ) THEN
           mlh = ntotal ; nlh = MAX( mlh, ( 3 * lirnh ) / 2 ) ; ulh = nnzh
           CALL EXTEND_arrays( matrix%row, lirnh, ulh, nlh, mlh, buffer,       &
                               status, alloc_status )
           IF ( status /= 0 ) THEN ; bad_alloc = 'matrix%row' ; GO TO 990
           END IF
           lirnh = nlh
         END IF
         IF ( ntotal > lh ) THEN
           mlh = ntotal ; nlh = MAX( mlh, ( 3 * lh ) / 2 ) ; ulh = nnzh
           CALL EXTEND_arrays( matrix%val, lh, ulh, nlh, mlh, buffer,          &
                               status, alloc_status )
           IF ( status /= 0 ) THEN ; bad_alloc = 'H' ; GO TO 990
           END IF
           lh = nlh
         END IF

!  Reorder H so that its lower triangle is stored in compressed column format.
!  First count how many nonzeros there are in each column

         IW( : nvar ) = 0
         DO j = 1, nnzh
          i = matrix%row( j )
          IF ( i /= matrix%col( j ) ) IW( i ) = IW( i ) + 1
         END DO

!  Now find the starting address for each columm in the storage format

         H_col_ptr( 1 ) = S%nextra + nvar + 1
         DO i = 1, nvar
           H_col_ptr( i + 1 ) = H_col_ptr( i ) + IW( i )
         END DO

!  Finally copy the data into its correct position ...

         DO j = 1, nnzh
           i = matrix%row( j )
           IF ( i /= matrix%col( j ) ) THEN  ! off-diagonal term
             matrix%val( H_col_ptr( i ) ) = matrix%val( j )
             matrix%row( H_col_ptr( i ) ) = matrix%col( j )
             H_col_ptr( i ) = H_col_ptr( i ) + 1
           ELSE                        ! diagonal term
             matrix%val( S%nextra + i ) = matrix%val( j )
           END IF
         END DO

!   ... and reposition the starting addresses

         H_col_ptr( 1 ) = 1
         DO i = 1, nvar
           H_col_ptr( i + 1 ) = H_col_ptr( i ) + IW( i )
         END DO

!  Form and factorize Lin and More's preconditioner

         pertur = zero
         n_stat = nvar
         IF ( iprint > 0 .AND. out >= 0 ) THEN
           IW( 1 ) = out
         ELSE
           IW( 1 ) = - 1
         END IF
         CALL DICFS( n_stat, nz0 - nvar,                                       &
                     matrix%val( S%nextra + nvar + 1 : ntotal ),               &
                     matrix%val( S%nextra + 1 : S%nextra + nvar ),             &
                     H_col_ptr, matrix%row( S%nextra + nvar + 1 : ntotal ),    &
                     matrix%val( nvar + 1 : S%nextra ), matrix%val( : nvar ),  &
                     L_col_ptr, matrix%row( nvar + 1 : S%nextra ),             &
                     icfact, pertur, IW, DIAG, W )
         IF ( n_stat < 0 ) THEN ; status = - 26 ; RETURN ; END IF

!  - - - - - - - - - - - - - - - - - -
!  A band preconditioner is to be used
!  - - - - - - - - - - - - - - - - - -

       ELSE

!  Factorize the band matrix

         IF ( use_band ) THEN
           CALL BAND_factor( nvar, S%nsemiw, DIAG, OFFDIA, S%nsemiw,           &
                             band_status )
           IF ( prnter .OR. iprint == 2 ) THEN
             IF ( prnter ) THEN
               WRITE( out, 2120 ) nvar, S%nsemiw, maxsbw
             ELSE
               WRITE( out, 2130 ) nvar, S%nsemiw, maxsbw
             END IF
           END IF
         ELSE

!  - - - - - - - - - - - - - - - - - - - - - - - - - -
!  A multi-frontal preconditioner, SILS, is to be used
!  - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Allocate the arrays for the analysis phase

           S%liw = INT( 1.2 * REAL( 2 * nnzh + 3 * nvar + 1, KIND = wp ) )

!  Choose initial values for the control parameters

           IF ( iprint >= 1000 ) THEN
             SILS_cntl%lp = 6 ; SILS_cntl%mp = 6
             SILS_cntl%wp = 6 ; SILS_cntl%sp = 6
             SILS_cntl%ldiag = 1
           ELSE
             SILS_cntl%lp = 0 ; SILS_cntl%mp = 0
             SILS_cntl%wp = 0 ; SILS_cntl%sp = 0
             SILS_cntl%ldiag = 0
           END IF

!  Choose the pivot sequence for the factorization by analyzing the
!  sparsity pattern of M

           CALL CPU_TIME( tt )
           matrix%n = nvar
           matrix%ne = nnzh
           IF ( seprec ) SILS_cntl%pivoting = 4
           CALL SILS_analyse( matrix, SILS_data, SILS_cntl, SILS_infoa )
           IF ( iprint >= 200 .AND. out > 0 ) THEN
             CALL CPU_TIME( time )
             WRITE( out, 2230 ) time - tt
           END IF

!  Define the work space for the factorization and the solver

           IF ( seprec ) THEN

             reallocate = .TRUE.
             IF ( ALLOCATED( DIAG ) ) THEN
               IF ( SIZE( DIAG ) < nvar ) THEN ; DEALLOCATE( DIAG )
               ELSE ; reallocate = .FALSE. ; END IF
             END IF
             IF ( reallocate ) THEN
               ALLOCATE( DIAG( nvar ), STAT = alloc_status )
               IF ( alloc_status /= 0 ) THEN
                  bad_alloc = 'DIAG' ; GO TO 980 ; END IF
             END IF
           ELSE
             reallocate = .TRUE.
             IF ( ALLOCATED( IW ) ) THEN
                IF ( SIZE( IW ) < nvar ) THEN ; DEALLOCATE( IW )
                ELSE ; reallocate = .FALSE. ; END IF
             END IF
             IF ( reallocate ) THEN
               ALLOCATE( IW( nvar ), STAT = alloc_status )
               IF ( alloc_status /= 0 ) THEN
                 bad_alloc = 'IW' ; GO TO 980 ; END IF
             END IF

             reallocate = .TRUE.
             IF ( ALLOCATED( OFFDIA ) ) THEN
               IF ( SIZE( OFFDIA, 1 ) /= 2 .OR.                                &
                    SIZE( OFFDIA, 2 ) < nvar ) THEN ; DEALLOCATE( OFFDIA )
               ELSE ; reallocate = .FALSE. ; END IF
             END IF
             IF ( reallocate ) THEN
               ALLOCATE( OFFDIA( 2, nvar ), STAT = alloc_status )
               IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'OFFDIA'; GO TO 980
               END IF
             END IF
           END IF

!  Factorize the matrix M, using the Schnabel-Eskow modified Cholesky method
!  or the Gill-Murray-Ponceleon-Saunders modification to the symmetric
!  indefinite factorization

           CALL CPU_TIME( tt )

           CALL SILS_factorize( matrix, SILS_data, SILS_cntl, SILS_infof )
           IF ( iprint >= 200 .AND. out > 0 ) THEN
             CALL CPU_TIME( time )
             WRITE( out, 2240 ) time - tt
           END IF

!  Test that the factorization succeeded

           IF ( SILS_infof%flag < 0 ) RETURN

           IF ( seprec ) THEN

!  Calculate the maximum perturbation made to the diagonals of H

             IF ( prnter .OR. iprint == 2 ) THEN
               IF ( prnter ) THEN
                 WRITE( out, 2060 ) SILS_infof%maxchange, nvar, nnzh,          &
                                     SILS_infoa%nrladu, SILS_infof%nrlbdu
               ELSE
                 WRITE( out, 2070 ) nvar, nnzh, SILS_infof%maxchange,          &
                                     SILS_infoa%nrladu, SILS_infof%nrlbdu
               END IF
             END IF
           ELSE

             CALL CPU_TIME( tt )

!  Modify the pivots

             CALL MDCHL_gmps( nvar, SILS_infof%rank, SILS_data, neg1, neg2,    &
                              IW, OFFDIA )
             CALL CPU_TIME( time )
             IF ( iprint >= 200 .AND. out > 0 )                                &
               WRITE( out, 2250 ) time - tt
             IF ( prnter .OR. iprint == 2 ) THEN
                IF ( prnter ) THEN
                   WRITE( out, 2090 ) nvar, nnzh, SILS_infoa%nrladu,           &
                                       SILS_infof%nrlbdu, neg1, neg2
                ELSE
                   WRITE( out, 2100 ) nvar, nnzh, SILS_infoa%nrladu,           &
                                       SILS_infof%nrlbdu, neg1, neg2
                END IF
             END IF
           END IF

!  Record the relative fill-in

           IF ( nnzh > 0 ) THEN
              ratio = DBLE( FLOAT( SILS_infof%nrlbdu ) ) / DBLE( FLOAT( nnzh ) )
           ELSE
              ratio = one
           END IF
         END IF

!  - - - - - - - - - - - - - - -
!  The factorization is complete
!  - - - - - - - - - - - - - - -

       END IF

!  Store the number of free variables, nfree, and list of free variables when
!  the factorization is performed in case they they are needed on a subsequent
!  entry

       nfixed = 0
       nfree = nvar
       IFREE( : nvar ) = IVAR( : nvar )

!  Allocate further workspace

!      S%lrmax = max_sc * ( max_sc + 1 ) / 2

       reallocate = .TRUE.
       IF ( ALLOCATED( IVUSE ) ) THEN
         IF ( SIZE( IVUSE ) < n ) THEN ; DEALLOCATE( IVUSE )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN
         ALLOCATE( IVUSE( n ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN
           bad_alloc = 'IVUSE' ; GO TO 980 ; END IF
       END IF

       reallocate = .TRUE.
       IF ( ALLOCATED( RHS ) ) THEN
         IF ( SIZE( RHS ) < nfree ) THEN ; DEALLOCATE( RHS )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN
         ALLOCATE( RHS( nfree ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'RHS' ; GO TO 980 ; END IF
       END IF

       SCU_matrix%n = nfree
       SCU_matrix%m = nfixed
       CALL SCU_restart_m_eq_0( SCU_data, SCU_info )

       reallocate = .TRUE.
       IF ( ALLOCATED( RHS2 ) ) THEN
         IF ( SIZE( RHS2 ) < nfree + max_sc ) THEN ; DEALLOCATE( RHS2 )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN
         ALLOCATE( RHS2( nfree + max_sc ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'RHS2' ; GO TO 980 ; END IF
       END IF

       reallocate = .TRUE.
       IF ( ALLOCATED( P2 ) ) THEN
         IF ( SIZE( P2 ) < nfree + max_sc ) THEN ; DEALLOCATE( P2 )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN
         ALLOCATE( P2( nfree + max_sc ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'P2' ; GO TO 980 ; END IF
       END IF

       reallocate = .TRUE.
       IF ( ALLOCATED( G ) ) THEN
         IF ( SIZE( G ) < n ) THEN ; DEALLOCATE( G )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN
         ALLOCATE( G( n ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'G' ; GO TO 980 ; END IF
       END IF

!  Record the time taken to assemble and factorize the preconditioner

       CALL CPU_TIME( time )
       S%tfactr = time - t ; S%tupdat = 0.0 ; S%tsolve = 0.0
     END IF

!  ---------------------------------------------------------------------
!  Stage 2 (ALTERNATIVE) - Update the Schur complement of M. This stage
!  needs only be performed when the integer ifactr is 2
!  ---------------------------------------------------------------------

     IF ( ifactr == 2 ) THEN
       IF ( iprint >= 200 .AND. out > 0 ) WRITE( out, 2260 )

!  Refactorize the remaining coefficient matrix if the last update and solve
!  took longer than the previous refactorization and subsequent solve

       S%nupdat = S%nupdat + nadd
       IF ( S%nupdat > max_sc ) THEN
         ifactr = 1
         IF ( iprint >= 2 .AND. out > 0 ) WRITE( out, 2170 )                   &
           S%nupdat, max_sc, S%tupdat + S%tsolve, S%tfactr + S%t1stsl
         refact = .TRUE.
         GO TO 100
       END IF

       CALL CPU_TIME( t )

!  Record the variables which are still free by first setting the appropriate
!  partition of IW to zero, and then resetting those components of
!  IW which correspond to the free variables to one

       IVUSE( : n ) = 0 ; IVUSE( IVAR( : nvar ) ) = 1

!  Compare this with the list of those which were free at the last
!  factorization.  Fix any variables which was free but no longer appears in
!  the list

       DO j = 1, nfree
         i = IFREE( j )
         IF ( i > 0 ) THEN
           IF ( IVUSE( i ) == 0 ) THEN

!  If more than max_sc variables have been fixed, refactorize the matrix

             IF ( nfixed >= max_sc ) THEN
               ifactr = 1
               refact = .TRUE.
               GO TO 100
             END IF

!  Update the factorization of the Schur complement to allow for the removal
!  of the J-th row and column of the original Hessian - this removal is
!  effected by appending the J-th row and column of the identity matrix
!  to the Hessian

             SCU_matrix%BD_val( nfixed + 1 ) = one
             SCU_matrix%BD_row( nfixed + 1 ) = j
             SCU_matrix%BD_col_start( nfixed + 2 ) = nfixed + 2
             scu_status = 1
 230         CONTINUE

!  Call SCU_append to update the Schur-complement

             CALL CPU_TIME( tt )
             CALL SCU_append( SCU_matrix, SCU_data, RHS, scu_status, SCU_info )

             IF ( iprint >= 200 .AND. out > 0 ) THEN
               CALL CPU_TIME( time ) ; WRITE( out, 2270 ) time - tt
             END IF

             IF ( scu_status > 0 ) THEN

!  SCU_append requires additional information. Compute the solution to
!  the equation H * S = RHS, returning the solution S in RHS

!  For Munskgaard's factorization

               IF ( munks ) THEN
                 CALL MDCHL_iccgb( nfree, matrix%val( S%nz01 : S%iaj ),        &
                                   matrix%col( S%nz01 : S%iaj ), S%liccgg,     &
                                   W1( : nfree, : 2 ),                         &
                                   IKEEP( : nfree, : 2 ), RHS, S%INFO_iccg,    &
                                   S%KEEP_iccg )

!  For Lin and More's factorization

               ELSE IF ( icfs ) THEN
                 task = 'N'
                 n_stat = nfree
                 CALL DSTRSOL( n_stat, matrix%val( nfree + 1 : S%nextra ),     &
                               matrix%val( : nfree ), L_col_ptr,               &
                               matrix%row( nfree + 1 : S%nextra ), RHS, task )
                 IF ( n_stat < 0 ) THEN ; status = - 26 ; RETURN ; END IF
                 task = 'T'
                 CALL DSTRSOL( n_stat, matrix%val( nfree + 1 : S%nextra ),     &
                               matrix%val( : nfree ), L_col_ptr,               &
                               matrix%row( nfree + 1 : S%nextra ), RHS, task )
                 IF ( n_stat < 0 ) THEN ; status = - 26 ; RETURN ; END IF

!  For the band factorization

               ELSE
                 IF ( use_band ) THEN
                   CALL BAND_solve( nfree, S%nsemiw, DIAG, OFFDIA, S%nsemiw,   &
                                    RHS, band_status )
                 ELSE

!  For the multifrontal factorization

                   CALL CPU_TIME( tt )
                   CALL SILS_solve( matrix, SILS_data, RHS, SILS_cntl,         &
                                    SILS_infos )
                   IF ( iprint >= 200 .AND. out > 0 ) THEN
                     CALL CPU_TIME( time ) ; WRITE( out, 2280 ) time - tt
                   END IF
                 END IF
               END IF
               GO TO 230
             END IF

!  If the Schur-complement is numerically indefinite, refactorize
!  the preconditioning matrix to alleviate the effect of rounding

             IF ( scu_status < 0 ) THEN
               WRITE( out, 2050 ) scu_status
               ifactr = 1 ; refact = .TRUE.
               GO TO 100
             END IF

!  Record that the relevant variable is now fixed

             IFREE( j ) = - i
             nfixed = nfixed + 1
           END IF
         END IF
       END DO
       CALL CPU_TIME( time )
       S%tupdat = time - t ; S%tsolve = 0.0
     END IF

!  -----------------------------------------------
!  Stage 3 - solve for the preconditioned gradient
!  -----------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - -
!  Initial solve using the original factorization
!  - - - - - - - - - - - - - - - - - - - - - - - -

!  Put the components of GRAD into RHS

     IF ( iprint >= 200 .AND. out > 0 ) WRITE( out, 2200 )
     CALL CPU_TIME( t )
     IF ( nfixed == 0 ) THEN
       RHS( : nvar ) = GRAD( : nvar )

!  Compute the solution to the equation M * s = RHS, returning the solution
!  s in RHS

!  Using Munskgaard's factorization

       IF ( munks ) THEN
         CALL MDCHL_iccgb( nfree, matrix%val( S%nz01 : S%iaj ),                &
                           matrix%col( S%nz01 : S%iaj ), S%liccgg,             &
                           W1( : nfree, : 2 ), IKEEP( : nfree, : 2 ), RHS,     &
                           S%INFO_iccg, S%KEEP_iccg )

!  For Lin and More's factorization

       ELSE IF ( icfs ) THEN
         task = 'N'
         n_stat = nfree
         CALL DSTRSOL( n_stat, matrix%val( nfree + 1 : S%nextra ),             &
                       matrix%val( : nfree ), L_col_ptr,                       &
                       matrix%row( nfree + 1 : S%nextra ), RHS, task )
         IF ( n_stat < 0 ) THEN ; status = - 26 ; RETURN ; END IF
         task = 'T'
         CALL DSTRSOL( n_stat, matrix%val( nfree + 1 : S%nextra ),             &
                       matrix%val( : nfree ), L_col_ptr,                       &
                       matrix%row( nfree + 1 : S%nextra ), RHS, task )
         IF ( n_stat < 0 ) THEN ; status = - 26 ; RETURN ; END IF

!  For the band factorization

       ELSE
         IF ( use_band ) THEN
           CALL BAND_solve( nfree, S%nsemiw, DIAG, OFFDIA, S%nsemiw, RHS,      &
                            band_status )
         ELSE

!  Using the multifrontal factorization

           CALL CPU_TIME( tt )
           CALL SILS_solve( matrix, SILS_data, RHS, SILS_cntl, SILS_infos )
           IF ( iprint >= 200 .AND. out > 0 ) THEN
             CALL CPU_TIME( time ) ; WRITE( out, 2280 ) time - tt
           END IF
         END IF
       END IF

!  Scatter the free components of the solution into Q

       Q( IVAR( : nvar ) ) = RHS( : nvar )
       CALL CPU_TIME( time )
       S%t1stsl = time - t
     ELSE

!  - - - - - - - - - - - - - - - - - - - - - - - - -
!  Subsequent solves using the original factorization
!  and the factorization of the Schur-complement
!  - - - - - - - - - - - - - - - - - - - - - - - - -

!  Solve for the preconditioned gradient using the Schur complement update.
!  Put the components of GRAD into RHS2

       G( IVAR( : nvar ) ) = GRAD( : nvar )
       DO j = 1, nfree
         i = IFREE( j )
         IF ( i > 0 ) THEN
           RHS2( j ) = G( i )
         ELSE
           RHS2( j ) = zero
         END IF
       END DO
       RHS2( nfree + 1 : nfree + nfixed ) = zero

!  Solve the linear system H * P2 = RHS2

       scu_status = 1
  360  CONTINUE

!  Call SCU_solve to solve the system

       CALL CPU_TIME( tt )
       CALL SCU_solve( SCU_matrix, SCU_data, RHS2, P2, RHS, scu_status )
       IF ( iprint >= 200 .AND. out > 0 ) THEN
         CALL CPU_TIME( time ) ; WRITE( out, 2290 ) time - tt
       END IF
       IF ( scu_status > 0 ) THEN

!  SCU_block_solve requires additional information. Compute the solution to
!  the equation H * S = RHS, returning the solution S in RHS

!  Using Munskgaard's factorization

         IF ( munks ) THEN
           CALL MDCHL_iccgb( nfree, matrix%val( S%nz01 : S%iaj ),              &
                             matrix%col( S%nz01 : S%iaj ), S%liccgg,           &
                             W1( : nfree, : 2 ), IKEEP( : nfree, : 2 ), RHS,   &
                             S%INFO_iccg, S%KEEP_iccg )

!  For Lin and More's factorization

         ELSE IF ( icfs ) THEN
           task = 'N'
           n_stat = nfree
           CALL DSTRSOL( n_stat, matrix%val( nfree + 1 : S%nextra ),           &
                         matrix%val( : nfree ), L_col_ptr,                     &
                         matrix%row( nfree + 1 : S%nextra ), RHS, task )
           IF ( n_stat < 0 ) THEN ; status = - 26 ; RETURN ; END IF
           task = 'T'
           CALL DSTRSOL( n_stat, matrix%val( nfree + 1 : S%nextra ),           &
                         matrix%val( : nfree ), L_col_ptr,                     &
                         matrix%row( nfree + 1 : S%nextra ), RHS, task )
           IF ( n_stat < 0 ) THEN ; status = - 26 ; RETURN ; END IF

!  For the band factorization

         ELSE
           IF ( use_band ) THEN
             CALL BAND_solve( nfree, S%nsemiw, DIAG, OFFDIA, S%nsemiw, RHS,    &
                              band_status )
           ELSE

!  Using the multifrontal factorization

             CALL CPU_TIME( tt )
             CALL SILS_solve( matrix, SILS_data, RHS, SILS_cntl, SILS_infos )
             IF ( iprint >= 200 .AND. out > 0 ) THEN
               CALL CPU_TIME( time ) ; WRITE( out, 2280 ) time - tt
             END IF
           END IF
         END IF
         GO TO 360
       END IF

!  Scatter the free components of the solution into Q

       DO j = 1, nfree
         i = IFREE( j )
         IF ( i > 0 ) Q( i ) = P2( j )
       END DO
       CALL CPU_TIME( time )
       S%tsolve = time - t
       IF ( iprint >= 10 .AND. out > 0 )                                       &
         WRITE( out, 2110 ) S%tupdat + S%tsolve, S%tfactr + S%t1stsl
     END IF

!  Successful return

     status = 0
     RETURN

!  Unsuccessful returns

 980 CONTINUE
     status = 12

 990 CONTINUE
     WRITE( error, 2990 ) alloc_status, TRIM( bad_alloc )
     RETURN

! Non-executable statements

!2040    FORMAT( ' Perturbation ', ES12.4, ' for diagonal ', I6 )
 2050    FORMAT( ' ** Message from -PRECN_use_preconditioner-', /,            &
                '    Value of status after SCU_solve = ', I3 )
 2060    FORMAT( /,' ** Preconditioner: diagonals are perturbed by at most ', &
                 ES12.4, /, '    Order of preconditioner         = ', I8,     &
                 /, '    # nonzeros in preconditioner    = ', I8, /,          &
                    '    Predicted # nonzeros in factors = ', I8, /,          &
                    '    Actual    # nonzeros in factors = ', I8 )
 2070    FORMAT( /, '    -- Preconditioner formed. n = ', I8, /,              &
                    '    -- NNZ  = ', I8, ' Max pert  = ', ES8.1, /,          &
                    '    -- Pred fill = ', I8, ' fill = ', I8 )
 2090    FORMAT( /, ' ** Preconditioner: ',/,                                 &
                    '    order of preconditioner         = ', I8, /,          &
                    '    # nonzeros in preconditioner    = ', I8, /,          &
                    '    Predicted # nonzeros in factors = ', I8, /,          &
                    '    actual    # nonzeros in factors = ', I8, /,          &
                    '    # negative 1 x 1 block pivots   = ', I8, /,          &
                    '    #          2 x 2 block pivots   = ', I8 )
 2100    FORMAT( /, '    -- Preconditioner formed.',/,'    -- n = ', I8,      &
                    ' NNZ  = ', I8, '    Pred fill = ', I8, ' Fill = ', I8,   &
                 /, '    -- # negative 1x1 pivots = ', I8,                    &
                    ' # 2x2 pivots = ', I8 )
 2110    FORMAT( /, ' t( updated ) = ', F7.3, ' vs t( factored ) = ', F7.3 )
 2120    FORMAT( /, ' ** Preconditioner: ', /,                                &
                    '    Order of preconditioner         = ', I8, /,          &
                    '    Semi-bandwidth used             = ', I8, /,          &
                    '    True semi-bandwidth             = ', I8 )
 2130    FORMAT( /, '    -- Preconditioner formed.', /, '    -- n = ', I8,    &
                    ' Semi-bandwidth = ', I8, '    true semi-bandwith = ', I8 )
 2140    FORMAT( /, ' ** Preconditioner: ', /,                                &
                    '    order of preconditioner         = ', I8, /,          &
                    '    # nonzeros in preconditioner    = ', I8, /,          &
                    '    # nonzeros in factors           = ', I8 )
 2150    FORMAT( /, '    -- Preconditioner formed.',/,'    -- n = ', I8,      &
                    ' NNZ  = ', I8,' fill = ', I8 )
 2160    FORMAT( /, '  Warning message from ICCG. INFO( 1 ) = ',I2 )
 2170    FORMAT( /, ' Refactorizing: update ', I6,                            &
                    ' out of an allowed total of ', I6, /,                    &
                    '                Time to update = ', F7.1,                &
                    ' v.s. time to refactorize = ', F7.1 )
 2200    FORMAT( /,' Solve ' )
 2210    FORMAT( /,' Factorization ' )
 2220    FORMAT( /,' time( ASMBL_assemble_hessian ) = ', F10.2 )
 2230    FORMAT( /,' time( SILS_analyse ) = ', F10.2 )
 2240    FORMAT( /,' time( SILS_factorize ) = ', F10.2 )
 2250    FORMAT( /,' time( MDCHL_gmps ) = ', F10.2 )
 2260    FORMAT( /,' Update ' )
 2270    FORMAT( /,' time( SCU_solve ) = ', F10.2 )
 2280    FORMAT( /,' time( SILS_solve ) = ', F10.2 )
 2290    FORMAT( /,' time( SCU_append ) = ', F10.2 )
 2990    FORMAT( ' ** Message from -LANCELOT_PRECN_use_preconditioner-', /,    &
               ' Allocation error (status = ', I0, ') for ', A )

!  End of subroutine PRECN_use_preconditioner

     END SUBROUTINE PRECN_use_preconditioner

!  End of module LANCELOT_PRECN

   END MODULE LANCELOT_PRECN_double

