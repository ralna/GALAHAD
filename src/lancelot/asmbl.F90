! THIS VERSION: GALAHAD 3.3 - 20/05/2021 AT 10:30 GMT.

!-*-*-*-*-*-  L A N C E L O T  -B-  ASMBL  M O D U L E  *-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  January 25th 1995

   MODULE LANCELOT_ASMBL_double

     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPACE_double
     USE GALAHAD_EXTEND_double, ONLY: EXTEND_arrays
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: ASMBL_save_type, ASMBL_assemble_hessian

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp

!  ================================
!  The ASMBL_save_type derived type
!  ================================

     TYPE :: ASMBL_save_type
       LOGICAL :: ptr_status
       INTEGER, DIMENSION( 30 ) :: ICNTL
       INTEGER, DIMENSION( 20 ) :: INFO
       REAL ( KIND = wp ), DIMENSION( 5 ) :: CNTL
     END TYPE ASMBL_save_type

   CONTAINS

!-*-*-  L A N C E L O T  -B-  ASMBL_assemble_hessian  S U B R O U T I N E -*-

     SUBROUTINE ASMBL_assemble_hessian(                                        &
                      n, ng, nel, ntotel, nvrels, nnza, maxsel, nvargp,        &
                      n_free, I_free, ISTADH, ICNA, ISTADA, INTVAR, IELVAR,    &
                      IELING, ISTADG, ISTAEV, ISTAGV, ISVGRP, A, GUVALS,       &
                      lnguvl, HUVALS, lnhuvl, GVALS2, GVALS3, GSCALE, ESCALE,  &
                      GXEQX, ITYPEE, INTREP, RANGE, iprint, error, out,        &
                      use_band, no_zeros, fixed_structure, nsemib, status,     &
                      alloc_status, bad_alloc,                                 &
                      lh_row, lh_col, lh_val, H_row, H_col, H_val, ROW_start,  &
                      POS_in_H, USED, FILLED, lrowst, lpos, lused, lfilled,    &
                      IVAR, GRAD_el, W_el, W_in, H_el, H_in, skipg,        &
                      nnzh, maxsbw, DIAG, OFFDIA, KNDOFG )

!  Assemble the second derivative matrix of a groups partially separable
!  function in either co-ordinate or band format

!  History -
!   ( based on Conn-Gould-Toint fortran 77 version LANCELOT A, ~1992 )
!   fortran 90 version originally released pre GALAHAD Version 1.0. January
!     25th 1995 as ASMBL_assemble_hessian as part of the ASMBL module
!   update released with GALAHAD Version 2.0. February 16th 2005
!   fortran 2003 version released in CUTEst, 5th November 2012
!   completely revised version 14th June 2013

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n, nel, ng, maxsel, nsemib, nvargp, nnza
      INTEGER, INTENT( IN ) :: nvrels, ntotel, n_free
      INTEGER, INTENT( IN ) :: lnguvl, lnhuvl, iprint, error, out
      INTEGER, INTENT( OUT ) :: status, alloc_status
      LOGICAL, INTENT( IN ) :: fixed_structure, no_zeros, use_band, skipg
      CHARACTER ( LEN = 80 ) :: bad_alloc
      INTEGER, INTENT( IN  ), DIMENSION( n ) :: I_free
      INTEGER, INTENT( IN ), DIMENSION( nnza ) :: ICNA
      INTEGER, INTENT( IN ), DIMENSION( ng + 1 ) :: ISTADA, ISTADG, ISTAGV
      INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: INTVAR, ISTAEV, ISTADH
      INTEGER, INTENT( IN ), DIMENSION( nvrels ) :: IELVAR
      INTEGER, INTENT( IN ), DIMENSION( ntotel ) :: IELING
      INTEGER, INTENT( IN ), DIMENSION( nvargp ) :: ISVGRP
      INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nnza ) :: A
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lnguvl ) :: GUVALS
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lnhuvl ) :: HUVALS
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GVALS2
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GVALS3
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GSCALE
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ntotel ) :: ESCALE
      LOGICAL, INTENT( IN ), DIMENSION( ng ) :: GXEQX
      LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP

!---------------------------------------------------------------
!   D u m m y   A r g u m e n t s   f o r   W o r k s p a c e 
!--------------------------------------------------------------

      INTEGER, INTENT( INOUT ) :: lh_row, lh_col, lh_val
      INTEGER, INTENT( INOUT ) :: lrowst, lpos, lused, lfilled
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROW_start
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: POS_in_H
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: USED
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: FILLED
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_col
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val

      INTEGER, INTENT( OUT ), DIMENSION( n ) :: IVAR
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: GRAD_el
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_in
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: H_el
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: H_in

!--------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!--------------------------------------------------

      INTEGER, INTENT( OUT ), OPTIONAL :: maxsbw, nnzh
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL,                             &
                                         DIMENSION( n_free ) :: DIAG
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL,                             &
                                         DIMENSION( nsemib, n_free ) :: OFFDIA
      INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

      INTERFACE
        SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                          lw1, lw2 )
        INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
        LOGICAL, INTENT( IN ) :: transp
        REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN  ), DIMENSION ( lw1 ) :: W1
!       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
        REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
        END SUBROUTINE RANGE
      END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, ii, j, jj, k, kk, ig, l, ijhess, irow, jcol, jcolst, ihnext
      INTEGER :: iel, iell, ielh, nvarel, ig1, listvs, listve, n_filled, nin
      REAL ( KIND = wp ) :: wki, hesnew, gdash, g2dash, scalee
      CHARACTER ( LEN = 2 ), DIMENSION( 36, 36 ) :: MATRIX
      CHARACTER ( LEN = 80 ) :: array_name
!     CHARACTER ( LEN = 80 ) :: array

!  renumber the free variables so that they are variables 1 to n_free

      IVAR( : n ) = 0
      DO i = 1, n_free
        IVAR( I_free( i ) ) = i
      END DO
      IF ( iprint >= 10 ) WRITE( out,  "( /, I0, ' free variables. They are ', &
     &     8I5, /, ( 14I5 ) )" ) n_free, ( I_free( i ), i = 1, n_free )

!  if a band storage scheme is to be used, initialize the entries within the
!  band as zero

      IF ( use_band ) THEN
        maxsbw = 0 ; DIAG = 0.0_wp ; OFFDIA = 0.0_wp

!  if a co-ordinate scheme is to be used, determine the rows structure of the 
!  second derivative matrix of a groups partially separable function with 
!  possible repititions if this has not already been done

      ELSE

!  allocate workspace

        lrowst = n_free + 1
        array_name = 'asmbl: ROW_start'
        CALL SPACE_resize_array( lrowst, ROW_start, status, alloc_status,      &
                array_name = array_name, bad_alloc = bad_alloc, out = error )
        IF ( status /= GALAHAD_ok ) GO TO 980

!  ==================================================
!  PASS 1: compute the numbers of entries in each row
!  ==================================================

!  ROW_start(i+1) will hold the number of entries (with repeats) in row i

        ROW_start( 2 : n_free + 1 ) = 0

!  consider the rank-one second order term for the i-th group

        DO ig = 1, ng
          IF ( skipg ) THEN
            IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
          IF ( GXEQX( ig ) ) CYCLE
          IF ( .NOT. fixed_structure .AND. GSCALE( ig ) == 0.0_wp ) CYCLE
          listvs = ISTAGV( ig )
          listve = ISTAGV( ig + 1 ) - 1

!  Form the j-th column of the rank-one matrix

          DO l = listvs, listve
            j = IVAR( ISVGRP( l ) )
            IF ( j == 0 ) CYCLE

!  find the entry in row i of this column

            DO k = listvs, listve
              i = IVAR( ISVGRP( k ) )
              IF ( i == 0 .OR. i > j ) CYCLE
              IF ( j - i > nsemib ) CYCLE
              ROW_start( i + 1 ) = ROW_start( i + 1 ) + 1

!  there is an entry in position (i,j)

            END DO
          END DO
        END DO

!  now consider the low rank first order terms for the i-th group

        DO ig = 1, ng
         IF ( skipg ) THEN
           IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
         IF ( .NOT. fixed_structure .AND. GSCALE( ig ) == 0.0_wp ) CYCLE

!  see if the group has any nonlinear elements

          DO iell = ISTADG( ig ), ISTADG( ig + 1 ) - 1
            iel = IELING( iell )
            listvs = ISTAEV( iel )
            listve = ISTAEV( iel + 1 ) - 1
            DO l = listvs, listve
              j = IVAR( IELVAR( l ) )
              IF ( j /= 0 ) THEN

!  find the entry in row i of this column

                DO k = listvs, l
                  i = IVAR( IELVAR( k ) )
                  IF ( ABS( i - j ) <= nsemib .AND. i /= 0 ) THEN

!  only the upper triangle of the matrix is stored

                    IF ( i <= j ) THEN
                      ii = i
                      jj = j
                    ELSE
                      ii = j
                      jj = i
                    END IF

!  there is an entry in position (ii,jj)

                    ROW_start( ii + 1 ) = ROW_start( ii + 1 ) + 1
                  END IF
                END DO
              END IF
            END DO
          END DO
        END DO
      
!  ROW_start(i) is changed to give the starting address for the list of 
!  column entries (with repeats) in row i (and ROW_start(n+1) points one
!  beyond the end)

!  compute starting addesses

        ROW_start( 1 ) = 1
         DO i = 2, n_free + 1
          ROW_start( i ) = ROW_start( i ) +  ROW_start( i - 1 )
        END DO

!  ===================================================
!  PASS 2: set the lists of column entries in each row
!  ===================================================

!  allocate space for column indices

        lpos = ROW_start( n_free + 1 ) - 1
        array_name = 'asmbl: POS_in_H'
        CALL SPACE_resize_array( lpos, POS_in_H, status, alloc_status,         &
                array_name = array_name, bad_alloc = bad_alloc, out = error )
        IF ( status /= GALAHAD_ok ) GO TO 980

!  consider the rank-one second order term for the i-th group

        DO ig = 1, ng
         IF ( skipg ) THEN
           IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
          IF ( GXEQX( ig ) ) CYCLE
          IF ( .NOT. fixed_structure .AND. GSCALE( ig ) == 0.0_wp ) CYCLE
          listvs = ISTAGV( ig )
          listve = ISTAGV( ig + 1 ) - 1

!  Form the j-th column of the rank-one matrix

          DO l = listvs, listve
            j = IVAR( ISVGRP( l ) )
            IF ( j == 0 ) CYCLE

!  find the entry in row i of this column

            DO k = listvs, listve
              i = IVAR( ISVGRP( k ) )
              IF ( i == 0 .OR. i > j ) CYCLE
              IF ( j - i > nsemib ) CYCLE
              POS_in_H( ROW_start( i ) ) = j
              ROW_start( i ) = ROW_start( i ) + 1

!  there is an entry in position (i,j)

            END DO
          END DO
        END DO

!  now consider the low rank first order terms for the i-th group

        DO ig = 1, ng
         IF ( skipg ) THEN
           IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
         IF ( .NOT. fixed_structure .AND. GSCALE( ig ) == 0.0_wp ) CYCLE

!  see if the group has any nonlinear elements

          DO iell = ISTADG( ig ), ISTADG( ig + 1 ) - 1
            iel = IELING( iell )
            listvs = ISTAEV( iel )
            listve = ISTAEV( iel + 1 ) - 1
            DO l = listvs, listve
              j = IVAR( IELVAR( l ) )
              IF ( j /= 0 ) THEN

!  find the entry in row i of this column

                DO k = listvs, l
                  i = IVAR( IELVAR( k ) )
                  IF ( ABS( i - j ) <= nsemib .AND. i /= 0 ) THEN

!  only the upper triangle of the matrix is stored

                    IF ( i <= j ) THEN
                      ii = i
                      jj = j
                    ELSE
                      ii = j
                      jj = i
                    END IF

!  there is an entry in position (i,j)

                    POS_in_H( ROW_start( ii ) ) = jj
                    ROW_start( ii ) = ROW_start( ii ) + 1
                  END IF
                END DO
              END IF
            END DO
          END DO
        END DO
      
!  restore the starting addresses

        DO i = n_free - 1, 1, - 1
          ROW_start( i + 1 ) = ROW_start( i )
        END DO
        ROW_start( 1 ) = 1

!  allocate workspace if required

        lused = n_free
        array_name = 'asmbl: USED'
        CALL SPACE_resize_array( lused, USED, status, alloc_status,            &
                array_name = array_name, bad_alloc = bad_alloc, out = error )
        IF ( status /= GALAHAD_ok ) GO TO 980

        lfilled = n_free
        array_name = 'asmbl: FILLED'
        CALL SPACE_resize_array( lfilled, FILLED, status, alloc_status,        &
                array_name = array_name, bad_alloc = bad_alloc, out = error )
        IF ( status /= GALAHAD_ok ) GO TO 980

!  =======================================================================
!  INTERMISSION: now pass through the nonzeros, setting up the position in 
!  the future H_row and H_col arrays of the data gathered from the groups
!  =======================================================================

        USED = 0
        k = 1
        DO i = 1, n_free
          n_filled = 0
          DO l = ROW_start( i ), ROW_start( i + 1 ) - 1
            j = POS_in_H( l )
            IF ( USED( j ) == 0 ) THEN
              n_filled = n_filled + 1
              FILLED( n_filled ) = j
              USED( j ) = k
              POS_in_H( l ) = k
              k = k + 1
            ELSE
              POS_in_H( l ) = USED( j )
            END IF
          END DO
          USED( FILLED( 1 : n_filled ) ) = 0
        END DO
        nnzh = k - 1

!  allocate space for the row and column indices and values

        lh_row = nnzh 
        array_name = 'asmbl: H_row'
        CALL SPACE_resize_array( lh_row, H_row, status, alloc_status,          &
                array_name = array_name, bad_alloc = bad_alloc, out = error )
        IF ( status /= GALAHAD_ok ) GO TO 980

        lh_col = nnzh
        array_name = 'asmbl: H_col'
        CALL SPACE_resize_array( lh_col, H_col, status, alloc_status,          &
                array_name = array_name, bad_alloc = bad_alloc, out = error )
        IF ( status /= GALAHAD_ok ) GO TO 980

        lh_val = nnzh
        array_name = 'asmbl: H_val'
        CALL SPACE_resize_array( lh_val, H_val, status, alloc_status,          &
                array_name = array_name, bad_alloc = bad_alloc, out = error )
        IF ( status /= GALAHAD_ok ) GO TO 980

        H_val( : nnzh ) = 0.0_wp
      END IF

!  ===============================================
!  PASS 3: set the row and column lists and values
!  ===============================================

!  consider the rank-one second order term for the i-th group

      DO ig = 1, ng
        IF ( skipg ) THEN
          IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
        IF ( GXEQX( ig ) ) CYCLE
        IF ( .NOT. fixed_structure .AND. GSCALE( ig ) == 0.0_wp ) CYCLE
        IF ( iprint >= 100 ) WRITE( out,                                       &
          "( ' Group ', I5, ' rank-one terms ' )" ) ig
        g2dash = GSCALE( ig ) * GVALS3( ig )
        IF ( iprint >= 100 ) WRITE( 6, * ) ' GVALS3( ig ) ', GVALS3( ig )
        ig1 = ig + 1
        listvs = ISTAGV( ig )
        listve = ISTAGV( ig1 ) - 1

!  form the gradient of the ig-th group

        GRAD_el( ISVGRP( listvs : listve ) ) = 0.0_wp

!  consider any nonlinear elements for the group

        DO iell = ISTADG( ig ), ISTADG( ig1 ) - 1
          iel = IELING( iell )
          k = INTVAR( iel )
          l = ISTAEV( iel )
          nvarel = ISTAEV( iel + 1 ) - l
          scalee = ESCALE( iell )

!  the iel-th element has an internal representation

          IF ( INTREP( iel ) ) THEN
            nin = INTVAR( iel + 1 ) - k
            CALL RANGE( iel, .TRUE., GUVALS( k : k + nin - 1 ),                &
                        H_el, nvarel, nin, ITYPEE( iel ), nin, nvarel )
            DO i = 1, nvarel
              j = IELVAR( l )
              GRAD_el( j ) = GRAD_el( j ) + scalee * H_el( i )
              l = l + 1
            END DO

!  the iel-th element has no internal representation

          ELSE
            DO i = 1, nvarel
              j = IELVAR( l )
              GRAD_el( j ) = GRAD_el( j ) + scalee * GUVALS( k )
              k = k + 1 ; l = l + 1
            END DO
          END IF
        END DO

!  include the contribution from the linear element

        DO k = ISTADA( ig ), ISTADA( ig1 ) - 1
          j = ICNA( k )
          GRAD_el( j ) = GRAD_el( j ) + A( k )
        END DO

!  the gradient is complete. Form the j-th column of the rank-one matrix

        DO l = listvs, listve
          jj = ISVGRP( l )
          j = IVAR( jj )
          IF ( j == 0 ) CYCLE

!  find the entry in row i of this column

          DO k = listvs, listve
            ii = ISVGRP( k )
            i = IVAR( ii )
            IF ( i == 0 .OR. i > j ) CYCLE

!  Skip all elements which lie outside a band of width nsemib

            IF ( use_band ) maxsbw = MAX( maxsbw, j - i )
            IF ( j - i > nsemib ) CYCLE
            hesnew = GRAD_el( ii ) * GRAD_el( jj ) * g2dash
            IF ( iprint >= 100 ) WRITE( out,                                   &
              "( ' Row ', I6, ' column ', I6, ' used. Value = ', ES24.16 )" )  &
                i, j, hesnew

!  obtain the appropriate storage location in H for the new entry

!  Case 1: band matrix storage scheme

            IF ( use_band ) THEN

!  the entry belongs on the diagonal

              IF ( i == j ) THEN
                DIAG( i ) = DIAG( i ) + hesnew

!  the entry belongs off the diagonal

              ELSE
                OFFDIA( j - i, i ) = OFFDIA( j - i, i ) + hesnew
              END IF

!  Case 2: co-ordinate storage scheme

            ELSE

!  there is an entry in position (i,j) to be stored in 
!  H_row/col(COL(ROW_start(i)))

              kk = POS_in_H( ROW_start( i ) )
              H_row( kk ) = i
              H_col( kk ) = j
              H_val( kk ) = H_val( kk ) + hesnew
              ROW_start( i ) = ROW_start( i ) + 1
            END IF
          END DO
        END DO
      END DO

!  reset the workspace array to zero

      W_el( : maxsel ) = 0.0_wp

!  now consider the low rank first order terms for the i-th group

      DO ig = 1, ng
        IF ( skipg ) THEN
          IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
        IF ( .NOT. fixed_structure .AND. GSCALE( ig ) == 0.0_wp ) CYCLE
        IF ( iprint >= 100 ) WRITE( out,                                       &
          "( ' Group ', I5, ' second-order terms ' )" )  ig
        IF ( GXEQX( ig ) ) THEN
          gdash = GSCALE( ig )
        ELSE
          gdash = GSCALE( ig ) * GVALS2( ig )
          IF ( iprint >= 100 ) WRITE( 6, * ) ' GVALS2( ig )', GVALS2( ig )
        END IF
        ig1 = ig + 1

!  see if the group has any nonlinear elements

        DO iell = ISTADG( ig ), ISTADG( ig + 1 ) - 1
          iel = IELING( iell )
          listvs = ISTAEV( iel )
          listve = ISTAEV( iel + 1 ) - 1
          nvarel = listve - listvs + 1
          ielh = ISTADH( iel )
          ihnext = ielh
          scalee = ESCALE( iell )
          DO l = listvs, listve
            j = IVAR( IELVAR( l ) )
            IF ( j /= 0 ) THEN

!  the iel-th element has an internal representation. Compute the j-th column
!  of the element Hessian matrix

              IF ( INTREP( iel ) ) THEN

!  compute the j-th column of the Hessian

                W_el( l - listvs + 1 ) = 1.0_wp

!  find the internal variables

                nin = INTVAR( iel + 1 ) - INTVAR( iel )
                CALL RANGE( iel, .FALSE., W_el, W_in, nvarel, nin,             &
                            ITYPEE( iel ), nvarel, nin )

!  multiply the internal variables by the element Hessian

                H_in( : nin ) = 0.0_wp

!  only the upper triangle of the element Hessian is stored

                jcolst = ielh - 1
                DO jcol = 1, nin
                  ijhess = jcolst
                  jcolst = jcolst + jcol
                  wki = W_in( jcol ) * gdash
                  DO irow = 1, nin
                    IF ( irow <= jcol ) THEN
                      ijhess = ijhess + 1
                    ELSE
                      ijhess = ijhess + irow - 1
                    END IF
                    H_in( irow ) = H_in( irow ) + wki * HUVALS( ijhess )
                  END DO
                END DO

!  scatter the product back onto the elemental variables

                CALL RANGE( iel, .TRUE., H_in, H_el, nvarel, nin,              &
                            ITYPEE( iel ), nin, nvarel )
                W_el( l - listvs + 1 ) = 0.0_wp
              END IF

!  find the entry in row i of this column

              DO k = listvs, l
                i = IVAR( IELVAR( k ) )

!  skip all elements which lie outside a band of width nsemib; only the upper 
!  triangle of the matrix is stored

                IF ( use_band .AND. i /= 0 ) maxsbw = MAX( maxsbw, ABS( j - i ))
                IF ( ABS( i - j ) <= nsemib .AND. i /= 0 ) THEN
                  IF ( i <= j ) THEN
                    ii = i
                    jj = j
                  ELSE
                    ii = j
                    jj = i
                  END IF

!  obtain the appropriate storage location in H for the new entry

                  IF ( INTREP( iel ) ) THEN
                    hesnew = scalee * H_el( k - listvs + 1 )
                  ELSE
                    hesnew = scalee * HUVALS( ihnext ) * gdash
                  END IF
                  IF ( iprint >= 100 ) WRITE( 6, "( ' Row ', I6, ' Column ',   &
                 &   I6, ' used from element ', I6, ' value = ', ES24.16 )" )  &
                    ii, jj, iel, hesnew

!  Case 1: band matrix storage scheme

                  IF ( use_band ) THEN

!  The entry belongs on the diagonal

                    IF ( ii == jj ) THEN
                      DIAG( ii ) = DIAG( ii ) + hesnew
                      IF ( k /= l ) DIAG( ii ) = DIAG( ii ) + hesnew

!  the entry belongs off the diagonal

                    ELSE
                      OFFDIA( jj - ii, ii ) = OFFDIA( jj - ii, ii ) + hesnew
                    END IF

!  Case 2: co-ordinate storage scheme

                  ELSE

!  there is an entry in position (i,j) to be stored in 
!  H_row/col(COL(ROW_start(i)))

                    kk = POS_in_H( ROW_start( ii ) )
                    H_row( kk ) = ii
                    H_col( kk ) = jj
                    H_val( kk ) = H_val( kk ) + hesnew
                    IF ( k /= l .AND. ii == jj )                               &
                      H_val( kk ) = H_val( kk ) + hesnew
                    ROW_start( ii ) = ROW_start( ii ) + 1
                  END IF
                END IF
                ihnext = ihnext + 1
              END DO
            END IF
          END DO
        END DO
      END DO
      
!  if required, remove any zero entries

      IF ( .NOT. use_band .AND. no_zeros ) THEN
        k = nnzh
        nnzh = 0
        DO l = 1, k
          IF ( H_val( l ) /= 0.0_wp ) THEN
            nnzh = nnzh + 1
            H_row( nnzh ) = H_row( l ) ; H_col( nnzh ) = H_col( l )
            H_val( nnzh ) = H_val( l )
          END IF
        END DO
      END IF

!  restore the starting addresses

!     DO i = n_free - 1, 1, - 1
!       ROW_start( i + 1 ) = ROW_start( i )
!     END DO
!     ROW_start( 1 ) = 1

!  ---------------------------------------
!  For debugging, print the nonzero values
!  ---------------------------------------

      IF ( iprint >= 10 ) THEN
        IF ( .NOT. use_band )                                                  &
          WRITE( out,                                                          &
           "( '    Row  Column    Value        Row  Column    Value ', /       &
         &    '    ---  ------    -----        ---  ------    ----- ', /       &
         &    ( 2I6, ES24.16, 2I6, ES24.16 ) )" )                              &
            ( H_row( i ), H_col( i ), H_val( i ), i = 1, nnzh )

!  for debugging, print the nonzero pattern of the matrix

        IF ( n <= 36 ) THEN
          MATRIX( : n, : n ) = '  '
          IF ( use_band ) THEN
            DO i = 1, n
              IF ( DIAG( i ) /= 0.0_wp ) MATRIX( i, i ) = ' *'
              DO j = 1, MIN( nsemib, n - i )
                IF ( OFFDIA( j, i ) /= 0.0_wp ) THEN
                   MATRIX( i + j, i ) = ' *'
                   MATRIX( i, i + j ) = ' *'
                END IF
              END DO
            END DO
          ELSE
            DO i = 1, nnzh
              IF ( H_row( i ) > n ) THEN
                WRITE( out,                                                    &
                  "( ' Entry out of bounds in CUTEST_assemble_hessian',        &
                 &   ' row number = ', I0 )" ) H_row( i )
!               STOP
              END IF
              IF ( H_col( i ) > n ) THEN
                WRITE( out,                                                    &
                  "( ' Entry out of bounds in CUTEST_assemble_hessian',        &
                 &   ' col number = ', I0 )" ) H_col( i )
!               STOP
              END IF
              MATRIX( H_row( i ), H_col( i ) ) = ' *'
              MATRIX( H_col( i ), H_row( i ) ) = ' *'
            END DO
          END IF
          WRITE( out, "( /, 5X, 36I2 )" ) ( i, i = 1, n )
          DO i = 1, n
            WRITE( out, "( I3, 2X, 36A2 )" ) i, ( MATRIX( i, j ), j = 1, n )
          END DO
        END IF
      END IF

!  successful return

      status = 0
      RETURN

!  unsuccessful returns

  980 CONTINUE
      WRITE( error, "( ' ** Message from -ASMBL_assemble_hessian-',            &
     &    /, ' Allocation error (status = ', I0, ') for ', A )" )              &
        alloc_status, bad_alloc
      RETURN

!  End of subroutine ASMBL_assemble_hessian

     END SUBROUTINE ASMBL_assemble_hessian

!  End of module LANCELOT_ASMBL

   END MODULE LANCELOT_ASMBL_double

