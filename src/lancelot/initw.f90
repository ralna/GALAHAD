! THIS VERSION: GALAHAD 3.3 - 20/05/2021 AT 10:30 GMT.

!-*-*-*-*-*-  L A N C E L O T  -B-  INITW  M O D U L E  *-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 1st 1995

     MODULE LANCELOT_INITW_double
  
       USE GALAHAD_EXTEND_double, ONLY: EXTEND_arrays
       USE LANCELOT_OTHERS_double
       IMPLICIT NONE
       
       PRIVATE
       PUBLIC :: INITW_initialize_workspace
       
!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

       REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
       REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
  
     CONTAINS

!-*-  L A N C E L O T -B- INITW_initialize_workspace  S U B R O U T I N E  -*

       SUBROUTINE INITW_initialize_workspace(                                  &
                        n , ng, nel   , ntotel, nvrels, nnza  , numvar,        &
                        nvargp, IELING, ISTADG, IELVAR, ISTAEV, INTVAR,        &
                        ISTADH, ICNA  , ISTADA, ITYPEE, GXEQX , INTREP,        &
                        altriv, direct, fdgrad, lfxi  , lgxi  , lhxi  ,        &
                        lggfx , ldx   , lnguvl, lnhuvl, ntotin, ntype ,        &
                        nsets , maxsel, RANGE , iprint, iout  , buffer,        &
! workspace
                        lwtran, litran, lwtran_min, litran_min,                &
                        l_link_e_u_v, llink_min,                               &
                        ITRANS, LINK_elem_uses_var, WTRANS,                    &
                        ISYMMD, ISWKSP, ISTAJC, ISTAGV,                        &
                        ISVGRP, ISLGRP, IGCOLJ, IVALJR,                        &
                        IUSED , ITYPER, ISSWTR, ISSITR,                        &
                        ISET  , ISVSET, INVSET, LIST_elements,                 &
                        ISYMMH,                                                &
                        IW_asmbl, NZ_comp_w, W_ws,                             &
                        W_el, W_in, H_el, H_in,                                &
                        status, alloc_status, bad_alloc,                       &
                        skipg, KNDOFG )

!  Compute the starting addresses for the partitions of the workspace
!  array FUVALS. Also fill relevant portions of the workspace arrays WTRANS and
!  ITRANS.

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

       INTEGER, INTENT( IN ) :: n, ng, nel, ntotel, nvrels, nnza, numvar
       INTEGER, INTENT( IN ) :: iprint, iout, buffer
       INTEGER, INTENT( OUT ) :: lfxi  , lgxi  , lhxi  , lggfx , ldx  
       INTEGER, INTENT( OUT ) :: lnguvl, lnhuvl, nvargp, status, alloc_status
       INTEGER, INTENT( OUT ) :: ntotin, ntype , nsets , maxsel
       LOGICAL, INTENT( IN ) :: direct, fdgrad, skipg
       LOGICAL, INTENT( OUT ) :: altriv
       CHARACTER ( LEN = 80 ), INTENT( OUT ) :: bad_alloc
       INTEGER, INTENT( IN ), DIMENSION( ntotel  ) :: IELING
       INTEGER, INTENT( IN ), DIMENSION( ng  + 1 ) :: ISTADA, ISTADG
       INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTAEV
       INTEGER, INTENT( IN ), DIMENSION( nvrels  ) :: IELVAR
       INTEGER, INTENT( IN ), DIMENSION( nnza    ) :: ICNA
       INTEGER, INTENT( OUT ), DIMENSION( nel + 1 ) :: ISTADH
       INTEGER, INTENT( INOUT ), DIMENSION( nel + 1 ) :: INTVAR
       INTEGER, INTENT( IN ), DIMENSION ( : ) :: ITYPEE
       LOGICAL, INTENT( IN ), DIMENSION( ng  ) :: GXEQX
       LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP

       INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG

!-------------------------------------------------------------
!   D u m m y   A r g u m e n t s  f o r   w o r k s p a c e 
!-------------------------------------------------------------

       INTEGER, INTENT( INOUT ) :: lwtran, litran, lwtran_min, litran_min
       INTEGER, INTENT( INOUT ) :: l_link_e_u_v, llink_min

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ITRANS
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: LINK_elem_uses_var
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WTRANS
  
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISYMMD
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISWKSP
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTAJC
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTAGV
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISVGRP
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISLGRP
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IGCOLJ
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IVALJR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IUSED 
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ITYPER
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISSWTR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISSITR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISET
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISVSET
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INVSET
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: LIST_elements
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: ISYMMH
  
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW_asmbl
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_comp_w
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_ws
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_el
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_in
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_el
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_in

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

       INTERFACE
         SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,    &
                            lw1, lw2 )
         INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
         LOGICAL, INTENT( IN ) :: transp
         REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
!        REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
         REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
         END SUBROUTINE RANGE
       END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

       INTEGER :: i , j , k , l , iielts, ientry, ig, is
       INTEGER :: nsizeh, nel1  , iel   , lwfree, lifree, lnwksp
       INTEGER :: nelvr , liwfro, lwfreo, iell  , itype , isofar
       INTEGER :: istarj, ivarp1, ivar  , jset  , inext , newvar, newset
       INTEGER :: ipt   , istrt , ninvr , ii, kk, ll
       INTEGER :: nwtran, mwtran, uwtran, llink , mlink , nlink, ulink
       INTEGER :: nitran, mitran, uitran, maxsin
       LOGICAL :: alllin, vrused, reallocate
!      CHARACTER ( LEN = 80 ) :: array

!  Set constants

       nel1 = nel + 1
       alllin = nel == 0

!  Set up INTVAR, the starting addresses for the element gradients with
!  respect to their internal variables. Also compute maxsin, the maximum
!  number of internal variables in an element

       IF ( .NOT. alllin ) THEN
         k = INTVAR( 1 )
         maxsin = k
         INTVAR( 1 ) = nel1
         DO iel = 2, nel
           l = INTVAR( iel )
           INTVAR( iel ) = INTVAR( iel - 1 ) + k
           k = l
           maxsin = MAX( maxsin, k )
         END DO
         INTVAR( nel1 ) = INTVAR( nel ) + k
       ELSE
         INTVAR( 1 ) = 1
         maxsin = 0
       END IF

!  Compute the total number of internal variables

       ntotin = INTVAR( nel1 ) - INTVAR( 1 )

!  Calculate the length, iielts, of workspace required to determine which
!  elements use each of the variables. Also find the maximum number of
!  variables in an element, maxsel. This is a dummy run merely to calculate
!  the space required

       llink = n + llink_min
       reallocate = .TRUE.
       IF ( ALLOCATED( LINK_elem_uses_var ) ) THEN
         IF ( SIZE( LINK_elem_uses_var ) < llink ) THEN
           DEALLOCATE( LINK_elem_uses_var )
         ELSE ; llink = SIZE( LINK_elem_uses_var ) ; reallocate = .FALSE.
         END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( LINK_elem_uses_var( llink ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'LINK_e' ; GO TO 600 ; END IF
       END IF

!  LINK_elem_uses_var( i ) will be used as a list of links chaining the 
!  elements using variable i. If LINK_elem_uses_var( i ) is negative, the 
!  list is empty

       LINK_elem_uses_var( : n ) = - 1
       iielts = n ; maxsel = 0
       IF ( .NOT. alllin ) THEN

!  Loop over the groups, considering each nonlinear element in turn

         DO iel = 1, nel
           maxsel = MAX( maxsel, ISTAEV( iel + 1 ) - ISTAEV( iel ) )
         END DO
         DO i = 1, ntotel
           iel = IELING( i )

!  Loop on the variables from the I-th element

           DO k = ISTAEV( iel ), ISTAEV( iel + 1 ) - 1
             ientry = IELVAR( k )
             IF ( LINK_elem_uses_var( ientry ) >= 0 ) THEN

!  If we have reached the end of the list of the elements using the variable
!  IELVAR( K ), add the IEL-th element to it. Otherwise, find the next entry
!  in the list

  30           CONTINUE
               IF ( LINK_elem_uses_var( ientry ) > 0 ) THEN
                 ientry = LINK_elem_uses_var( ientry )
                 GO TO 30
               ELSE
                IF ( iielts == llink ) THEN
                   nlink = llink
                   ulink = iielts; mlink = iielts + 1
!                  array = 'LINK_elem_uses_var'
!                  CALL EXTEND_array( array, llink, ulink, nlink,              &
!                                     mlink, status, iprint, iout )
                   CALL EXTEND_arrays( LINK_elem_uses_var, llink, ulink,       &
                                       nlink, mlink, buffer, status,           &
                                       alloc_status)
                   IF ( status /= 0 ) THEN
                     bad_alloc = 'LINK_elem_uses_var' ; GO TO 610 ; END IF
                   llink = nlink
                 END IF
                 iielts = iielts + 1
                 LINK_elem_uses_var( ientry ) = iielts
                 LINK_elem_uses_var( iielts ) = 0
               END IF
             ELSE

!  The list of elements involving the variable IELVAR( K ) was
!  previously empty. Indicate that the list has now been started and
!  that its end has been reached

               LINK_elem_uses_var( ientry ) = 0
             END IF
           END DO
         END DO
       END IF
       
       l_link_e_u_v = iielts

!  -- Calculate the starting addresses for the integer workspace --

!  ISWKSP( j ), j = 1, ..., MAX( ntotel, nel, n + n ), is used for
!  workspace by the matrix-vector product subroutine HSPRD

       IF ( direct ) THEN
         lnwksp = MAX( MAX( ntotel, nel ), n + n )
       ELSE
         lnwksp = MAX( MAX( ntotel, nel ), n )
       END IF
       
       reallocate = .TRUE.
       IF ( ALLOCATED( ISWKSP ) ) THEN
         IF ( SIZE( ISWKSP ) < lnwksp ) THEN ; DEALLOCATE( ISWKSP )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( ISWKSP( lnwksp ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'ISWKSP' ; GO TO 600 ; END IF
       END IF

!  IUSED( j ), j = 1, ..., MAX( n, ng ) Will be used as workspace by
!  the matrix-vector product subroutine HSPRD

       reallocate = .TRUE.
       IF ( ALLOCATED( IUSED ) ) THEN
         IF ( SIZE( IUSED ) < MAX( n, ng ) ) THEN ; DEALLOCATE( IUSED )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( IUSED( MAX( n, ng ) ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'IUSED' ; GO TO 600 ; END IF
       END IF

!  ISLGRP( j ), j = 1, ..., ntotel, will contain the number of the group
!  which uses nonlinear element j

       reallocate = .TRUE.
       IF ( ALLOCATED( ISLGRP ) ) THEN
         IF ( SIZE( ISLGRP ) < ntotel ) THEN ; DEALLOCATE( ISLGRP )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( ISLGRP( ntotel ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'ISLGRP' ; GO TO 600 ; END IF
       END IF

!  ISTAJC( j ), j = 1, ..., n, will contain the starting addresses for
!  the list of nontrivial groups which use the j-th variable.
!  ISTAJC( n + 1 ) will point to the first free location in IGCOLJ after
!  the list of nontrivial groups for the n-th variable

       reallocate = .TRUE.
       IF ( ALLOCATED( ISTAJC ) ) THEN
         IF ( SIZE( ISTAJC ) < n + 1 ) THEN ; DEALLOCATE( ISTAJC )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( ISTAJC( n + 1 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'ISTAJC' ; GO TO 600 ; END IF
       END IF

!  ISTAGV( j ), j = 1, ..., ng, will contain the starting addresses for
!  the list of variables which occur in the J-th group. ISTAGV( ng + 1 )
!  will point to the first free location in ISVGRP after the list of variables
!  for the NG-th group

       reallocate = .TRUE.
       IF ( ALLOCATED( ISTAGV ) ) THEN
         IF ( SIZE( ISTAGV ) < ng + 1 ) THEN ; DEALLOCATE( ISTAGV )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( ISTAGV( ng + 1 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'ISTAGV' ; GO TO 600 ; END IF
       END IF

!  Allocate LIST_elements

       reallocate = .TRUE.
       IF ( ALLOCATED( LIST_elements ) ) THEN
         IF ( SIZE( LIST_elements ) < l_link_e_u_v ) THEN 
           DEALLOCATE( LIST_elements )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( LIST_elements( l_link_e_u_v ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'LIST_e' ; GO TO 600 ; END IF
       END IF

!  Determine which elements use each variable. Initialization

       IF ( .NOT. alllin ) THEN

!  LINK_elem_uses_var( i ) will be used as a list of links chaining the 
!  elements using variable i. If LINK_elem_uses_var( i ) is negative, the 
!  list is empty

         LINK_elem_uses_var( : n ) = - 1
         LIST_elements( : n ) = - 1   ! needed for epcf90 debugging compiler
         iielts = n

!  Loop over the groups, considering each nonlinear element in turn

         DO i = 1, ntotel
           iel = IELING( i )

!  Loop on the variables of the I-th element

           DO k = ISTAEV( iel ), ISTAEV( iel + 1 ) - 1
             ientry = IELVAR( k )
             IF ( LINK_elem_uses_var( ientry ) >= 0 ) THEN

!  If we have reached the end of the list of the elements using the variable
!  IELVAR( K ), add the I-th element to it and record that the end of the list
!  has occured. Otherwise, find the next entry in the list

  110          CONTINUE
               IF ( LINK_elem_uses_var( ientry ) > 0 ) THEN
                 ientry = LINK_elem_uses_var( ientry )
                 GO TO 110
               ELSE
                 iielts = iielts + 1
                 LINK_elem_uses_var( ientry ) = iielts
                 LINK_elem_uses_var( iielts ) = 0
                 LIST_elements( iielts ) = i
               END IF
             ELSE

!  The list of elements involving the variable IELVAR( K ) was previously
!  empty. Indicate that the list has now been started, record the element
!  which contains the variable and indicate that the end of the list has
!  been reached

               LINK_elem_uses_var( ientry ) = 0
               LIST_elements( ientry ) = i
             END IF
           END DO
         END DO
       END IF

!  Set up symmetric addresses for the upper triangular storage
!  schemes for the element hessians

       IF ( maxsin > 0 ) THEN
         reallocate = .TRUE.
         IF ( ALLOCATED( ISYMMD ) ) THEN
            IF ( SIZE( ISYMMD ) < maxsin ) THEN ; DEALLOCATE( ISYMMD )
            ELSE ; reallocate = .FALSE.
            END IF
         END IF
         IF ( reallocate ) THEN 
            ALLOCATE( ISYMMD( maxsin ), STAT = alloc_status )
            IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ISYMMD' ; GO TO 600
            END IF
         END IF
         
         reallocate = .TRUE.
         IF ( ALLOCATED( ISYMMH ) ) THEN
           IF ( SIZE( ISYMMH, 1 ) /= maxsin .OR. SIZE( ISYMMH, 2 ) /= maxsin ) &
             THEN  ; DEALLOCATE( ISYMMH ) ; ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ISYMMH( maxsin, maxsin ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ISYMMH' ; GO TO 600
           END IF
         END IF
         
         CALL OTHERS_symmh( maxsin, ISYMMH, ISYMMD )
       ELSE
         ALLOCATE( ISYMMD( 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ISYMMD' ; GO TO 600
         END IF
         ALLOCATE( ISYMMH( 0, 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ISYMMH' ; GO TO 600
         END IF
       END IF

!  Set up the starting addresses for the element Hessians
!  with respect to their internal variables and a pointer beyond
!  the end of the space required for the Hessians

       lggfx = INTVAR( nel1 )
       IF ( .NOT. alllin ) THEN
         DO i = 1, nel
           ISTADH( i ) = lggfx
           nsizeh = INTVAR( i + 1 ) - INTVAR( i )
           lggfx = lggfx + nsizeh * ( nsizeh + 1 ) / 2
         END DO
       END IF
       ISTADH( nel1 ) = lggfx

!  ALTRIV specifies whether all the groups are trivial

       altriv = .TRUE.

!  Pass 1: Count the total number of variables in all the groups, nvargp

       nvargp = 0

!  Start by initializing the counting array to zero

       ISWKSP( : numvar ) = 0

!  Loop over the groups. See if the IG-th group is trivial

       DO ig = 1, ng

!  Check to see if all of the groups are trivial

         IF ( skipg ) THEN ; IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
         IF ( .NOT. GXEQX( ig ) ) altriv = .FALSE.

!  Loop over the nonlinear elements from the IG-th group

         DO k = ISTADG( ig ), ISTADG( ig + 1 ) - 1
           iel = IELING( k )

!  Run through all the elemental variables changing the I-th entry of
!  ISWKSP from zero to one if variable I appears in an element

           DO j = ISTAEV( iel ), ISTAEV( iel + 1 ) - 1
             i = IELVAR( j )
             IF ( ISWKSP( i ) < ig ) THEN
               ISWKSP( i ) = ig
               nvargp = nvargp + 1
             END IF
           END DO
         END DO

!  Consider variables which arise from the linear element

         DO j = ISTADA( ig ), ISTADA( ig + 1 ) - 1
           i = ICNA( j )
           IF ( i <= numvar ) THEN
             IF ( ISWKSP( i ) < ig ) THEN
                ISWKSP( i ) = ig
                nvargp = nvargp + 1
             END IF
           END IF
         END DO
       END DO

!  ISVGRP( j ), j = 1, ..., nvargp, will contain the indices of the
!  variables which are used by each group in turn. Those for group i occur
!  in locations ISTAGV( i ) to ISTAGV( i + 1 ) - 1

!  Allocate the array ISVGRP

       reallocate = .TRUE.
       IF ( ALLOCATED( ISVGRP ) ) THEN
         IF ( SIZE( ISVGRP ) < nvargp ) THEN ; DEALLOCATE( ISVGRP )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( ISVGRP( nvargp ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'ISVGRP' ; GO TO 600 ; END IF
       END IF

!  Store the indices of variables which appears in each group and how many
!  groups use each variable. Reinitialize counting arrays to zero

       ISTAJC( 2 : n + 1 ) = 0
       ISWKSP( : numvar ) = 0

!  Pass 2: store the list of variables

       nvargp = 0
       ISTAGV( 1 ) = 1

!  Loop over the groups. See if the IG-th group is trivial

       DO ig = 1, ng

         IF ( skipg ) THEN 
           IF ( KNDOFG( ig ) == 0 ) THEN
             ISLGRP( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) = ig
             ISTAGV( ig + 1 ) = nvargp + 1
             CYCLE
           END IF
         END IF

!  Again, loop over the nonlinear elements from the IG-th group

         DO k = ISTADG( ig ), ISTADG( ig + 1 ) - 1
           iel = IELING( k )

!  Run through all the elemental variables changing the I-th entry of
!  ISWKSP from zero to one if variable I appears in an element

           DO j = ISTAEV( iel ), ISTAEV( iel + 1 ) - 1
             i = IELVAR( j )
             IF ( ISWKSP( i ) < ig ) THEN
               ISWKSP( i ) = ig

!  Record the nonlinear variables from the ig-th group

               nvargp = nvargp + 1
               ISVGRP( nvargp ) = i
             END IF
           END DO

!  Record that nonlinear element K occurs in group IELGRP( IEL )

           ISLGRP( k ) = ig
         END DO

!  Consider variables which arise from the linear element

         DO j = ISTADA( ig ), ISTADA( ig + 1 ) - 1
           i = ICNA( j )
           IF ( i <= numvar ) THEN
             IF ( ISWKSP( i ) < ig ) THEN
               ISWKSP( i ) = ig

!  Record the linear variables from the ig-th group

               nvargp = nvargp + 1
               ISVGRP( nvargp ) = i
             END IF
           END IF
         END DO

!  Record that one further nontrivial group uses variable l - 1

         IF ( .NOT. GXEQX( ig ) ) THEN
           DO j = ISTAGV( ig ), nvargp
             l = ISVGRP( j ) + 1
             ISTAJC( l ) = ISTAJC( l ) + 1
           END DO
         END IF

!  Record the starting address of the variables in the next group

         ISTAGV( ig + 1 ) = nvargp + 1
       END DO
       ISWKSP( : n ) = 0

!  IGCOLJ( j ), j = 1, ..., nvargp, will contain the indices of the
!  nontrivial groups which use each variable in turn. Those for variable i
!  occur in locations ISTAJC( i ) to ISTAJC( i + 1 ) - 1

       reallocate = .TRUE.
       IF ( ALLOCATED( IGCOLJ ) ) THEN
         IF ( SIZE( IGCOLJ ) < nvargp ) THEN ; DEALLOCATE( IGCOLJ )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( IGCOLJ( nvargp ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'IGCOLJ' ; GO TO 600 ; END IF
       END IF

!  IVALJR( j ), j = 1, ..., nvargp, will contain the positions in GRJAC
!  of the nonzeros of the Jacobian
!  of the groups corresponding to the variables as ordered in ISVGRP( j )

       reallocate = .TRUE.
       IF ( ALLOCATED( IVALJR ) ) THEN
         IF ( SIZE( IVALJR ) < nvargp ) THEN ; DEALLOCATE( IVALJR )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( IVALJR( nvargp ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'IVALJR' ; GO TO 600
         END IF
       END IF

!  Set the starting addresses for the lists of nontrivial groups which use
!  each variable in turn

       k = 1
       ISTAJC( k ) = 1
       DO i = 2, n + 1
         k = k + 1
         ISTAJC( k ) = ISTAJC( k ) + ISTAJC( k - 1 )
       END DO

!  Consider the IG-th group in order to associate variables with groups

       DO ig = 1, ng
         IF ( skipg ) THEN ; IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
         IF ( .NOT. GXEQX( ig ) ) THEN
           DO i = ISTAGV( ig ), ISTAGV( ig + 1 ) - 1
             l = ISVGRP( i )

!  Record that group IG uses variable ISVGRP( i )

             j = ISTAJC( l )
             IGCOLJ( j ) = ig

!  Store the locations in the Jacobian of the groups of the nonzeros
!  corresponding to each variable in the IG-TH group. Increment the starting
!  address for the pointer to the next group using variable ISVGRP( i )

             IVALJR( i ) = j
             ISTAJC( l ) = j + 1
           END DO
         END IF
       END DO

!  Reset the starting addresses for the lists of groups using each variable

       DO i = n, 2, - 1
         ISTAJC( i ) = ISTAJC( i - 1 )
       END DO
       ISTAJC( 1 ) = 1

!  Initialize workspace values for subroutine HSPRD

       IUSED( : MAX( n, ng ) ) = 0

!  Initialize general workspace arrays

       maxsin = MAX( 1, maxsin )
       maxsel = MAX( 1, maxsel )

       IF ( ALLOCATED( IW_asmbl ) ) DEALLOCATE( IW_asmbl )
       ALLOCATE( IW_asmbl( n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'IW_asmb' ; GO TO 600
       END IF
       
       IF ( ALLOCATED( NZ_comp_w ) ) DEALLOCATE( NZ_comp_w )
       ALLOCATE( NZ_comp_w( ng ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'NZ_com' ; GO TO 600
       END IF
       
       IF ( ALLOCATED( W_ws ) ) DEALLOCATE( W_ws )
       ALLOCATE( W_ws( MAX( n, ng ) ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'W_ws' ; GO TO 600
       END IF
       
       IF ( ALLOCATED( W_el ) ) DEALLOCATE( W_el )
       ALLOCATE( W_el( maxsel ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'W_el' ; GO TO 600
       END IF
       
       IF ( ALLOCATED( W_in ) ) DEALLOCATE( W_in )
       ALLOCATE( W_in( maxsin ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'W_in' ; GO TO 600
       END IF
       
       IF ( ALLOCATED( H_el ) ) DEALLOCATE( H_el )
       ALLOCATE( H_el( maxsel ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'H_el' ; GO TO 600
       END IF
       
       IF ( ALLOCATED( H_in ) ) DEALLOCATE( H_in )
       ALLOCATE( H_in( maxsin ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'H_in' ; GO TO 600
       END IF

!  Define further partitions of the workspace whenever finite-difference
!  gradients are used

       IF ( fdgrad ) THEN

!  The range transformation for each nonlinear element is of a given type.
!  Suppose there are NTYPE non-trivial types. ITYPER( i ) gives the type
!  of nonlinear element i for i = 1, ...., nel

         reallocate = .TRUE.
         IF ( ALLOCATED( ITYPER ) ) THEN
           IF ( SIZE( ITYPER ) < nel ) THEN ; DEALLOCATE( ITYPER )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ITYPER( nel ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'ITYPER' ; GO TO 600 ; END IF
         END IF

!  The range transformation from elemental to internal variables is defined by
!  a matrix W. For each non-trivial transformation, the matrix W is recorded.
!  The information for the I-th type starts in location 
!  ISSWTR( i ), i = 1, ...., ntype of WTRANS

         reallocate = .TRUE.
         IF ( ALLOCATED( ISSWTR ) ) THEN
           IF ( SIZE( ISSWTR ) < nel ) THEN ; DEALLOCATE( ISSWTR )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ISSWTR( nel ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'ISSWTR' ; GO TO 600 ; END IF
         END IF

!  For each type of nonlinear element using a nontrivial range transformation,
!  integer information is also recorded. The information for the i-th type
!  starts in location ISSITR( i ), i = 1, ...., ntype of ITRANS

         reallocate = .TRUE.
         IF ( ALLOCATED( ISSITR ) ) THEN
           IF ( SIZE( ISSITR ) < nel ) THEN ; DEALLOCATE( ISSITR )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ISSITR( nel ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
             bad_alloc = 'ISSITR' ; GO TO 600 ; END IF
         END IF

!  The following pieces of integer information are recorded for the I-th type
!  of nonlinear element:

!    ITRANS( ISSITR( i ) + 1 ):
!            the number of internal variables, ninvr
!    ITRANS( ISSITR( i ) + 2 ):
!            the number of elemental variables, nelvr
!    ITRANS( ISSITR( i ) + 2 + j ), j = 1, ..., ninvr + nelvr:
!            pivot sequences for the LU factors of W.

!  After the factorization and compression, only ninvr linearly independent
!  columns of W are stored

!  -- Make an initial allocation of WTRANS and ITRANS

         reallocate = .TRUE.
         IF ( ALLOCATED( WTRANS ) ) THEN
           IF ( SIZE( WTRANS ) < lwtran_min ) THEN
              DEALLOCATE( WTRANS ) ; lwtran = lwtran_min
           ELSE ; reallocate = .FALSE. ; END IF
         ELSE ; lwtran = lwtran_min ; END IF
         IF ( reallocate ) THEN 
           ALLOCATE( WTRANS( lwtran ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
              bad_alloc = 'WTRANS' ; GO TO 600 ; END IF
         END IF
         
         reallocate = .TRUE.
         IF ( ALLOCATED( ITRANS ) ) THEN
           IF ( SIZE( ITRANS ) < litran_min ) THEN
             DEALLOCATE( ITRANS ) ; litran = litran_min
           ELSE ; reallocate = .FALSE. ; END IF
         ELSE ; litran = litran_min ; END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ITRANS( litran ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
             bad_alloc = 'ITRANS' ; GO TO 600 ; END IF
         END IF

!  ---------------------------------------------------
!  Consider only elements which use internal variables
!  ---------------------------------------------------

         ntype = 0 ; lwfree = 1 ; lifree = 1 
         W_el( : maxsel ) = zero

!  Loop over all nonlinear elements

   LIEL: DO iel = 1, nel
           IF ( INTREP( iel ) ) THEN

!  Calculate the range transformation matrix WTRANS

             is = ISTAEV( iel )
             ninvr = INTVAR( iel + 1 ) - INTVAR( iel )
             nelvr = ISTAEV( iel + 1 ) - is
             mwtran = lwfree + ninvr * nelvr - 1
                  
!  Ensure that there is enough space

             IF ( mwtran > lwtran ) THEN
               nwtran = 2 * ( lwfree + ninvr * nelvr - 1 )
               uwtran = lwfree - 1
!              array = 'WTRANS'
!              CALL EXTEND_array( array, lwtran, uwtran, nwtran,               &
!                                 mwtran, status, iprint, iout )
               CALL EXTEND_arrays( WTRANS, lwtran, uwtran, nwtran, mwtran,     &
                                   buffer, status, alloc_status )
               IF ( status /= 0 ) THEN
                 bad_alloc = 'WTRANS' ; GO TO 610 ; END IF
               lwtran = nwtran
             END IF
   
             k = lwfree - 1
             is = is - 1
   
             DO i = 1, nelvr
               W_el( i ) = one
               CALL RANGE ( iel, .FALSE., W_el( : nelvr ),                     &
                            WTRANS( k + 1 : k + ninvr ), nelvr, ninvr,         &
                            ITYPEE( iel ), nelvr, ninvr )
               W_el( i ) = zero
               k = k + ninvr

!  Check to see if any of the columns belong to duplicated variables

               ii = IELVAR( is + i )
               DO j = 1, i - 1
                 IF ( IELVAR( is + j ) == ii ) GO TO 300
               END DO
               CYCLE

!  Amalgamate columns from duplicate variables

  300          CONTINUE
               kk = lwfree + ( j - 1 ) * ninvr - 1
               ll = k - ninvr
               WTRANS( kk + 1 : kk + ninvr ) =                                 &
                 WTRANS( kk + 1 : kk + ninvr ) + WTRANS( ll + 1 : ll + ninvr )
               WTRANS( ll + 1 : ll + ninvr ) = zero
             END DO

!  Compare this transformation matrix with previous ones

         LI: DO i = 1, ntype
               IF ( ITRANS( ISSITR( i ) ) /= ninvr .OR.                        &
                    ITRANS( ISSITR( i ) + 1 ) /= nelvr ) CYCLE LI
               DO j = 0, ninvr * nelvr - 1
                 IF ( WTRANS( lwfree + j ) /=                                  &
                      WTRANS( ISSWTR( i ) + j ) ) CYCLE LI
               END DO

!  The transformation is an existing one. Record which.

               ITYPER( iel ) = i
               CYCLE LIEL
             END DO LI

             mitran = lifree + ninvr + nelvr + 1
                  
!  Ensure that there is enough space

             IF ( mitran > litran ) THEN
               nitran = 2 * ( lifree + ninvr + nelvr + 1 )
               uitran = lifree - 1
!              array = 'ITRANS'
!              CALL EXTEND_array( array, litran, uitran, nitran,               &
!                                 mitran, status, iprint, iout )
               CALL EXTEND_arrays( ITRANS, litran, uitran, nitran, mitran,     &
                                   buffer, status, alloc_status )
               IF ( status /= 0 ) THEN 
                 bad_alloc = 'ITRANS' ; GO TO 610 ; END IF
               litran = nitran
             END IF

!  The transformation defines a new type. Record its details

             ntype = ntype + 1
             ITYPER( iel ) = ntype
             ITRANS( lifree ) = ninvr
             ITRANS( lifree + 1 ) = nelvr
             ITRANS( lifree + 2 : mitran + 1 ) = 0
             ISSITR( ntype ) = lifree
             ISSWTR( ntype ) = lwfree
             lifree = lifree + 2 + ninvr + nelvr
             lwfree = lwfree + ninvr * nelvr
           ELSE
             ITYPER( iel ) = 0
           END IF
         END DO LIEL

!  For each type of element with internal variables:

         DO i = 1, ntype
           lwfree = ISSWTR( i ) ; lifree = ISSITR( i )
           ninvr = ITRANS( lifree )
           nelvr = ITRANS( lifree + 1 )

!  Factorize W. Use Gaussian elimination with complete pivoting.
!  Determine the "most independent" set of columns of W

           CALL OTHERS_gauss_elim(                                             &
               ninvr, nelvr, ITRANS( lifree + 2 : lifree + ninvr + 1 ),        &
               ITRANS( lifree + ninvr + 2 : lifree + ninvr + nelvr + 1 ),      &
               WTRANS( lwfree : lwfree + ninvr * nelvr - 1 ) )
      
         END DO

!  Compress the data structures to remove redundant information

!  Compress integer data

         litran = 0
         lwtran = 0
         DO i = 1, ntype
           liwfro = ISSITR( i ) - 1
           ninvr = ITRANS( liwfro + 1 )
           k = 2 * ( ninvr + 1 )
           DO j = 1, k
              ITRANS( litran + j ) = ITRANS( liwfro + j )
           END DO
           ISSITR( i ) = litran + 1
           litran = litran + k

!  Compress real data

           lwfreo = ISSWTR( i ) - 1
           DO j = 1, ninvr * ninvr
              WTRANS( lwtran + j ) = WTRANS( lwfreo + j )
           END DO
           ISSWTR( i ) = lwtran + 1
           lwtran = lwtran + ninvr * ninvr
         END DO

!  ----------------------------------------------------------------------
!  The list of variables is allocated to nsets disjoints sets. Variable I
!  occurs in set ISET
!  ----------------------------------------------------------------------

         reallocate = .TRUE.
         IF ( ALLOCATED( ISET ) ) THEN
           IF ( SIZE( ISET ) < n ) THEN ; DEALLOCATE( ISET )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ISET( n ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
             bad_alloc = 'ISET' ; GO TO 600 ; END IF
         END IF
         
         reallocate = .TRUE.
         IF ( ALLOCATED( ISVSET ) ) THEN
           IF ( SIZE( ISVSET ) < n + 2 ) THEN ; DEALLOCATE( ISVSET )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ISVSET( n + 2 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
             bad_alloc = 'ISVSET' ; GO TO 600 ; END IF
         END IF
         
         reallocate = .TRUE.
         IF ( ALLOCATED( INVSET ) ) THEN
           IF ( SIZE( INVSET ) < MAX( n + 1, nel ) ) THEN ; DEALLOCATE( INVSET )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( INVSET( MAX( n + 1, nel ) ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
             bad_alloc = 'INVSET' ; GO TO 600 ; END IF
         END IF

!  Assign initial set numbers to each variable

         nsets = 0
         ISET( : n ) = n

!  Use the Curtis-Powell-Reid algorithm to determine which set each variable
!  belongs to. Loop over the variables.

         DO i = 1, n

!  Loop over the elements which use variable i. The elements are obtained from
!  a linked-list

           vrused = .FALSE.
           ipt = LINK_elem_uses_var( i )
           IF ( ipt >= 0 ) THEN
             iell = LIST_elements( i )
  420        CONTINUE
             iel = IELING( iell )
             itype = ITYPER( iel )
!            WRITE( 6, * ) ' element ', iel

!  Check that the variable belongs to the "independence" set of elements with
!  internal variables

             IF ( itype > 0 ) THEN
               lifree = ISSITR( itype )
               ninvr = ITRANS( lifree )
               DO j = 1, ninvr
                 k = j - 1
                 l = ITRANS( lifree + ninvr + 1 + j ) - 1
                 IF ( i == IELVAR( ISTAEV( iel ) + l ) ) GO TO 440
               END DO
               GO TO 450
  440          CONTINUE
             END IF
             vrused = .TRUE.
  450        CONTINUE

!  Loop over the complete list of variables used by element iel

!DIR$ IVDEP
             DO j = ISTAEV( iel ), ISTAEV( iel + 1 ) - 1

!  If variable IV is used, flag the set which contains it

               INVSET( ISET( IELVAR( j ) ) ) = 1
             END DO

!  Check the link-list to see if further elements use the variable

             IF ( ipt > 0 ) THEN
               iell = LIST_elements( ipt )
               ipt = LINK_elem_uses_var( ipt )
               GO TO 420
             END IF
           END IF

!  See if the variable may be placed in the first nsets sets

           IF ( vrused ) THEN
             DO j = 1, nsets
               IF ( INVSET( j ) == 0 ) GO TO 480
               INVSET( j ) = 0
             END DO

!  The variable needs a new set

             nsets = nsets + 1

!  The variable will be placed in set j

             j = nsets

  480        CONTINUE
             ISET( i ) = j

!  Reset the flags to zero

             INVSET( j : nsets ) = 0
           ELSE

!  The variable is not to be used

             ISET( i ) = n
           END IF
         END DO

!  Check that there is at least one set

         IF ( nsets /= 0 ) THEN
           ISET( : n ) = MIN( ISET( : n ), nsets + 1 )

!  ---------------------------------------------------------------------
!  Obtain a list, INVSET, of the variables corresponding to each set
!  ---------------------------------------------------------------------

!  Clear ISVSET

           ISVSET( 2 : nsets + 2 ) = 0

!  Count the number of elements in each set and store in ISVSET.
!  Negate the set numbers in ISET, so that they are flagged as
!  ISET is gradually overwritten by variable indices.

           DO k = 1, n
             j = ISET( k )
             ISET( k ) = - j
             ISVSET( j + 1 ) = ISVSET( j + 1 ) + 1
           END DO

!  Compute the starting addresses for each set within IISET

           ISVSET( 1 ) = 1
           DO j = 2, nsets + 2
             ISVSET( j ) = ISVSET( j ) + ISVSET( j - 1 )
           END DO

!  Store in INVSET the variable whose set number is the
!  ISVSET( j )-th entry of INVSET

           isofar = 0
           DO j = 1, nsets + 1
             istarj = ISVSET( j )
             DO ivarp1 = isofar + 1, n
                IF ( istarj < ivarp1 ) GO TO 530
             END DO
             ivarp1 = n + 1
  530        CONTINUE
             isofar = ivarp1 - 1
             INVSET( j ) = isofar
           END DO

!  Reorder the elements into set order. Fill in each set from the front. As a
!  new entry is placed in set K increase the pointer ISVSET( k ) by one
!  and find the new variable, INVSET( k ), that corresponds to the set now
!  pointed to by ISVSET( k )

           DO j = 1, nsets + 1

!  Determine the next unplaced entry, ISVSET, in ISET

  560        CONTINUE
             istrt = ISVSET( j )
             IF ( istrt == ISVSET( j + 1 ) ) CYCLE
             IF ( ISET( istrt ) > 0 ) CYCLE

!  Extract the variable and set numbers of the starting element

             ivar = INVSET( j )
             jset = - ISET( istrt )

!  Move elements in a cycle, ending back at set J

             DO k = istrt, n

!  Find the first empty location in set JSET in INVSET

               inext = ISVSET( jset )

!  Extract the variable index of the next element

               newvar = INVSET( jset )

!  Update ISVSET( jset ), find the new variable index and store it in
!  INVSET( jset )

               istarj = inext + 1
               ISVSET( jset ) = istarj
               DO ivarp1 = newvar + 1, n
                  IF ( istarj < ivarp1 ) GO TO 570
               END DO
               ivarp1 = n + 1
  570          CONTINUE
               INVSET( jset ) = ivarp1 - 1
               IF ( jset == j ) EXIT

!  Extract the number of the set of the next element

               newset = - ISET( inext )

!  Store the variable index of the current element

               ISET( inext ) = ivar

!  Make the next element into the current one

               ivar = newvar
               jset = newset
             END DO

!  Store the variable index of the starting element

             ISET( istrt ) = ivar
             GO TO 560
           END DO

!  Revise ISVSET to point to the start of each set

           ISVSET( nsets + 1 : 2 : - 1 ) = ISVSET( nsets : 1 : - 1 )
           ISVSET( 1 ) = 1
         END IF

       ELSE

!  Allocate unused arrays to have length zero

         reallocate = .TRUE.
         IF ( ALLOCATED( ITYPER ) ) THEN
           IF ( SIZE( ITYPER ) /= 0 ) THEN ; DEALLOCATE( ITYPER )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ITYPER( 0 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'ITYPER' ; GO TO 600 ; END IF
         END IF

         IF ( ALLOCATED( ISSWTR ) ) THEN
           IF ( SIZE( ISSWTR ) /= 0 ) THEN ; DEALLOCATE( ISSWTR )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ISSWTR( 0 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'ISSWTR' ; GO TO 600 ; END IF
         END IF

         reallocate = .TRUE.
         IF ( ALLOCATED( ISSITR ) ) THEN
           IF ( SIZE( ISSITR ) /= 0 ) THEN ; DEALLOCATE( ISSITR )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ISSITR( 0 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
             bad_alloc = 'ISSITR' ; GO TO 600 ; END IF
         END IF

         reallocate = .TRUE.
         lwtran = 0
         IF ( ALLOCATED( WTRANS ) ) THEN
           IF ( SIZE( WTRANS ) /= 0 ) THEN ; DEALLOCATE( WTRANS )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( WTRANS( lwtran ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN
             bad_alloc = 'WTRANS' ; GO TO 600 ; END IF
         END IF
         
         reallocate = .TRUE.
         litran = 0
         IF ( ALLOCATED( ITRANS ) ) THEN
           IF ( SIZE( ITRANS ) /= 0 ) THEN ; DEALLOCATE( ITRANS )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ITRANS( litran ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN ; 
             bad_alloc = 'ITRANS' ; GO TO 600 ; END IF
         END IF

         reallocate = .TRUE.
         IF ( ALLOCATED( ISET ) ) THEN
           IF ( SIZE( ISET ) /= 0 ) THEN ; DEALLOCATE( ISET )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ISET( 0 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
             bad_alloc = 'ISET' ; GO TO 600 ; END IF
         END IF
         
         reallocate = .TRUE.
         IF ( ALLOCATED( ISVSET ) ) THEN
           IF ( SIZE( ISVSET ) /= 0 ) THEN ; DEALLOCATE( ISVSET )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( ISVSET( 0 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
             bad_alloc = 'ISVSET' ; GO TO 600 ; END IF
         END IF
         
         reallocate = .TRUE.
         IF ( ALLOCATED( INVSET ) ) THEN
           IF ( SIZE( INVSET ) /= 0 ) THEN ; DEALLOCATE( INVSET )
           ELSE ; reallocate = .FALSE. ; END IF
         END IF
         IF ( reallocate ) THEN 
           ALLOCATE( INVSET( 0 ), STAT = alloc_status )
           IF ( alloc_status /= 0 ) THEN 
             bad_alloc = 'INVSET' ; GO TO 600 ; END IF
         END IF

       END IF

!  Set the length of the remaining partitions of the workspace for array bound
!  checking in calls to other subprograms

!  -- Set the starting addresses for the partitions within FUVALS --

!  A full description of the partitions of FUVALS is given in the introductory
!  comments to the LANCELOT package

       lfxi   = 0
       lgxi   = lfxi + nel
       lhxi   = INTVAR( nel1 ) - 1
       lggfx  = lggfx - 1
       ldx    = lggfx + n

!  Print all of the starting addresses for the workspace array partitions

       IF ( iprint >= 3 ) WRITE( iout,                                         &
            "( /,' Starting addresses for the partitions of FUVALS ', /,       &
         &       ' ----------------------------------------------- ', //,      &
         &       '   lfxi   lgxi   lhxi  lggfx ','   ldx ', /, 5I7 )" )        &
           lfxi, lgxi, lhxi, lggfx, ldx

!  Set the length of each partition of the real workspace array FUVALS for
!  array bound checking in calls to other subprograms

       lnguvl = MAX( 1, lhxi - lfxi )
       lnhuvl = MAX( 1, lggfx - lfxi )
       status = 0
       RETURN

!  Unsuccessful returns

  600  CONTINUE
       status = 1000 + alloc_status

  610  CONTINUE
       WRITE( iout, 2600 ) TRIM( bad_alloc ), alloc_status
       RETURN

!  Non-executable statements

 2600    FORMAT( ' ** Message from -INITW_initialize_workspace-', /,          &
                 ' Allocation error, for ', A, ', status = ', I0 )

!  End of subroutine INITW_initialize_workspace

       END SUBROUTINE INITW_initialize_workspace

!  End of module LANCELOT_INITW

     END MODULE LANCELOT_INITW_double
