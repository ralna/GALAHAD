! THIS VERSION: GALAHAD 2.1 - 4/02/2008 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*  test_mop  P R O G R A M  *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Daniel Robinson

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

PROGRAM test_mop

! ***************************************************************
!                                                               ! 
!              Test function for subroutine mop_Ax              !
!                                                               !
! ***************************************************************
!                                                               !
! Verifies that the subroutine mop_Ax is working properly by    !
! performing a number of matrix vector multiplications for      !
! problems with known solutions.  The only variable that the    !
! user should ever need to change (and they never should have   !
! to) is given by:                                              !
!                                                               !
!   print_level  scalar variable of type integer that           !
!                controls level of information to be            !
!                printed.  Values and their meanings are:       !
!                                                               !
!                print_level = 0   nothing is printed.          !
!                print_level = 1   summary of results.          !
!                print_level = 2   Above, in addition to        !
!                                  individual problem results.  !
!                                                               !
!                Default value is print_level = 1.              !
!                                                               !
!****************************************************************

  USE GALAHAD_SMT_double
  USE GALAHAD_MOP_double

  IMPLICIT NONE
   
!  Define the working precision to be double

  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set parameters

  REAL ( KIND = wp ), PARAMETER :: zero           = 0.0_wp
  REAL ( KIND = wp ), PARAMETER :: one            = 1.0_wp
  REAL ( KIND = wp ), PARAMETER :: two            = 2.0_wp
  REAL ( KIND = wp ), PARAMETER :: three          = 3.0_wp
  REAL ( KIND = wp ), PARAMETER :: four           = 4.0_wp
  REAL ( KIND = wp ), PARAMETER :: five           = 5.0_wp
  REAL ( KIND = wp ), PARAMETER :: six            = 6.0_wp
  REAL ( KIND = wp ), PARAMETER :: ten            = 10.0_wp
  REAL ( KIND = wp ), PARAMETER :: twelve         = 12.0_wp
  REAL ( KIND = wp ), PARAMETER :: fifteen        = 15.0_wp
  REAL ( KIND = wp ), PARAMETER :: seventeen      = 17.0_wp
  REAL ( KIND = wp ), PARAMETER :: twentytwo      = 22.0_wp
  REAL ( KIND = wp ), PARAMETER :: fiftyfive      = 55.0_wp
  REAL ( KIND = wp ), PARAMETER :: sixtyfour      = 64.0_wp
  REAL ( KIND = wp ), PARAMETER :: seventyseven   = 77.0_wp
  REAL ( KIND = wp ), PARAMETER :: ninetyfour     = 94.0_wp
  REAL ( KIND = wp ), PARAMETER :: hundredfifteen = 115.0_wp

! Interfaces

  INTERFACE IAMAX
      FUNCTION IDAMAX( n, X, incx )
       INTEGER :: IDAMAX
       INTEGER, INTENT( IN ) :: n, incx
       DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n-1 ) + 1 ) :: X
     END FUNCTION IDAMAX
   END INTERFACE

!***************************************************************
!                                                              !
!  BEGIN:  PARAMETERS THAT MAY BE MODIFIED BY AN INFORMED USER !
!                                                              !
!***************************************************************

  INTEGER, PARAMETER :: print_level = 1

!*************************************************************
!                                                            !
!  END:  PARAMETERS THAT MAY BE MODIFIED BY THE USER         !
!                                                            !
!*************************************************************

! Parameters

  INTEGER, PARAMETER :: out       = 6
  INTEGER, PARAMETER :: error     = 6
  INTEGER, PARAMETER :: out_opt   = 6
  INTEGER, PARAMETER :: error_opt = 6
  REAL( KIND = wp ),PARAMETER :: tol = ten**(-12)

! Local varibles

  INTEGER :: print_level_mop_Ax, print_level_mop_getval, stat
  INTEGER :: tally, row_tally, col_tally, i, j, m, n
  INTEGER :: nprob, mx_type, storage_number, err_loc, number_wrong
  REAL( KIND = wp ) :: max_error, value, err, val1, val2
  REAL( KIND = wp ) :: sqrt3, alpha, beta
  REAL( KIND = wp ), DIMENSION( 1:5 ) :: R_sol, R_sol_trans, Rfx, R, X
  REAL( KIND = wp ), DIMENSION( 1:5 , 1:5 ) :: A
  LOGICAL :: symm, trans
  REAL( KIND = wp ), DIMENSION(:), ALLOCATABLE :: u, v
  TYPE( SMT_type ) :: B, B2, uB, Bv, uBv

! set sqrt( 3 )

  sqrt3 = sqrt( three )

! intialize some variables
 
  max_error    = zero
  err          = one
  number_wrong = 0
  nprob        = 1
   
! set values for optional input values to mop_Ax.

  print_level_mop_Ax     = 0
  print_level_mop_getval = 0

  symm        = .TRUE.  ! dummy value
  trans       = .TRUE.  ! dummy value

! load A

!******************************************
!                                         !
!          |   1   2   3   4   5   |      ! 
!          |   6   7   8   9  10   |      !
!     A =  |  11  12  13  14  15   |      !
!          |  16  17  18  19  20   |      !
!          |  21  22  23  24  25   |      !
!                                         !
!******************************************  

  value = one

  DO i = 1, 5
     DO j = 1, 5
        A( i, j ) = value
        value = value + one
     END DO
  END DO

! Allocate matrix B : over-estimation in general.

  ALLOCATE( B%val( 1:25 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)
  
  ALLOCATE( B%row( 1:25 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)
  
  ALLOCATE( B%col( 1:25 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)
  
  ALLOCATE( B%ptr( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)
  
  ALLOCATE( B%id( 1:25 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  CALL SMT_put( B%id, 'Matrix id goes here!', stat )
  IF ( stat /= 0 ) WRITE( error, 1003 )

  !B%id( 1:20 ) = 'Matrix id goes here!'

! Define X, R, and Rfx to be a constant vector of  sqrt(3).

  DO i = 1, 5
     X( i ) = sqrt3;    R( i ) = sqrt3;    Rfx( i ) = sqrt3
  END DO

!*************************************************************
!  BEGIN : TESTING mop_Ax                                    *
!*************************************************************

! First, test the storage format DIAGONAL.
! ***************************************

  ! First assume alpha = beta = one

  alpha = one
  beta  = one

  ! First problem "A" = DIAG( A )
  ! ****************************

  ! Define necessary storage components.

  B%m = 5
  B%n = 5
  CALL SMT_put( B%type, 'DIAGONAL', stat )
  IF ( stat /= 0 ) WRITE( error, 1003 )

  DO i = 1, 5
     B%val( i ) = A( i, i )
  END DO
  
  ! the solution
  
  DO i = 1, 5
     R_sol( i ) = sqrt3 * ( one + A( i, i ) )
  END DO
  
  ! get solution from mop_Ax
  
  CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
               print_level_mop_Ax, symm, trans )
  
  err_loc   = IAMAX( 5, R_sol - R, 1 )
  err       = ABS( R_sol( err_loc ) - R( err_loc ) )
  max_error = MAX( err, max_error )
  
  IF ( err > tol ) THEN

     number_wrong = number_wrong + 1

     IF ( print_level >= 2 ) THEN
        WRITE( out, 2000 ) nprob
     END IF

  ELSE
 
     IF ( print_level >= 2 ) THEN
        WRITE( out, 2001 ) nprob
     END IF

  END IF

  nprob = nprob + 1
     
  ! re-load R
  
  R = Rfx

  ! Second problem.
  ! *****************

  ! Same matrix, just testing optional parameters.
  
  CALL mop_Ax( alpha, B, X, beta, R )
  
  err_loc   = IAMAX( 5, R_sol - R, 1 )
  err       = ABS( R_sol( err_loc ) - R( err_loc ) )
  max_error = MAX( err, max_error )

  IF ( err > tol ) THEN
     
     number_wrong = number_wrong + 1
     
     IF ( print_level >= 2 ) THEN
        WRITE( out, 2000 ) nprob
     END IF
     
  ELSE
     
     IF ( print_level >= 2 ) THEN
        WRITE( out, 2001 ) nprob
     END IF
     
  END IF
  
  nprob = nprob + 1
  
  ! re-load R
  
  R = Rfx

  ! Third problem ( alpha = 2, beta = 3 ) 
  ! *************************************

  alpha = two
  beta  = three

  ! solution

  DO i = 1,5
     R_sol( i ) = sqrt3 * ( five + REAL(i-1) * twelve )
  END DO

  ! get solution from mop_Ax
  
  CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
               print_level_mop_Ax, symm, trans )
  
  err_loc   = IAMAX( 5, R_sol - R, 1 )
  err       = ABS( R_sol( err_loc ) - R( err_loc ) )
  max_error = MAX( err, max_error )
  
  IF ( err > tol ) THEN
     
     number_wrong = number_wrong + 1
     
     IF ( print_level >= 2 ) THEN
        WRITE( out, 2000 ) nprob
     END IF
     
  ELSE
     
     IF ( print_level >= 2 ) THEN
        WRITE( out, 2001 ) nprob
     END IF
     
  END IF

  nprob = nprob +  1
  
  ! re-load R
  
  R = Rfx

  ! Fourth problem ( reset alpha = beta = one )
  ! *****************

  alpha = one
  beta  = one

  ! Test with zeros, i.e. "A" = 0
  
  B%val = zero
  
  ! solution
  
  R_sol = sqrt3
  
  
  ! get solution from mop_Ax
  
  CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
               print_level_mop_Ax, symm, trans )

  err_loc   = IAMAX( 5, R_sol - R, 1 )
  err       = ABS( R_sol( err_loc ) - R( err_loc ) )
  max_error = MAX( err, max_error )

  IF ( err > tol ) THEN
     
     number_wrong = number_wrong + 1
     
     IF ( print_level >= 2 ) THEN
        WRITE( out, 2000 ) nprob
     END IF
     
  ELSE
     
     IF ( print_level >= 2 ) THEN
        WRITE( out, 2001 ) nprob
     END IF
     
  END IF

  nprob = nprob +  1
  
  ! re-load R
  
  R = Rfx

! *****************************************************************
! Proceed testing of remaining storage types ( alpha = beta = one )
! *****************************************************************

  alpha = one
  beta  = one
  
  DO mx_type = 1, 6
     
     SELECT CASE ( mx_type )
        
     CASE ( 1 )
        
        ! short and wide
        
        m = 3
        n = 5
        symm = .FALSE.

     CASE ( 2 )

        ! tall and skinny

        m = 5
        n = 3
        symm = .FALSE.

     CASE ( 3 )

        ! square/symmetric

        m = 5
        n = 5
        symm = .TRUE.

     CASE ( 4 )

        ! short and wide ( with zero row and column )

        m = 3
        n = 5
        symm = .FALSE.

        A( 2, 1:5 ) = zero
        A( 1:5, 2 ) = zero

        ! remaining cases inherit this new "A".

     CASE ( 5 )

        ! tall and skinny ( with zero row and column )

        m = 5
        n = 3
        symm = .FALSE.

     CASE ( 6 )

        ! square/symmetric ( with zero row and column )

        m = 5
        n = 5
        symm = .TRUE.
        
     END SELECT

     ! the solutions for both "A" and "A^T" cases

     IF ( symm .AND. mx_type == 3 ) THEN

        R_sol( 1 ) = sqrt3 + sqrt3 * fiftyfive
        R_sol( 2 ) = sqrt3 + sqrt3 * sixtyfour
        R_sol( 3 ) = sqrt3 + sqrt3 * seventyseven
        R_sol( 4 ) = sqrt3 + sqrt3 * ninetyfour
        R_sol( 5 ) = sqrt3 + sqrt3 * hundredfifteen

        R_sol_trans = R_sol

     ELSEIF ( symm .AND. mx_type == 6 ) THEN

        R_sol( 1 ) = sqrt3 + sqrt3 * fiftyfive - sqrt3 * six
        R_sol( 2 ) = sqrt3
        R_sol( 3 ) = sqrt3 + sqrt3 * seventyseven - sqrt3 * twelve
        R_sol( 4 ) = sqrt3 + sqrt3 * ninetyfour - sqrt3 * seventeen
        R_sol( 5 ) = sqrt3 + sqrt3 * hundredfifteen - sqrt3 * twentytwo

        R_sol_trans = R_sol
        
     ELSE

        DO i = 1, m
           R_sol( i ) = sqrt3 + sqrt3 * SUM( A( i, 1:n ) )
        END DO
        DO i = 1, n
           R_sol_trans( i ) = sqrt3 + sqrt3 * SUM( A( 1:m , i ) )
        END DO

     END IF

     ! Matrix characterists are have now been defined.
     ! Now test the different storage formats.

     ! Form storage structure for B.

     DO storage_number = 1, 4

        B%m = m
        B%n = n

        IF ( storage_number == 1 ) THEN
           CALL SMT_put( B%type, 'DENSE', stat )
        ELSEIF ( storage_number == 2 ) THEN
           CALL SMT_put( B%type, 'SPARSE_BY_ROWS', stat )
        ELSEIF ( storage_number == 3 ) THEN
           CALL SMT_put( B%type, 'SPARSE_BY_COLUMNS', stat )
        ELSEIF ( storage_number == 4 ) THEN
           CALL SMT_put( B%type, 'COORDINATE', stat )
        END IF

        SELECT CASE ( SMT_get( B%type ) )

        CASE ( 'DENSE' )

           IF ( symm ) THEN
              
              tally = 1
              
              DO i = 1, m
                 DO j = 1, i
                    B%val( tally ) = A( i, j )
                    tally = tally + 1
                 END DO
              END DO
              
           ELSE
              
              DO i = 1, m
                 DO j = 1, n
                    B%val( n*(i-1) + j ) = A( i, j )
                 END DO
              END DO
              
           END IF
           
        CASE ( 'SPARSE_BY_ROWS' )

        tally = 1

        IF( symm ) THEN

           B%ptr( 1 ) = 1
           
           DO i = 1, m
              
              row_tally = 0
              
              DO j = 1, i
                 
                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%col( tally ) = j
                    row_tally = row_tally + 1
                    tally = tally + 1
                 END IF

              END DO
              
              B%ptr( i + 1 ) = B%ptr( i ) + row_tally
              
           END DO
           
        ELSE

           B%ptr( 1 ) = 1
           
           DO i = 1, m
              
              row_tally = 0

              DO j = 1, n

                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%col( tally ) = j
                    row_tally = row_tally + 1
                    tally = tally + 1
                 END IF
              END DO
              
              B%ptr( i + 1 ) = B%ptr( i ) + row_tally
              
           END DO
           
        END IF
        
     CASE ( 'SPARSE_BY_COLUMNS' )

        tally = 1

        IF( symm ) THEN

           B%ptr( 1 ) = 1
           
           DO j = 1, n
              
              col_tally = 0

              DO i = j, m
                 
                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%row( tally ) = i
                    col_tally = col_tally + 1
                    tally = tally + 1
                 END IF

              END DO
              
              B%ptr( j + 1 ) = B%ptr( j ) + col_tally
              
           END DO

           
        ELSE

           B%ptr( 1 ) = 1
           
           DO j = 1, n
              
              col_tally = 0

              DO i = 1, m

                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%row( tally ) = i
                    col_tally = col_tally + 1
                    tally     = tally + 1
                 END IF

              END DO
              
              B%ptr( j + 1 ) = B%ptr( j ) + col_tally
              
           END DO
           
        END IF

     CASE ( 'COORDINATE' )

        tally = 1

        IF( symm ) THEN

           DO i = 1, m
              DO j = 1, i
                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%row( tally ) = i
                    B%col( tally ) = j
                    tally          = tally + 1
                 END IF
              END DO
           END DO
           
        ELSE

           DO i = 1, m
              DO j = 1, n
                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%row( tally ) = i
                    B%col( tally ) = j
                    tally          = tally + 1
                 END IF
              END DO
           END DO
           
        END IF

        B%ne = tally-1

     CASE DEFAULT

        WRITE(error, 1002 )

     END SELECT

     ! Get ( not transposed ) solution from mop_Ax: r <- r + Ax

     ! New problem.
     !*************

     trans = .FALSE.
     
     CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
                  print_level_mop_Ax, symm, trans )

     err_loc   = IAMAX( m, R_sol - R, 1 )
     err       = ABS( R_sol( err_loc ) - R( err_loc ) )
     max_error = MAX( err, max_error )

     IF ( err > tol ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1
     
     ! re-load R
     
     R = Rfx

     ! Get ( transposed ) solution from mop_Ax: r <- r + A^T x
     
     ! New problem.
     !*************

     trans = .TRUE.
     
     CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
                  print_level_mop_Ax, symm, trans )

     err_loc   = IAMAX( n, R_sol_trans - R, 1 )
     err       = ABS( R_sol_trans( err_loc ) - R( err_loc ) )
     max_error = MAX( err, max_error )

     IF ( err > tol ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1
     
     ! re-load R
     
     R = Rfx
     
  END DO
  
END DO

! *****************************************************************
! Same tests, except now alpha = 3
! *****************************************************************

alpha = three
beta  = one

! Re-load original "A" ( the zero rows were added above in CASE 4.

!******************************************
!                                         !
!          |   1   2   3   4   5   |      ! 
!          |   6   7   8   9  10   |      !
!     A =  |  11  12  13  14  15   |      !
!          |  16  17  18  19  20   |      !
!          |  21  22  23  24  25   |      !
!                                         !
!******************************************  

value = one

DO i = 1, 5
   DO j = 1, 5
      A( i, j ) = value
      value = value + one
   END DO
END DO

DO mx_type = 1, 6
     
   SELECT CASE ( mx_type )
      
   CASE ( 1 )
      
      ! short and wide
      
      m = 3
      n = 5
        symm = .FALSE.

     CASE ( 2 )

        ! tall and skinny

        m = 5
        n = 3
        symm = .FALSE.

     CASE ( 3 )

        ! square/symmetric

        m = 5
        n = 5
        symm = .TRUE.

     CASE ( 4 )

        ! short and wide ( with zero row and column )

        m = 3
        n = 5
        symm = .FALSE.

        A( 2, 1:5 ) = zero
        A( 1:5, 2 ) = zero

        ! remaining cases inherit this new "A".

     CASE ( 5 )

        ! tall and skinny ( with zero row and column )

        m = 5
        n = 3
        symm = .FALSE.

     CASE ( 6 )

        ! square/symmetric ( with zero row and column )

        m = 5
        n = 5
        symm = .TRUE.
        
     END SELECT

     ! the solutions for both "A" and "A^T" cases.  Note:
     ! Symmetric "A" is reflection of lower triangular part
     ! of A to the upper triangle.

     IF ( symm .AND. mx_type == 3 ) THEN

        R_sol(1) = sqrt3 + three * sqrt3 * fiftyfive
        R_sol(2) = sqrt3 + three * sqrt3 * sixtyfour
        R_sol(3) = sqrt3 + three * sqrt3 * seventyseven
        R_sol(4) = sqrt3 + three * sqrt3 * ninetyfour
        R_sol(5) = sqrt3 + three * sqrt3 * hundredfifteen

        R_sol_trans = R_sol

     ELSEIF ( symm .AND. mx_type == 6 ) THEN
 
        R_sol(1) = sqrt3 + three * ( sqrt3 * fiftyfive - sqrt3 * six )
        R_sol(2) = sqrt3
        R_sol(3) = sqrt3 + three * sqrt3 * ( seventyseven - twelve )
        R_sol(4) = sqrt3 + three * sqrt3 * ( ninetyfour - seventeen )
        R_sol(5) = sqrt3 + three * sqrt3 * ( hundredfifteen - twentytwo )

        R_sol_trans = R_sol
        
     ELSE

        DO i = 1, m
           R_sol( i ) = sqrt3 + three * sqrt3 * SUM( A( i, 1:n ) )
        END DO
        DO i = 1, n
           R_sol_trans( i ) = sqrt3 + three * sqrt3 * SUM( A( 1:m , i ) )
        END DO

     END IF

     ! Matrix characterists are have now been defined.
     ! Now test the different storage formats.

     ! Form storage structure for B.

     DO storage_number = 1, 4

        B%m = m
        B%n = n

        IF ( storage_number == 1 ) THEN
           CALL SMT_put( B%type, 'DENSE', stat )
        ELSEIF ( storage_number == 2 ) THEN
           CALL SMT_put( B%type, 'SPARSE_BY_ROWS', stat )
        ELSEIF ( storage_number == 3 ) THEN
           CALL SMT_put( B%type, 'SPARSE_BY_COLUMNS', stat )
        ELSEIF ( storage_number == 4 ) THEN
           CALL SMT_put( B%type, 'COORDINATE', stat )
        END IF


        SELECT CASE ( SMT_get( B%type ) )

        CASE ( 'DENSE' )

           IF ( symm ) THEN
              
              tally = 1
              
              DO i = 1, m
                 DO j = 1, i
                    B%val( tally ) = A( i, j )
                    tally = tally + 1
                 END DO
              END DO
              
           ELSE
              
              DO i = 1, m
                 DO j = 1, n
                    B%val( n*(i-1) + j ) = A( i, j )
                 END DO
              END DO
              
           END IF
           
        CASE ( 'SPARSE_BY_ROWS' )

        tally = 1

        IF( symm ) THEN

           B%ptr( 1 ) = 1
           
           DO i = 1, m
              
              row_tally = 0
              
              DO j = 1, i
                 
                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%col( tally ) = j
                    row_tally = row_tally + 1
                    tally = tally + 1
                 END IF

              END DO
              
              B%ptr( i + 1 ) = B%ptr( i ) + row_tally
              
           END DO
           
        ELSE

           B%ptr( 1 ) = 1
           
           DO i = 1, m
              
              row_tally = 0

              DO j = 1, n

                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%col( tally ) = j
                    row_tally = row_tally + 1
                    tally = tally + 1
                 END IF
              END DO
              
              B%ptr( i + 1 ) = B%ptr( i ) + row_tally
              
           END DO
           
        END IF
        
     CASE ( 'SPARSE_BY_COLUMNS' )

        tally = 1

        IF( symm ) THEN

           B%ptr( 1 ) = 1
           
           DO j = 1, n
              
              col_tally = 0

              DO i = j, m
                 
                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%row( tally ) = i
                    col_tally = col_tally + 1
                    tally = tally + 1
                 END IF

              END DO
              
              B%ptr( j + 1 ) = B%ptr( j ) + col_tally
              
           END DO

           
        ELSE

           B%ptr( 1 ) = 1
           
           DO j = 1, n
              
              col_tally = 0

              DO i = 1, m

                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%row( tally ) = i
                    col_tally = col_tally + 1
                    tally     = tally + 1
                 END IF

              END DO
              
              B%ptr( j + 1 ) = B%ptr( j ) + col_tally
              
           END DO
           
        END IF

     CASE ( 'COORDINATE' )

        tally = 1

        IF( symm ) THEN

           DO i = 1, m
              DO j = 1, i
                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%row( tally ) = i
                    B%col( tally ) = j
                    tally          = tally + 1
                 END IF
              END DO
           END DO
           
        ELSE

           DO i = 1, m
              DO j = 1, n
                 IF ( A( i, j ) /= zero ) THEN
                    B%val( tally ) = A( i, j )
                    B%row( tally ) = i
                    B%col( tally ) = j
                    tally          = tally + 1
                 END IF
              END DO
           END DO
           
        END IF

        B%ne = tally-1

     CASE DEFAULT

        WRITE(error, 1002 )

     END SELECT

     ! Get ( not transposed ) solution from mop_Ax: r <- r + Ax

     ! New problem.
     !*************

     trans = .FALSE.
     
     CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
                  print_level_mop_Ax, symm, trans )

     err_loc   = IAMAX( m, R_sol - R, 1 )
     err       = ABS( R_sol( err_loc ) - R( err_loc ) )
     max_error = MAX( err, max_error )

     IF ( err > tol ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1
     
     ! re-load R
     
     R = Rfx

     ! Get ( transposed ) solution from mop_Ax: r <- r + A^T x
     
     ! New problem.
     !*************

     trans = .TRUE.
     
     CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
                  print_level_mop_Ax, symm, trans )

     err_loc   = IAMAX( n, R_sol_trans - R, 1 )
     err       = ABS( R_sol_trans( err_loc ) - R( err_loc ) )
     max_error = MAX( err, max_error )

     IF ( err > tol ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1
     
     ! re-load R
     
     R = Rfx

  END DO

END DO

! ****************************************************************
! Finally, proceed with testing of alpha and beta
! *****************************************************************

m = 5
n = 5

! Load A with matrix of ones.

DO i = 1, m
   DO j = 1, n
      A( i, j ) = one
   END DO
END DO

symm = .TRUE.

! Load B using coordinate storage.  This is arbitrary.

tally = 1

IF( symm ) THEN
   
   DO i = 1, m
      DO j = 1, i
         IF ( A( i, j ) /= zero ) THEN
            B%val( tally ) = A( i, j )
            B%row( tally ) = i
            B%col( tally ) = j
            tally          = tally + 1
         END IF
      END DO
   END DO
   
ELSE
   
   DO i = 1, m
      DO j = 1, n
         IF ( A( i, j ) /= zero ) THEN
            B%val( tally ) = A( i, j )
            B%row( tally ) = i
            B%col( tally ) = j
            tally          = tally + 1
         END IF
      END DO
   END DO
   
END IF

B%ne = tally-1

! New problem ( alpha = 0, beta = 0 )
!************************************

alpha = zero
beta  = zero

! solution

R_sol = zero

! get solution from mop_Ax

CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
             print_level_mop_Ax, symm, trans )

! compute error

err_loc   = IAMAX( n, R_sol - R, 1 )
err       = ABS( R_sol( err_loc ) - R( err_loc ) )
max_error = MAX( err, max_error )

IF ( err > tol ) THEN
   
   number_wrong = number_wrong + 1
   
   IF ( print_level >= 2 ) THEN
      WRITE( out, 2000 ) nprob
   END IF
   
ELSE
   
   IF ( print_level >= 2 ) THEN
      WRITE( out, 2001 ) nprob
   END IF
   
END IF

nprob = nprob + 1

! re-load R

R = Rfx

! New problem ( alpha = 0, beta = 2 )
! ***********************************

alpha = zero
beta  =  two

! solution

R_sol = two * sqrt3

! get solution from mop_Ax

CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
             print_level_mop_Ax, symm, trans )

! compute error

err_loc   = IAMAX( n, R_sol - R, 1 )
err       = ABS( R_sol( err_loc ) - R( err_loc ) )
max_error = MAX( err, max_error )

IF ( err > tol ) THEN
   
   number_wrong = number_wrong + 1
   
   IF ( print_level >= 2 ) THEN
      WRITE( out, 2000 ) nprob
   END IF
   
ELSE
   
   IF ( print_level >= 2 ) THEN
      WRITE( out, 2001 ) nprob
   END IF
   
END IF

nprob = nprob + 1

! re-load R

R = Rfx

! New problem ( alpha = -3, beta = 0 )
!************************************

alpha = -three
beta  =  zero

! solution

R_sol = - fifteen * sqrt3

! get solution from mop_Ax

CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
             print_level_mop_Ax, symm, trans )

! compute error

err_loc   = IAMAX( n, R_sol - R, 1 )
err       = ABS( R_sol( err_loc ) - R( err_loc ) )
max_error = MAX( err, max_error )

IF ( err > tol ) THEN
   
   number_wrong = number_wrong + 1
   
   IF ( print_level >= 2 ) THEN
      WRITE( out, 2000 ) nprob
   END IF
   
ELSE
   
   IF ( print_level >= 2 ) THEN
      WRITE( out, 2001 ) nprob
   END IF
   
END IF

nprob = nprob + 1

! re-load R

R = Rfx

! New problem ( alpha = 5, beta = -3 )
!************************************

alpha =  five
beta  = -three

! solution

R_sol = twentytwo * sqrt3

! get solution from mop_Ax

CALL mop_Ax( alpha, B, X, beta, R, out_opt, error_opt, &
             print_level_mop_Ax, symm, trans )

! compute error

err_loc   = IAMAX( n, R_sol - R, 1 )
err       = ABS( R_sol( err_loc ) - R( err_loc ) )
max_error = MAX( err, max_error )

IF ( err > tol ) THEN
   
   number_wrong = number_wrong + 1
   
   IF ( print_level >= 2 ) THEN
      WRITE( out, 2000 ) nprob
   END IF
   
ELSE
   
   IF ( print_level >= 2 ) THEN
      WRITE( out, 2001 ) nprob
   END IF
   
END IF

nprob = nprob + 1

! re-load R

R = Rfx

! Print summary, if required.

  IF ( print_level >= 1 ) THEN

     WRITE( out, 3002 ) ! header
     WRITE( out, 3000 ) nprob-1, number_wrong
     
     IF ( print_level >= 2 ) THEN
        WRITE( out, 3001 ) tol, max_error
     END IF

     WRITE( out, 3003 ) ! footer

  END IF

!*************************************************************
!  END : TESTING mop_Ax                                      *
!*************************************************************

!*************************************************************
!  BEGIN : TESTING mop_getval                                *
!*************************************************************

  ! Reset some values
  
  nprob        = 1
  number_wrong = 0

  !*******************
  ! Test non-symmetric
  !*******************

  symm = .FALSE.

  ! The test matrix

  !******************!
  !                  !      
  !  A =  | 0  1  |  !
  !       | 2  3  |  !
  !                  !
  !******************!

  B%m = 2
  B%n = 2

  DO storage_number = 1, 5

     IF ( storage_number == 1 ) THEN

        CALL SMT_put( B%type, 'DENSE', stat )

        B%val(1) = zero
        B%val(2) = one
        B%val(3) = two
        B%val(4) = three

     ELSEIF ( storage_number == 2 ) THEN

        CALL SMT_put( B%type, 'SPARSE_BY_ROWS', stat )

        B%ptr(1) = 1
        B%ptr(2) = 2
        B%ptr(3) = 4

        B%col(1) = 2
        B%col(2) = 1
        B%col(3) = 2

        B%val(1) = one
        B%val(2) = two
        B%val(3) = three


     ELSEIF ( storage_number == 3 ) THEN

        CALL SMT_put( B%type, 'SPARSE_BY_COLUMNS', stat )

        B%ptr(1) = 1
        B%ptr(2) = 2
        B%ptr(3) = 4

        B%row(1) = 2
        B%row(2) = 1
        B%row(3) = 2

        B%val(1) = two
        B%val(2) = one
        B%val(3) = three

     ELSEIF ( storage_number == 4 ) THEN

        CALL SMT_put( B%type, 'COORDINATE', stat )

        B%row(1) = 1
        B%row(2) = 2
        B%row(3) = 2

        B%col(1) = 2
        B%col(2) = 1
        B%col(3) = 2

        B%val(1) = one
        B%val(2) = two
        B%val(3) = three

     END IF

     ! A2(1,1)
     !********

     CALL mop_getval( B, 1, 1, value, symm, out, error, print_level_mop_getval)

     IF ( value /= zero ) THEN

        number_wrong = number_wrong + 1

        IF ( print_level >=2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF

     ELSE

        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF

     END IF
     
     nprob = nprob + 1

     ! A2(1,2)
     !********

     CALL mop_getval( B, 1, 2, value, symm, out, error, print_level_mop_getval)
     
     IF ( value /= one ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >=2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1

     ! A2(2,1)
     !********

     CALL mop_getval( B, 2, 1, value, symm, out, error, print_level_mop_getval)
     
     IF ( value /= two ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >=2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1

     ! A2(2,2)
     !********

     CALL mop_getval( B, 2, 2, value, symm, out, error, print_level_mop_getval)
     
     IF ( value /= three ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >=2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1

  END DO
  
  !*******************
  ! Test symmetric
  !*******************

 ! The test matrix

  !******************!     
  !                  !    
  !  A =  | 0  1  |  !
  !       | 1  2  |  !
  !                  !
  !******************!

  B%m = 2
  B%n = 2

  symm = .TRUE.

  DO storage_number = 1, 5

     IF ( storage_number == 1 ) THEN

        CALL SMT_put( B%type, 'DENSE', stat )

        B%val(1) = zero
        B%val(2) = one
        B%val(3) = two

     ELSEIF ( storage_number == 2 ) THEN

        CALL SMT_put( B%type, 'SPARSE_BY_ROWS', stat )

        B%ptr(1) = 1
        B%ptr(2) = 1
        B%ptr(3) = 3

        B%col(1) = 1
        B%col(2) = 2

        B%val(1) = one
        B%val(2) = two

     ELSEIF ( storage_number == 3 ) THEN

        CALL SMT_put( B%type, 'SPARSE_BY_COLUMNS', stat )

        B%ptr(1) = 1
        B%ptr(2) = 2
        B%ptr(3) = 3

        B%row(1) = 2
        B%row(2) = 2

        B%val(1) = one
        B%val(2) = two

     ELSEIF ( storage_number == 4 ) THEN

        CALL SMT_put( B%type, 'COORDINATE', stat )

        B%row(1) = 2
        B%row(2) = 2

        B%col(1) = 1
        B%col(2) = 2

        B%val(1) = one
        B%val(2) = two

     END IF

     ! A2(1,1)
     !********

     CALL mop_getval( B, 1, 1, value, symm, out, error, print_level_mop_getval)

     IF ( value /= zero ) THEN

        number_wrong = number_wrong + 1

        IF ( print_level >=2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF

     ELSE

        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF

     END IF
     
     nprob = nprob + 1

     ! A2(1,2)
     !********

     CALL mop_getval( B, 1, 2, value, symm, out, error, print_level_mop_getval)
     
     IF ( value /= one ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >=2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1

     ! A2(2,1)
     !********

     CALL mop_getval( B, 2, 1, value, symm, out, error, print_level_mop_getval)
     
     IF ( value /= one ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >=2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1

     ! A2(2,2)
     !********

     CALL mop_getval( B, 2, 2, value, symm, out, error, print_level_mop_getval)
     
     IF ( value /= two ) THEN
        
        number_wrong = number_wrong + 1
        
        IF ( print_level >=2 ) THEN
           WRITE( out, 2000 ) nprob
        END IF
        
     ELSE
        
        IF ( print_level >= 2 ) THEN
           WRITE( out, 2001 ) nprob
        END IF
        
     END IF
     
     nprob = nprob + 1

  END DO

  ! Print summary, if required.

  IF ( print_level >= 1 ) THEN
     
     WRITE( out, 3004 ) ! header
     WRITE( out, 3000 ) nprob-1, number_wrong
     WRITE( out, 3005 ) ! footer

  END IF

  ! De-allocate arrays.

  DEALLOCATE( B%val, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B%row, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B%col, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B%ptr, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)

  DEALLOCATE( B%id, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B%type, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)

!*************************************************************
!  END : TESTING mop_getval                                *
!*************************************************************

!*************************************************************
!  BEGIN : TESTING mop_scaleA                               *
!*************************************************************

  nprob = 0
  number_wrong = 0

  ! Allocate and define matrices.
  
  ALLOCATE( B%val( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( uB%val( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( Bv%val( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( uBv%val( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( B2%val( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)
  
  ALLOCATE( B%row( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)
  
  ALLOCATE( uB%row( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)
  
  ALLOCATE( Bv%row( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( uBv%row( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( B2%row( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( B%col( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( uB%col( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( Bv%col( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( uBv%col( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( B2%col( 1:6 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)
  
  ALLOCATE( B%ptr( 1:4 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( B2%ptr( 1:3 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)
  
  ALLOCATE( B%id( 1:25 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( B2%id( 1:25 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( u( 1:2 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  ALLOCATE( v( 1:3 ), STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1000)

  CALL SMT_put( B2%id, 'This is copy of B.!', stat )
  IF ( stat /= 0 ) WRITE( error, 1003 )

  ! Define the solution for the following problem:
  ! B = | 1   3   4 |  with scalings u = | 2 |  and  v = | 2 |
  !     | 2   5   6 |                    |-3 |           | 4 |
  !                                                      | 6 |

  u(1) = 2
  u(2) = -3

  v(1) = 2
  v(2) = 4
  v(3) = 6
  
  ! First just scale by u on left.

  CALL SMT_put( uB%type, 'COORDINATE', stat )

  uB%ne = 6
  uB%m = 2
  uB%n = 3

  uB%row(1) = 1
  uB%col(1) = 1 
  uB%val(1) = 2

  uB%row(2) = 1
  uB%col(2) = 2
  uB%val(2) = 6

  uB%row(3) = 1
  uB%col(3) = 3
  uB%val(3) = 8

  uB%row(4) = 2
  uB%col(4) = 1
  uB%val(4) = -6

  uB%row(5) = 2
  uB%col(5) = 2
  uB%val(5) = -15

  uB%row(6) = 2
  uB%col(6) = 3
  uB%val(6) = -18

  ! Next just scale on right by v.

  CALL SMT_put( Bv%type, 'COORDINATE', stat )

  Bv%ne = 6
  Bv%m = 2
  Bv%n = 3

  Bv%row(1) = 1
  Bv%col(1) = 1 
  Bv%val(1) = 2

  Bv%row(2) = 1
  Bv%col(2) = 2
  Bv%val(2) = 12

  Bv%row(3) = 1
  Bv%col(3) = 3
  Bv%val(3) = 24

  Bv%row(4) = 2
  Bv%col(4) = 1
  Bv%val(4) = 4

  Bv%row(5) = 2
  Bv%col(5) = 2
  Bv%val(5) = 20

  Bv%row(6) = 2
  Bv%col(6) = 3
  Bv%val(6) = 36

  ! Finally scale by both u and v.

  CALL SMT_put( uBv%type, 'COORDINATE', stat )
  
  uBv%ne = 6
  uBv%m = 2
  uBv%n = 3

  uBv%row(1) = 1
  uBv%col(1) = 1 
  uBv%val(1) = 4

  uBv%row(2) = 1
  uBv%col(2) = 2
  uBv%val(2) = 24

  uBv%row(3) = 1
  uBv%col(3) = 3
  uBv%val(3) = 48

  uBv%row(4) = 2
  uBv%col(4) = 1
  uBv%val(4) = -12

  uBv%row(5) = 2
  uBv%col(5) = 2
  uBv%val(5) = -60

  uBv%row(6) = 2
  uBv%col(6) = 3
  uBv%val(6) = -108

  ! Now test each storage format one at a time.

  do storage_number = 1, 4

     ! The test matrix and a copy.

     B%m = 2
     B%n = 3

     if ( storage_number == 1 ) then

        CALL SMT_put( B%type, 'COORDINATE', stat )

        B%ne = 6
  
        B%row(1) = 1
        B%col(1) = 1 
        B%val(1) = 1
        
        B%row(2) = 1
        B%col(2) = 2
        B%val(2) = 3
        
        B%row(3) = 1
        B%col(3) = 3
        B%val(3) = 4
        
        B%row(4) = 2
        B%col(4) = 1
        B%val(4) = 2
        
        B%row(5) = 2
        B%col(5) = 2
        B%val(5) = 5
        
        B%row(6) = 2
        B%col(6) = 3
        B%val(6) = 6 
        
     elseif ( storage_number == 2 ) then

        CALL SMT_put( B%type, 'SPARSE_BY_ROWS', stat )

        B%col(1) = 1
        B%val(1) = 1

        B%col(2) = 2
        B%val(2) = 3

        B%col(3) = 3
        B%val(3) = 4

        B%col(4) = 1
        B%val(4) = 2

        B%col(5) = 2
        B%val(5) = 5

        B%col(6) = 3
        B%val(6) = 6

        B%ptr(1) = 1
        B%ptr(2) = 4
        B%ptr(3) = 7

     elseif ( storage_number == 3 ) then

        CALL SMT_put( B%type, 'DENSE', stat ) 

        B%val(1) = 1
        B%val(2) = 3
        B%val(3) = 4
        B%val(4) = 2
        B%val(5) = 5
        B%val(6) = 6

     elseif ( storage_number == 4 ) then

        CALL SMT_put( B%type, 'SPARSE_BY_COLUMNS', stat )

        B%row(1) = 1
        B%val(1) = 1

        B%row(2) = 2
        B%val(2) = 2

        B%row(3) = 1
        B%val(3) = 3

        B%row(4) = 2
        B%val(4) = 5

        B%row(5) = 1
        B%val(5) = 4

        B%row(6) = 2
        B%val(6) = 6

        B%ptr(1) = 1
        B%ptr(2) = 3
        B%ptr(3) = 5
        B%ptr(4) = 7

     end if

     ! Make a copy of B into B2.

     B2 = B

     ! First compute and check scaling of u on right.

     call mop_scaleA( B, u )

     do i = 1, 2
        do j = 1, 3
           call mop_getval(  B, i, j, val1 )
           call mop_getval( uB, i, j, val2 )
           nprob = nprob + 1
          ! write(*,*) abs(val1-val2)
           if ( abs(val1-val2) >= tol ) then
              number_wrong = number_wrong + 1
           end if
        end do
     end do

     B = B2

     call mop_scaleA( B, v=v )

     do i = 1, 2
        do j = 1, 3
           call mop_getval(  B, i, j, val1 )
           call mop_getval( Bv, i, j, val2 )
           nprob = nprob + 1
           !write(*,*) abs(val1-val2)
           if ( abs(val1-val2) >= tol ) then
              number_wrong = number_wrong + 1
           end if
        end do
     end do

     B = B2

     call mop_scaleA( B, u, v )

     do i = 1, 2
        do j = 1, 3
           call mop_getval(   B, i, j, val1 )
           call mop_getval( uBv, i, j, val2 )
           nprob = nprob + 1
           !write(*,*) abs(val1-val2)
           if ( abs(val1-val2) >= tol ) then
              number_wrong = number_wrong + 1
           end if
        end do
     end do

  end do

  ! Next, test it on a symmetric problem.
  ! *************************************

  ! Define the solution for the following problem:
  ! B = | 2   3 |  with scalings u = | 5 |
  !     | 3  -4 |                    |-1 |

  u(1) = 5
  u(2) = -1
 
  ! Perform the scaling of rows and columns by u.

  CALL SMT_put( uBv%type, 'COORDINATE', stat )
  
  uBv%ne = 3
  uBv%m  = 2
  uBv%n  = 2

  uBv%row(1) = 1
  uBv%col(1) = 1 
  uBv%val(1) = 50

  uBv%row(2) = 2
  uBv%col(2) = 1
  uBv%val(2) = -15

  uBv%row(3) = 2
  uBv%col(3) = 2
  uBv%val(3) = -4

  ! Now test each storage format one at a time.

  do storage_number = 1, 4

     ! The test matrix and a copy.

     B%m = 2
     B%n = 2

     if ( storage_number == 1 ) then

        CALL SMT_put( B%type, 'COORDINATE', stat )

        B%ne = 3
  
        B%row(1) = 1
        B%col(1) = 1 
        B%val(1) = 2
        
        B%row(2) = 2
        B%col(2) = 1
        B%val(2) = 3
        
        B%row(3) = 2
        B%col(3) = 2
        B%val(3) = -4
        
     elseif ( storage_number == 2 ) then

        CALL SMT_put( B%type, 'SPARSE_BY_ROWS', stat )

        B%col(1) = 1
        B%val(1) = 2

        B%col(2) = 1
        B%val(2) = 3

        B%col(3) = 2
        B%val(3) = -4

        B%ptr(1) = 1
        B%ptr(2) = 2
        B%ptr(3) = 4

     elseif ( storage_number == 3 ) then

        CALL SMT_put( B%type, 'DENSE', stat ) 

        B%val(1) = 2
        B%val(2) = 3
        B%val(3) = -4

     elseif ( storage_number == 4 ) then 

        CALL SMT_put( B%type, 'SPARSE_BY_COLUMNS', stat )
        
        B%row(1) = 1
        B%val(1) = 2
        
        B%row(2) = 2
        B%val(2) = 3

        B%row(3) = 2
        B%val(3) = -4

        B%ptr(1) = 1
        B%ptr(2) = 3
        B%ptr(3) = 4

     end if

     ! Compute and check solution from mop_scaleA.
     
     call mop_scaleA( B, u(1:2), symmetric=.true. )

     do i = 1, 2
        do j = 1, 2
           call mop_getval(   B, i, j, val1, symmetric=.true. )
           call mop_getval( uBv, i, j, val2, symmetric=.true. )
           nprob = nprob + 1
           !write(*,*) abs(val1-val2)
           if ( abs(val1-val2) >= tol ) then
              number_wrong = number_wrong + 1
           end if
        end do
     end do
 
  end do

  ! Finally, test a simple diagonal case.
  ! *************************************

  ! Define the solution for the following problem:
  ! B = | 2   0 |  with scalings u = | 5 |
  !     | 0  -4 |                    |-1 |
  
  CALL SMT_put( B%type, 'DIAGONAL', stat ) 

  B%m = 2
  B%n = 2 
  
  B%val(1) = 2
  B%val(2) = -4
    
  u(1) = 5
  u(2) = -1

  uBv%row(1) = 1
  uBv%row(2) = 2
  
  uBv%col(1) = 1
  uBv%col(2) = 2

  uBv%val(1) = 50
  uBv%val(2) = -4

  ! Compute and check solution from mop_scaleA.
     
  call mop_scaleA( B, u(1:2), symmetric=.true. )

  do i = 1, 2
     do j = 1, 2
        call mop_getval(   B, i, j, val1, symmetric=.true. )
        call mop_getval( uBv, i, j, val2, symmetric=.true. )
        nprob = nprob + 1
        !write(*,*) abs(val1-val2)
        if ( abs(val1-val2) >= tol ) then
           number_wrong = number_wrong + 1
        end if
     end do
  end do

  ! De-allocate arrays.

  DEALLOCATE( B%val, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B%row, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B%col, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B%ptr, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)

  DEALLOCATE( B%id, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B%type, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)

  DEALLOCATE( B2%val, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B2%row, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B2%col, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B2%ptr, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)

  DEALLOCATE( B2%id, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( B2%type, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)

  DEALLOCATE( uB%val, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( uB%row, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( uB%col, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( uB%type, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)

  DEALLOCATE( Bv%val, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( Bv%row, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( Bv%col, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( Bv%type, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( uBv%val, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( uBv%row, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( uBv%col, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)
  
  DEALLOCATE( uBv%type, STAT=stat )
  IF ( stat /= 0 ) WRITE( error, 1001)

  IF ( print_level >= 1 ) THEN
     
     WRITE( out, 3006 ) ! header
     WRITE( out, 3000 ) nprob-1, number_wrong
     WRITE( out, 3005 ) ! footer
     
  END IF

!*************************************************************
!  END : TESTING mop_scaleA                                 *
!*************************************************************

! Format statements

1000 FORMAT(1X,'ERROR : test_mop : allocation error.')
1001 FORMAT(1X,'ERROR : test_mop : de-allocation error.')
1002 FORMAT(1X,'ERROR : test_mop : improper storage format.')
1003 FORMAT(1X,'ERROR : test_mop : error while using SMT_put.')
2000 FORMAT(1X,'Problem ', i3, ' [BAD]')
2001 FORMAT(1X,'Problem ', i3, ' [OK]')
3000 FORMAT(/,5X,'Number of Problems Tested : ',i3,/     &
              5X,'Number of Problems Wrong  : ',i3,/     )
3001 FORMAT(5X,'Tolerance Used            : ',ES21.14,/  &
            5X,'Maximum Error Encountered : ',ES21.14,/  )
3002 FORMAT(/,2X,'*******************************************************',/  &
              2X,'TEST RESULTS FOR SUBROUTINE : mop_Ax')
3003 FORMAT(2X,  '*******************************************************' )
3004 FORMAT(/,2X,'*******************************************************',/  &
              2X,'TEST RESULTS FOR SUBROUTINE : mop_getval')
3005 FORMAT(2X,  '*******************************************************' )
3006 FORMAT(/,2X,'*******************************************************',/  &
              2X,'TEST RESULTS FOR SUBROUTINE : mop_scaleA') 

END PROGRAM test_mop

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                           *-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*   END PROGRAM  test_mop  P R O G R A M    *-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                           *-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
