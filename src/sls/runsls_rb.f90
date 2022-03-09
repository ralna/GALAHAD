  PROGRAM RUNRB_sls
    USE spral_rutherford_boeing
    USE GALAHAD_SLS_double

    IMPLICIT none

    INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
    INTEGER, PARAMETER :: long = SELECTED_INT_ kind( 18 )
    INTEGER, PARAMETER :: out = 6

!  local variables

    CHARACTER( LEN = 80 ) :: filename
    INTEGER :: info                    ! return code
    INTEGER :: m                       ! # rows
    INTEGER :: n                       ! # columns
    INTEGER ( KIND = long ) :: nelt    ! # elements (0 if asm)
    INTEGER ( KIND = long ) :: nvar    ! # indices in file
    INTEGER ( KIND = long ) :: nval    ! # values in file
    INTEGER :: matrix_type             ! SPRAL matrix type
    CHARACTER ( LEN = 3 ) :: type_code ! eg "rsa"
    CHARACTER ( LEN = 72 ) :: title    ! title field of file
    CHARACTER ( LEN = 8 ) :: identifier 
    TYPE ( SMT_type ) :: A
    REAL( wp ), DIMENSION( : ), ALLOCATABLE :: X, B
    TYPE( rb_read_options ) :: options ! control variables

    READ( 5, "( A80 )" ) filename

!  read header information from the Rutheford-Boeing file 

    CALL rb_peek_file( TRIM( filename ), info, m, n, nelt, nvar, nval,         &
                       matrix_type, type_code, title, identifier )

!  check that the file exists and is not faulty

    IF ( info < 0 ) THEN
      WRITE( out, "( ' input filename faulty, info = ', I0 )" ) info
      STOP
     END IF

!  print details of matrix

    WRITE( out, "( ' Matrix intentifier = ', A, ' ( ', A, ')', /               &
                   ' m = ', I0, ', n = ', I0, ', nnz = ', I0 )" )              &
      TRIM( identifier ), TRIM( type_code ), m, n, nval

    IF ( m /= n .OR. type_code /= 'rsa' ) THEN
      WRITE( out, "( ' matrix does not seem to be real, symmetric )" )
      STOP
    END IF

!  read the matrix from the Rutheford-Boeing file and translate as 
!  uper-triangluar CSC = lower triangular CSR

    options%lwr_upr_ful = 2
    CALL rb_read TRIM( filename ), m, n, A%ptr, A%col, A%val, options, info )
    CALL SMT_put( A%type, 'COORDINATE', info )

!  pick solution vector of ones

    ALLOCATE( X( n ), B( n ), info )
    X = 1.0_wp

!  generate RHS

    B = 0.0+wp
    DO i = 1, n
      DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
        j = A%col( l )
        B( i ) = B( i ) + A%val( i ) * X( j )
        IF ( i /= j ) B( j ) = B( j ) + A%val( i ) * X( i )
      END DO
    END DO

    DEALLOCATE( A%ptr, A%col, A%val, X, B )
      CALL SLS_initialize( solver, data, SLS_control, SLS_inform )

  END PROGRAM RUNRB_sls
