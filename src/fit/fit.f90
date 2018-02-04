! THIS VERSION: GALAHAD 2.4 - 29/03/2010 AT 08:00 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ F I T   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Daniel Robinson

!  History -
!   originally released GALAHAD Version 2.4 March 29th 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_FIT_double

!      -------------------------------------------------
!     |                                                 |
!     | Fit polynomials to function and derivative data |
!     |                                                 |
!      -------------------------------------------------

      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: FIT_initialize, FIT_read_specfile, FIT_hermite_interpolation,  &
                FIT_puiseux_interpolation, FIT_evaluate_polynomial,            &
                FIT_terminate

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: FIT_control_type

!   error and warning diagnostics occur on stream error 
   
        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!  if space_critical is true, every effort will be made to use as little
!   space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!  if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: FIT_inform_type

!  return status. Possible values are:
!  
!     0 Normal termination with the required fit
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status

!   - 3 the restriction
!        prob%n     >=  1
!       has been violated

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   data derived type with private components
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: FIT_data_type
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: POINTS_puiseux
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: F_puiseux, C
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: A, B
      END TYPE

   CONTAINS

!-*-*-*-*-*-   F I T _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE FIT_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for FIT. This routine should be called before
!  FIT_solve
! 
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( FIT_data_type ), INTENT( INOUT ) :: data
      TYPE ( FIT_control_type ), INTENT( INOUT ) :: control        
      TYPE ( FIT_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

      RETURN  

!  End of FIT_initialize

      END SUBROUTINE FIT_initialize

!-*-*-*-*-   F I T _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE FIT_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by FIT_initialize could (roughly) 
!  have been set as:

! BEGIN FIT SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END FIT SPECIFICATIONS

!  Dummy arguments

     TYPE ( FIT_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: space_critical = print_level + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 3 ), PARAMETER :: specname = 'FIT'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'

!  Logical key-words

     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

     spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

    IF ( PRESENT( alt_specname ) ) THEN
      CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
    ELSE
      CALL SPECFILE_read( device, specname, spec, lspec, control%error )
    END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character value

     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

     RETURN

     END SUBROUTINE FIT_read_specfile

!-*- F I T _ H E R M I T E _ I N T E R P O L A T I O N  S U B R O U T I N E -*-

     SUBROUTINE FIT_hermite_interpolation( n, POINTS, F, COEF, data, control,  &
                                           inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!    Interpolate function and derivative values to fit the polynomial 
!      y(theta) = sum_{i=1}^n COEF(i)*theta^(i-1)

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Arguments:

!   n is an INTEGER variable, which must be set by the user to the number of 
!    data values provided.  RESTRICTION: n >= 1

!   POINTS is a REAL array of length n, which must be set by the user to the 
!    value of the vector of interpolation points. The components in POINTS may 
!    be in any order subject to equal values occuring contiguously. A string 
!    of k equal values in POINTS indicates that the user wishes to fit the 
!    first (k-1) derivatives at that value of POINTS. It may be beneficial to 
!    rearrange the values within POINTS for stability reasons.  See Leja 
!    ordering for more detail in "Fast QP decomposition of Vandermonde-like 
!    matrices and polynomial least squares approximation" by Lothar Reichel, 
!    SIAM J. Matrix Anal. Appl., 12(3):552-564, 1991

!   F is a REAL array of length n, which must be set by the user to the 
!    values to be interpolated in the same order as the interpolation points

!   COEF is a REAL array of length n, which will be set on output to the 
!    coefficients of the polynomial 
!      y(theta) = sum_{i=1}^n COEF(i)*theta^(i-1)
!    with monomial basis that interpolates the data provided in POINTS and F.

!  data is a structure of type FIT_data_type which holds private internal data
!
!  control is a structure of type FIT_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to FIT_initialize. See FIT_initialize 
!   for details
!
!  inform is a structure of type FIT_inform_type that provides 
!    information on exit from FIT_solve. The component %status 
!    must be set to 1 on initial entry, and on exit has possible values:

!  This code is based on Algorithm 21.2 in the book "Accuracy and Stability
!  of Numerical Algorithms" by Nicholas J. Higham, SIAM, 1996

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: POINTS, F
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: COEF
     TYPE ( FIT_data_type ), INTENT( INOUT ) :: data
     TYPE ( FIT_control_type ), INTENT( IN ) :: control
     TYPE ( FIT_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     INTEGER :: jp1, kp1, kp1pj, nm1
     REAL ( KIND = wp ) :: c_last, temp, akp1, rkp1
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  ensure that input parameters are within allowed ranges

      IF ( n <= 0 ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 910
      END IF 

!  allocate workspace arrays

     array_name = 'fit: data%C'
     CALL SPACE_resize_array( n, data%C, inform%status,                        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     nm1 = n - 1

!  compute the confluent divided differences: Higham, Algorith 21.2, stage I

     data%C = F
     DO kp1 = 1, nm1
       rkp1 = kp1
       c_last = data%C( kp1 )
       DO jp1 = kp1 + 1, n
         IF ( POINTS( jp1 ) == POINTS( jp1 - kp1 ) ) THEN
           data%C( jp1 ) = data%C( jp1 ) / rkp1
         ELSE
           temp = data%C( jp1 )
           data%C( jp1 ) = ( data%C( jp1 ) - c_last ) /                        &
                           ( POINTS( jp1 ) - POINTS( jp1 - kp1 ) )
           c_last = temp
         END IF
       END DO
     END DO

!  recover the solution: Higham, Algorith 21.2, stage II

     COEF = data%C
     DO kp1 = n - 1, 1, - 1
       akp1 = POINTS( kp1 )
       DO kp1pj = kp1, n - 1
         COEF( kp1pj ) = COEF( kp1pj ) - akp1 * COEF( kp1pj + 1 )
       END DO
     END DO

!  print details

     IF ( control%out > 0 .AND. control%print_level > 1 ) THEN
       WRITE( control%out, 2000 ) prefix, ' Hermite points = ', POINTS
       WRITE( control%out, 2000 ) prefix, ' Hermite F      = ', F
       WRITE( control%out, 2000 ) prefix, ' coefficients   = ', COEF
     END IF

!  successful return

     inform%status = GALAHAD_ok
     RETURN

!  error returns

 910 CONTINUE 
     IF ( control%error > 0 .AND. control%print_level > 0 ) THEN
       SELECT CASE ( inform%status )
       CASE ( GALAHAD_ok )
       CASE ( GALAHAD_error_allocate )
         WRITE( control%out, 2010 ) prefix, 'allocation error'
         WRITE( control%out, 2020 ) prefix, inform%alloc_status,               &
                                            inform%bad_alloc
       CASE ( GALAHAD_error_deallocate )
         WRITE( control%out, 2010 ) prefix, 'de-allocation error'
         WRITE( control%out, 2020 ) prefix, inform%alloc_status,               &
                                            inform%bad_alloc
       CASE ( GALAHAD_error_restrictions )
         WRITE( control%out, 2010 ) prefix, 'input restriction violated'
       CASE DEFAULT
         WRITE( control%out, 2010 ) prefix, 'undefined error'
       END SELECT
     END IF
     RETURN  

!  non-executable statements

2000 FORMAT( A, A18, 5ES12.4, :, ( /, 12X, 6ES12.4 ) )
2010 FORMAT( /, A, ' FIT_hermite_interpolation: ', A )
2020 FORMAT( /, A, ' allocation error status ', I0, ' for ', A )

!  End of FIT_hermite_interpolation

     END SUBROUTINE FIT_hermite_interpolation

!-*- F I T _ P U I S E U X _ I N T E R P O L A T I O N  S U B R O U T I N E -*-

     SUBROUTINE FIT_puiseux_interpolation( n, POINTS, F, COEF, data, control,  &
                                           inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!    Interpolate function and derivative values to fit the polynomial 
!      x(mu) = sum_{i=1}^n COEF(i)*sqrt(mu)^(i-1)

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Arguments:

!   n is an INTEGER variable, which must be set by the user to the number of 
!    data values provided.  RESTRICTION: n >= 1

!   POINTS is a REAL array of length n, which must be set by the user to the 
!    value of the vector of interpolation points. The components in POINTS may 
!    be in any order subject to equal values occuring contiguously. A string 
!    of k equal values in POINTS indicates that the user wishes to fit the 
!    first (k-1) derivatives at that value of POINTS. It may be beneficial to 
!    rearrange the values within POINTS for stability reasons.  See Leja 
!    ordering for more detail in "Fast QP decomposition of Vandermonde-like 
!    matrices and polynomial least squares approximation" by Lothar Reichel, 
!    SIAM J. Matrix Anal. Appl., 12(3):552-564, 1991. RESTRICTION: POINTS > 0

!   F is a REAL array of length n, which must be set by the user to the 
!    values to be interpolated in the same order as the interpolation points

!   COEF is a REAL array of length n, which will be set on output to the 
!    coefficients of the polynomial 
!      x(mu) = sum_{i=1}^n COEF(i)*sqrt(mu)^(i-1)
!    with monomial basis that interpolates the data provided in THETA and F.

!  data is a structure of type FIT_data_type which holds private internal data
!
!  control is a structure of type FIT_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to FIT_initialize. See FIT_initialize 
!   for details
!
!  inform is a structure of type FIT_inform_type that provides 
!    information on exit from FIT_solve. The component %status 
!    must be set to 1 on initial entry, and on exit has possible values:

!  This code is based on Algorithm 21.2 in the book "Accuracy and Stability
!  of Numerical Algorithms" by Nicholas J. Higham, SIAM, 1996

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: POINTS, F
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: COEF
     TYPE ( FIT_data_type ), INTENT( INOUT ) :: data
     TYPE ( FIT_control_type ), INTENT( IN ) :: control
     TYPE ( FIT_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     INTEGER :: i, j, k, l, n_a, n_b, order
     REAL ( KIND = wp ) :: der, power
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  ensure that input parameters are within allowed ranges

      IF ( n <= 0 ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 910
      END IF 

!  allocate workspace arrays

     array_name = 'fit: data%POINTS_puiseux'
     CALL SPACE_resize_array( n, data%POINTS_puiseux, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'fit: data%F_piiseux'
     CALL SPACE_resize_array( n, data%F_puiseux, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  record theta = sqrt( mu ) the square roots of the interpolation points, 
!  making sure that the points are positive

     DO i = 1, n
       IF ( POINTS( i ) > zero ) THEN
         data%POINTS_puiseux( i ) = SQRT( POINTS( i ) )
       ELSE
         inform%status = GALAHAD_error_restrictions
         GO TO 910
       END IF
     END DO

!   use Hermite interpolation to fit y(theta) = x(theta^2), where 
!   y(theta) = sum_i=0^n-1 coef * theta^i. We need to find the derivative 
!   values of y(theta) wrt theta, and use the relationships

!   y^(2k)(theta) = a^(k)_0 x^(k) + ... + a^(k)_j theta^2j x^(k+j) + ... 
!                   + a^(k)_k theta^2k x^(2k)

!   y^(2k+1)(theta) = b^(k)_0 theta x^(k+1) + ... 
!                   + b^(k)_j theta^2j+1 x^(k+j+1) + ... 
!                   + b^(k)_k theta^2k+1 x^(2k+1)

!   and identities

!   b^(k)_0 = 2 (a^(k)_0 + a^(k)_1 )
!   b^(k)_j-1 = 2 (a^(k)_j-1 + j * a^(k)_j )
!   b^(k)_k = 2 a^(k)_k )

!   and 

!   a^(k+1)_0 = b^(k)_0
!   a^(k+1)_j = 2 b^(k)_j-1 + ( 2 j + 1 ) * b^(k)_j )
!   a^(k+1)_k+1 = 2 b^(k)_k )

!  Compute the function and derivative values for the fit. First, determine 
!  the highest order of derivative used

     order = 0
     k = 0
     DO i = 1, n - 1
       IF ( POINTS( i + 1 ) == POINTS( i ) ) THEN
         k = k + 1
       ELSE
         order = MAX( order, k )
         k = 0
       END IF
     END DO
     order = MAX( order, k )

!  if derivatives are present, compute the coefficients a^(k)_j and b^(k)_j.
!  See how much space is required

     IF ( order > 0 ) THEN
       IF ( MOD( order, 2 ) == 0 ) THEN
         n_a = order / 2
         n_b = n_a
       ELSE
         n_a = order / 2
         n_b = n_a + 1
       END IF

!  allocate space for the coefficients

       array_name = 'fit: data%A'
       CALL SPACE_resize_array( 0, n_a, n_a, data%A, inform%status,            &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'fit: data%B'
       CALL SPACE_resize_array( 0, n_b, n_b, data%B, inform%status,            &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  now find the coefficients themselves

       data%B( 0, 1 ) = two
       DO i = 2, order
         IF ( MOD( i, 2 ) == 0 ) THEN
           k = i / 2
           data%A( 0, k ) = data%B( 0, k )
           DO j = 1, k - 1
             data%A( j, k ) = two * data%B( j - 1 , k ) +                      &
                            ( two * j + one ) * data%B( j , k )
           END DO
           data%A( k, k ) = two * data%B( k - 1, k )
         ELSE
           k = i / 2 + 1
           DO j = 0, k - 2
             data%B( j, k ) = two * ( data%A( j, k - 1 ) +                     &
                              ( j + one ) * data%A( j + 1, k - 1 ) ) 
           END DO
           data%B( k - 1, k ) = two * data%A( k - 1, k - 1 )
         END IF
       END DO

!  finally, compute the derivatives of y

       i = 0
       data%F_puiseux( 1 ) = F( 1 )
       DO l = 2, n
         IF ( POINTS( l - 1 ) == POINTS( l ) ) THEN
           i = i + 1
           IF ( MOD( i, 2 ) == 0 ) THEN
             k = i / 2
             power = one
             der = data%A( 0, k ) * F( l - k )
             DO j = 1, k
               power = power * POINTS( l )
               der = der + data%A( j, k ) * power * F( l + j - k )
             END DO
             data%F_puiseux( l ) = der
           ELSE
             k = i / 2 + 1
             power = data%POINTS_puiseux( l )
             der = data%B( 0, k ) * power * F( l - k + 1 )
             DO j = 1, k - 1
               power = power * POINTS( l )
               der = der + data%B( j, k ) * power * F( l + j - k + 1 )
             END DO
             data%F_puiseux( l ) = der
           END IF
         ELSE
           i = 0
           data%F_puiseux( l ) = F( l )
         END IF
       END DO
     ELSE
       data%F_puiseux( : n ) = F( : n )
     END IF

!  print details

     IF ( control%out > 0 .AND. control%print_level > 1 ) THEN
       WRITE( control%out, 2000 ) prefix, ' Puiseux points = ', POINTS
       WRITE( control%out, 2000 ) prefix, ' Puiseux F      = ', F
     END IF

!  use Hermite interpolation to calculate the coefficients

     CALL FIT_hermite_interpolation( n, data%POINTS_puiseux, data%F_puiseux,   &
                                     COEF, data, control, inform )

!  successful return

     IF ( inform%status == GALAHAD_ok ) RETURN

!  error returns

 910 CONTINUE 
     IF ( control%error > 0 .AND. control%print_level > 0 ) THEN
       SELECT CASE ( inform%status )
       CASE ( GALAHAD_ok )
       CASE ( GALAHAD_error_allocate )
         WRITE( control%out, 2010 ) prefix, 'allocation error'
         WRITE( control%out, 2020 ) prefix, inform%alloc_status,               &
                                            inform%bad_alloc
       CASE ( GALAHAD_error_deallocate )
         WRITE( control%out, 2010 ) prefix, 'de-allocation error'
         WRITE( control%out, 2020 ) prefix, inform%alloc_status,               &
                                            inform%bad_alloc
       CASE ( GALAHAD_error_restrictions )
         WRITE( control%out, 2010 ) prefix, 'input restriction violated'
       CASE DEFAULT
         WRITE( control%out, 2010 ) prefix, 'undefined error'
       END SELECT
     END IF
     RETURN  

!  non-executable statements

2000 FORMAT( A, A18, 5ES12.4, :, ( /, 12X, 6ES12.4 ) )
2010 FORMAT( /, A, ' FIT_puiseux_interpolation: ', A )
2020 FORMAT( /, A, ' allocation error status ', I0, ' for ', A )

!  End of FIT_puiseux_interpolation

     END SUBROUTINE FIT_puiseux_interpolation

!-*-*-*- F I T _ E V A L U A T E _ P O L Y N O M I A L  F U M C T I O N -*-*-*-

     FUNCTION FIT_evaluate_polynomial( n, COEF, theta )
     REAL ( KIND = wp ) :: FIT_evaluate_polynomial

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!    Evaluate the polynomial
!      p(theta) = sum_{i=1}^n COEF(i)*theta^(i-1)
!    using Horner's method

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Arguments:

!   n is an INTEGER variable, which must be set by the user to the number of 
!    data values provided.

!   COEF is a REAL array of length n, which give the coefficients of the 
!    polynomial 
!      p(theta) = sum_{i=1}^n COEF(i)*theta^(i-1)

!   theta is a REAL variable that specifies the value of the argument theta
!    for which the value p(theta) is required

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ) :: theta
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: COEF

!  Local variables

     INTEGER :: i
     REAL ( KIND = wp ) :: polynomial

!  evaluate the polynomial

     polynomial = COEF( n )
     DO i = n - 1, 1, - 1
       polynomial = COEF( i ) + theta * polynomial
     END DO

     FIT_evaluate_polynomial = polynomial
     RETURN

!  End of FIT_evaluate_polynomial

     END FUNCTION FIT_evaluate_polynomial

!-*-*-*-*-*-*-   F I T _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

     SUBROUTINE FIT_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................................
!      .                                                            .
!      .  Deallocate internal arrays at the end of the computation  .
!      .                                                            .
!      ..............................................................

!  Arguments:
!  =========
!
!   data    see preface
!   control see preface
!   inform  see preface

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( FIT_data_type ), INTENT( INOUT ) :: data
     TYPE ( FIT_control_type ), INTENT( IN ) :: control        
     TYPE ( FIT_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate all remaining allocated arrays

     array_name = 'fit: data%A'
     CALL SPACE_dealloc_array( data%A,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'fit: data%B'
     CALL SPACE_dealloc_array( data%B,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'fit: data%C'
     CALL SPACE_dealloc_array( data%C,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'fit: data%POINTS_puiseux'
     CALL SPACE_dealloc_array( data%POINTS_puiseux,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'fit: data%F_puiseux'
     CALL SPACE_dealloc_array( data%F_puiseux,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine FIT_terminate

     END SUBROUTINE FIT_terminate

!  End of module FIT

   END MODULE GALAHAD_FIT_double
