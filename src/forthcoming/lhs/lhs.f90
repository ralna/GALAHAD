! THIS VERSION: GALAHAD 2.8 - 29/06/2016 AT 13:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ L H S   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  This code is distributed under the GNU LGPL license.
!  Principal author: John Burkardt, 2003-2012
!  Modularised and "polished" in the GALAHAD way by Nick Gould, 2016
!
!  History -
!   see http://people.sc.fsu.edu/~jburkardt/f_src/ihs/ihs.html
!   originally released GALAHAD Version 2.8. June 29th 2016

  MODULE GALAHAD_LHS_double

!     --------------------------------------------------
!    |                                                  |
!    | LHS, compute an array of Latin Hypercube Samples |
!    |                                                  |
!     --------------------------------------------------

     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: LHS_initialize, LHS_read_specfile, LHS_ihs, LHS_get_seed,       &
               LHS_terminate

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: sp = KIND( 1.0 )

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER, PARAMETER :: i4_huge = HUGE( 1 )
     REAL ( KIND = wp ), PARAMETER :: r8_huge = HUGE( 1.0_wp )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: LHS_control_type

!   error and warning diagnostics occur on stream error

       INTEGER :: error = 6

!   general output occurs on stream out

       INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, >= 2 gives increasingly verbose (debugging)
!   output

       INTEGER :: print_level = 0

!   the duplication factor. This must be at least 1, a value of 5 is reasonable
!
       INTEGER :: duplication = 5

!   if %space_critical true, every effort will be made to use as little
!    space as possible. This may result in longer computation time

       LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

     END TYPE LHS_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: LHS_inform_type

!  return status. 0 = success, /= 0 failure

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

     END TYPE LHS_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: LHS_data_type
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: LIST
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: POINT
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: AVAIL
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VEC
     END TYPE LHS_data_type

   CONTAINS

!-*-*-  G A L A H A D -  L H S _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE LHS_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for LHS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LHS_data_type ), INTENT( INOUT ) :: data
     TYPE ( LHS_control_type ), INTENT( OUT ) :: control
     TYPE ( LHS_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

     RETURN

!  End of subroutine LHS_initialize

     END SUBROUTINE LHS_initialize

!-*-*-*-*-   L H S _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE LHS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by LHS_initialize could (roughly)
!  have been set as:

! BEGIN LHS SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  print-level                                     0
!  duplication-factor                              5
!  space-critical                                  no
!  deallocate-error-fatal                          no
! END LHS SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LHS_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: duplication = print_level + 1
     INTEGER, PARAMETER :: space_critical = duplication + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'LHS '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( duplication )%keyword = 'duplication-factor'

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
     CALL SPECFILE_assign_value( spec( duplication ),                          &
                                 control%duplication,                          &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

     END SUBROUTINE LHS_read_specfile

!-*-*-*-*-*-  G A L A H A D -  L H S _ I H S   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE LHS_ihs( n_dimen, n_points, seed, X, control, inform, data )

!*****************************************************************************
!
!! IHS implements the improved distributed hyper-cube sampling algorithm.
!
!  Discussion:
!
!    n_points points in an n_dimen dimensional Latin hyper-cube are
!    to be selected. Each of the coordinate dimensions is discretized
!    to the values 1 through n.  The points are to be chosen in such
!    a way that no two points have any coordinate value in common.
!    This is a standard Latin hypercube requirement, and there are
!    many solutions.
!
!    This algorithm differs in that it tries to pick a solution
!    which has the property that the points are "spread out"
!    as evenly as possible.  It does this by determining an optimal
!    even spacing, and using the DUPLICATION factor to allow it
!    to choose the best of the various options available to it.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license
!
!  Modified:
!
!    02 April 2003
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Brian Beachkofski, Ramana Grandhi,
!    Improved Distributed Hypercube Sampling,
!    American Institute of Aeronautics and Astronautics Paper 2002-1274
!
!  Parameters:
!
!    Input, integer, n_dimen, the spatial dimension
!
!    Input, integer  n_points, the number of points to be generated
!
!    Input/output, integer seed, a seed for the random number generator
!
!    Output, integer X(n_dimen,n), the points
!
!    Output, integer status, successful = 0, error /=0

!    control, inform, data - see LHS_initialize

!*****************************************************************************

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n_dimen, n_points
      INTEGER, INTENT( INOUT ), DIMENSION( n_dimen, n_points ) :: X
      TYPE ( LHS_control_type ), INTENT( IN ) :: control
      TYPE ( LHS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( LHS_data_type ), INTENT( INOUT ) :: data

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: best, count, i, j, k, point_index, seed, duplication
      REAL ( KIND = wp ) :: dist, min_all, min_can, opt
      CHARACTER ( LEN = 80 ) :: array_name

!  set global values

      duplication = MAX( control%duplication, 1 )
      opt = REAL( n_points, KIND = wp ) /                                      &
      ( REAL( n_points, KIND = wp ) ) ** ( 1.0_wp / REAL( n_dimen, KIND = wp ) )

!  allocate sufficient space for the problem

     array_name = 'lhs: data%POINT'
     CALL SPACE_resize_array( n_dimen, duplication * n_points, data%POINT,     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'lhs: data%LIST'
     CALL SPACE_resize_array( duplication * n_points, data%LIST,               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'lhs: data%AVAIL'
     CALL SPACE_resize_array( n_dimen, n_points, data%AVAIL,                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'lhs: data%VEC'
     CALL SPACE_resize_array( n_dimen, data%VEC,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) RETURN

!  pick the first point

      CALL i4vec_uniform_ab( n_dimen, 1, n_points, seed,                       &
                             X( 1 : n_dimen, n_points ), inform%status )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  initialize AVAIL, and set an entry in a random row of each column of
!  AVAIL to n

      DO j = 1, n_points
        data%AVAIL( 1 : n_dimen, j ) = j
      END DO

      DO i = 1, n_dimen
        data%AVAIL( i, X( i, n_points ) ) = n_points
      END DO

!  main loop: assign a value to X(1:n_dimen,count) for count = n-1 down to 2

      DO count = n_points - 1, 2, - 1

!  generate valid points

        DO i = 1, n_dimen
          DO k = 1, duplication
            data%LIST( count * ( k - 1 ) + 1 : k * count )                     &
              = data%AVAIL( i, 1 : count )
          END DO

          DO k = count * duplication, 1, - 1
            CALL i4_uniform_ab( 1, k, seed, point_index, inform%status )
            IF ( inform%status /= GALAHAD_ok ) RETURN
            data%POINT( i, k ) = data%LIST( point_index )
            data%LIST( point_index ) = data%LIST( k )
          END DO
        END DO

!  for each candidate, determine the distance to all the points that have
!  already been selected, and save the minimum value

        min_all = r8_huge
        best = 0

        DO k = 1, duplication * count
          min_can = r8_huge
          DO j = count + 1, n_points
            data%VEC( 1 : n_dimen )                                            &
              = REAL( data%POINT( 1 : n_dimen, k ) - X( 1 : n_dimen, j ),      &
                      KIND = wp )
            dist = TWO_NORM( data%VEC( 1 : n_dimen ) )
            min_can = min ( min_can, dist )
          END DO

          IF ( ABS( min_can - opt ) < min_all ) THEN
            min_all = ABS( min_can - opt )
            best = k
          END IF
        END DO

        X( 1 : n_dimen,count ) = data%POINT( 1 : n_dimen,best )

!  having chosen X(*,count), update AVAIL

        DO i = 1, n_dimen
          DO j = 1, n_points
            IF ( data%AVAIL( i, j ) == X( i, count ) ) THEN
              data%AVAIL( i, j ) = data%AVAIL( i, count )
            END IF
          END DO
        END DO

      END DO

!  for the last point, there is only one choice

      X( 1 : n_dimen, 1 ) = data%AVAIL( 1 : n_dimen, 1 )

      inform%status = GALAHAD_ok
      RETURN

!  internal procedures

      CONTAINS

!-*-  G A L A H A D -  i 4 v e c _ u n i f o r m _ a b  S U B R O U T I N E  -*-

        SUBROUTINE i4vec_uniform_ab( n, a, b, seed, X, status )

!*****************************************************************************
!
!! I4VEC_UNIFORM_AB returns a scaled pseudo-random i4vec
!
!  Discussion:
!
!    An i4vec is a vector of integers
!
!    The pseudorandom numbers should be scaled to be uniformly distributed
!    between a and b
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license
!
!  Modified:
!
!    27 November 2006
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer n, the dimension of the vector
!
!    Input, integer a, b, the limits of the interval
!
!    Input/output, integer seed, the "seed" value, which
!    should NOT be 0.  On output, seed has been updated
!
!    Output, integer X(n), a vector of numbers between a and b
!
!    Output, integer status, successful = 0, error /=0

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

        INTEGER, INTENT( IN ) :: n, a, b
        INTEGER, INTENT( INOUT) :: seed
        INTEGER, INTENT( OUT ) :: status
        INTEGER, INTENT( OUT), DIMENSION( n ) :: X

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

        INTEGER :: i
        REAL ( kind = sp ) :: r

        IF ( seed == 0 ) THEN
!         write ( *, '(a)' ) ' '
!         write ( *, '(a)' ) 'I4VEC_UNIFORM_AB - Fatal error!'
!         write ( *, '(a)' ) '  Input value of SEED = 0.'
          status = GALAHAD_error_restrictions ; RETURN
        END IF

        DO i = 1, n
          k = seed / 127773
          seed = 16807 * ( seed - k * 127773 ) - k * 2836
          IF ( seed < 0 ) seed = seed + i4_huge
          r = REAL( seed, KIND = sp ) * 4.656612875E-10

!  scale r to lie between a-0.5 and b+0.5

          r = ( 1.0_sp - r ) * ( REAL( MIN( a, b ), kind = sp ) - 0.5_sp ) +   &
                         r   * ( REAL( MAX( a, b ), kind = sp ) + 0.5_sp )

!  use rounding to convert r to an integer between a and b

          X( i ) = MIN( MAX( NINT( r, KIND = sp ), MIN( a, b ) ), MAX( a, b ) )
        END DO

        status = GALAHAD_ok
        RETURN

!  end of subroutine i4vec_uniform_ab

        END SUBROUTINE i4vec_uniform_ab

!-*-*-  G A L A H A D -  i 4 _ u n i f o r m _ a b  S U B R O U T I N E  -*-*-

        SUBROUTINE i4_uniform_ab( a, b, seed, x, status )

!*****************************************************************************
!
!! I4_UNIFORM_AB returns a scaled pseudo-random i4 between a and b
!
!  Discussion:
!
!    An i4 is an integer value
!
!    The pseudorandom number will be scaled to be uniformly distributed
!    between a and b
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license
!
!  Modified:
!
!    02 October 2012
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Paul Bratley, Bennett Fox, Linus Schrage,
!    A Guide to Simulation,
!    Second Edition,
!    Springer, 1987,
!    ISBN: 0387964673,
!    LC: QA76.9.C65.B73.
!
!    Bennett Fox,
!    Algorithm 647:
!    Implementation and Relative Efficiency of Quasirandom
!    Sequence Generators,
!    ACM Transactions on Mathematical Software,
!    Volume 12, Number 4, December 1986, pages 362-376.
!
!    Pierre L'Ecuyer,
!    Random Number Generation,
!    in Handbook of Simulation,
!    edited by Jerry Banks,
!    Wiley, 1998,
!    ISBN: 0471134031,
!    LC: T57.62.H37.
!
!    Peter Lewis, Allen Goodman, James Miller,
!    A Pseudo-Random Number Generator for the System/360,
!    IBM Systems Journal,
!    Volume 8, Number 2, 1969, pages 136-143.
!
!  Parameters:
!
!    Input, integer a, b, the limits of the interval
!
!    Input/output, integer seed, the "seed" value, which
!    should NOT be 0.  On output, seed has been updated
!
!    Output, integer x, a number between A and B.
!
!    Output, integer status, successful = 0, error /=0

!*****************************************************************************

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

        INTEGER, INTENT( IN ) :: a, b
        INTEGER, INTENT( INOUT ) :: seed
        INTEGER, INTENT( OUT ) :: x, status

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

        INTEGER :: k
        REAL ( KIND = sp ) r

        IF ( seed == 0 ) THEN
!         write ( *, '(a)' ) ' '
!         write ( *, '(a)' ) 'I4_UNIFORM_AB - Fatal error!'
!         write ( *, '(a)' ) '  Input value of SEED = 0.'
          status = GALAHAD_error_restrictions ; RETURN
        END IF

        k = seed / 127773
        seed = 16807 * ( seed - k * 127773 ) - k * 2836
        IF ( seed < 0 ) seed = seed + i4_huge
        r = REAL( SEED, KIND = sp ) * 4.656612875E-10
!
!  scale R to lie between a-0.5 and b+0.5
!
        r = ( 1.0_sp - r ) * ( REAL( MIN( a, b ), KIND = sp ) - 0.5_sp ) +     &
                       r   * ( REAL( MAX( a, b ), KIND = sp ) + 0.5_sp )
!
!  use rounding to convert r to an integer between a and b
!
        x = MIN( MAX( NINT( r ), MIN( a, b ) ), MAX( a, b ) )

        status = GALAHAD_ok
        RETURN

!   end of SUBROUTINE i4_uniform_ab

        END SUBROUTINE i4_uniform_ab

!  end of subroutine LHS_ihs

      END SUBROUTINE LHS_ihs

!-*-*-*-*-*-  G A L A H A D -  g e t _ s e e d  S U B R O U T I N E  -*-*-*-*-*-

      SUBROUTINE LHS_get_seed( seed )

!*****************************************************************************
!
!! GET_SEED returns a seed for the random number generator.
!
!  Discussion:
!
!    The seed depends on the current time, and ought to be (slightly)
!    different every millisecond.  Once the seed is obtained, a random
!    number generator should be called a few times to further process
!    the seed.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    02 August 2004
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Output, integer SEED, a pseudorandom seed value.

!*****************************************************************************

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( OUT ) :: seed

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      REAL ( KIND = wp ) :: temp
      CHARACTER ( LEN = 5 ) zone
      CHARACTER ( LEN = 8 ) :: today
      CHARACTER ( LEN = 10 ) :: time
      INTEGER, DIMENSION( 8 ) :: values

      CALL DATE_AND_TIME( today, time, zone, values )

      temp = 0.0_wp
      temp = temp + REAL( values( 2 ) - 1, kind = wp ) /  11.0_wp
      temp = temp + REAL( values( 3 ) - 1, kind = wp ) /  30.0_wp
      temp = temp + REAL( values( 5 ),     kind = wp ) /  23.0_wp
      temp = temp + REAL( values( 6 ),     kind = wp ) /  59.0_wp
      temp = temp + REAL( values( 7 ),     kind = wp ) /  59.0_wp
      temp = temp + REAL( values( 8 ),     kind = wp ) / 999.0_wp
      temp = temp                                      /   6.0_wp

      DO while ( temp <= 0.0_wp )
        temp = temp + 1.0_wp
      END DO

      DO while ( 1.0_wp < temp )
        temp = temp - 1.0_wp
      END DO

      seed = INT( REAL( i4_huge, KIND = wp ) * temp )

!  never use a seed of 0 or maximum integer

      IF ( seed == 0 ) seed = 1
      IF ( seed == i4_huge ) seed = seed - 1

      RETURN

!  end of subroutine LHS_get_seed

      END SUBROUTINE LHS_get_seed

!-*-*-  G A L A H A D -  L H S _ t e r m i n a t e  S U B R O U T I N E -*-*-

      SUBROUTINE LHS_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LHS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LHS_control_type ), INTENT( IN ) :: control
      TYPE ( LHS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

      array_name = 'lhs: data%LIST'
      CALL SPACE_dealloc_array( data%LIST, inform%status,                      &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lhs: data%POINT'
      CALL SPACE_dealloc_array( data%POINT, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lhs: data%AVAIL'
      CALL SPACE_dealloc_array( data%AVAIL, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lhs: data%VEC'
      CALL SPACE_dealloc_array( data%VEC, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  end of subroutine LHS_terminate

      END SUBROUTINE LHS_terminate

!  end of module GALAHAD_LHS

  END MODULE GALAHAD_LHS_double
