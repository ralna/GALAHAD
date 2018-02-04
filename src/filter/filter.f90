! THIS VERSION: GALAHAD 2.6 - 28/11/2014 AT 12:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D _ F I L T E R   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.0. May 25th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_FILTER_double

     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPECFILE_double

!      ---------------------------------
!     |  NLP filter manipulation tools  |
!      ---------------------------------

     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: FILTER_read_specfile, filter_initialize_filter,                 &
               FILTER_acceptable, FILTER_update_filter, FILTER_resize_filter,  &
               FILTER_dealloc_filter, FILTER_terminate

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
     REAL ( KIND = wp ), PARAMETER :: minus_infinity = - infinity / two 

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - -
!   filter derived type 
!  - - - - - - - - - - -

     TYPE, PUBLIC :: FILTER_filter_type
       REAL ( KIND = wp ) :: o, v
     END TYPE FILTER_filter_type

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: FILTER_control_type

!   error and warning diagnostics occur on stream error 
   
       INTEGER :: error = 6

!   general output occurs on stream out

       INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER :: print_level = 0

!   initial estimate for the maximum filter size

       INTEGER :: initial_filter_size = 100

!   if space_critical is true, every effort will be made to use as little
!    space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!   if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'

      CHARACTER ( LEN = 30 ) :: prefix = '""                            '

     END TYPE FILTER_control_type

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: FILTER_inform_type

!  return status. See TRU_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the number of filter entries

       INTEGER :: filter_size = 0

     END TYPE FILTER_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: FILTER_data_type
       INTEGER :: n_filter, max_filter
       LOGICAL :: first_filter_in_use
       TYPE ( FILTER_filter_type ), POINTER, DIMENSION( : ) :: filter => NULL()
       TYPE ( FILTER_filter_type ), POINTER, DIMENSION( : ) :: filter1 => NULL()
       TYPE ( FILTER_filter_type ), POINTER, DIMENSION( : ) :: filter2 => NULL()
     END TYPE FILTER_data_type

  CONTAINS

!-*-*-*-   F I L T E R _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-

     SUBROUTINE FILTER_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by FILTER_initialize could (roughly) 
!  have been set as:

! BEGIN FILTER SPECIFICATIONS (DEFAULT)
!  error-printout-device                          6
!  printout-device                                6
!  print-level                                    0
!  initial-filter-size                            100
!  space-critical                                 NO
!  deallocate-error-fatal                         NO
!  output-line-prefix                             ''
! END FILTER SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FILTER_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: alive_unit = out + 1
     INTEGER, PARAMETER :: print_level = alive_unit + 1
     INTEGER, PARAMETER :: initial_filter_size = print_level + 1
     INTEGER, PARAMETER :: space_critical = initial_filter_size + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 6 ), PARAMETER :: specname = 'FILTER'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level' 
     spec( initial_filter_size )%keyword = 'initial-filter-size'

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
     CALL SPECFILE_assign_value( spec( initial_filter_size ),                  &
                                 control%initial_filter_size,                  &
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

     RETURN

     END SUBROUTINE FILTER_read_specfile

!  G A L A H A D -  F I L T E R _ u p d a t e _ f i l t e r  S U B R O U T I N E

     SUBROUTINE FILTER_initialize_filter( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*

!   initialize the filter

!  *-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FILTER_data_type ), INTENT( OUT ) :: data
     TYPE ( FILTER_control_type ), INTENT( IN ) :: control
     TYPE ( FILTER_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

     data%max_filter = control%initial_filter_size

     array_name = 'filter: data%filter1'
     CALL FILTER_resize_filter( data%max_filter, data%filter1, inform%status,  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) RETURN

     data%n_filter = 0 ; inform%filter_size = data%n_filter
     data%first_filter_in_use = .TRUE.
     data%filter => data%filter1

     RETURN

!  end of subroutine FILTER_initialize_filter

     END SUBROUTINE FILTER_initialize_filter

!-*- G A L A H A D -  F I L T E R _ a c c e p t a b l e  S U B R O U T I N E -*-

     SUBROUTINE FILTER_acceptable( o_new, v_new, data, control, acceptable,    &
                                   o, v )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!   check if (o_new,v_new) is acceptable for the current filter and 
!   optionally (o,v)

!   a new point (o_new,v_new) will be acceptable to a filter entry (o_i,v_i) if
!      v_new <= v_i
!   or
!      o_new <= o_i

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = wp ), INTENT( IN ) :: o_new, v_new
     LOGICAL, INTENT( OUT ) :: acceptable
     TYPE ( FILTER_data_type ), INTENT( IN ) :: data
     TYPE ( FILTER_control_type ), INTENT( IN ) :: control
     REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: o, v

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i

!  check acceptability for current iterate

      IF ( PRESENT( o ) .AND. PRESENT( v ) ) THEN
        IF ( control%out > 0 .AND. control%print_level > 0 )                   &
          WRITE( control%out, "( ' o_new,o ', 2ES22.14, /,                     &
         &                       ' v_new,v ', 2ES22.14 )" ) o_new, o, v_new, v
        IF ( o_new > o .AND. v_new > v ) THEN
          acceptable = .FALSE.
          RETURN
        END IF
      END IF

!  check acceptability for the filter

      DO i = 1, data%n_filter
        IF ( control%out > 0 .AND. control%print_level > 0 )                   &
          WRITE( control%out, "( ' o_new,o_i', 2ES22.14, /, ' v_new,v_i',      &
         &  2ES22.14 )" )  o_new, data%filter( i )%o, v_new, data%filter( i )%v
        IF ( o_new > data%filter( i )%o .AND.                                  &
             v_new > data%filter( i )%v ) THEN
          acceptable = .FALSE.
          RETURN
        END IF
      END DO
      acceptable = .TRUE.

      RETURN

!  End of subroutine FILTER_acceptable

      END SUBROUTINE FILTER_acceptable

!  G A L A H A D -  F I L T E R _ u p d a t e _ f i l t e r  S U B R O U T I N E

     SUBROUTINE FILTER_update_filter( o_new, v_new, data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!   add an entry (o_new,v_new) to the filter

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = wp ), INTENT( IN ) :: o_new, v_new
     TYPE ( FILTER_data_type ), INTENT( INOUT ) :: data
     TYPE ( FILTER_control_type ), INTENT( IN ) :: control
     TYPE ( FILTER_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j
     CHARACTER ( LEN = 80 ) :: array_name

!  remove any filter enteries that will be dominated by the incoming entry

     DO i = 1, data%n_filter
       IF ( ( o_new < data%filter( i )%o .AND.                                 &
              v_new <= data%filter( i )%v ) .OR.                               &
            ( v_new < data%filter( i )%v .AND.                                 &
              o_new <= data%filter( i )%o ) ) THEN
         DO j = data%n_filter, i, - 1
           data%n_filter = data%n_filter - 1
           IF ( ( o_new < data%filter( j )%o .AND.                             &
                  v_new <= data%filter( j )%v ) .OR.                           &
              ( v_new < data%filter( j )%v .AND.                               &
                o_new <= data%filter( j )%o ) ) THEN
           ELSE
             data%filter( i ) = data%filter( j )
             EXIT
           END IF
         END DO
       END IF
       IF ( i >= data%n_filter ) EXIT
     END DO

!  if necessary, increase the space required to hold the filter
     
     IF ( data%n_filter + 1 > data%max_filter ) THEN
       data%max_filter = 2 * data%max_filter
       IF (  data%first_filter_in_use ) THEN

         array_name = 'filter: data%filter2'
         CALL FILTER_resize_filter( data%max_filter,                           &
                data%filter2, inform%status,                                   &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) RETURN

         data%filter2( : data%n_filter ) = data%filter1( : data%n_filter ) 
         data%filter => data%filter2
         data%first_filter_in_use = .FALSE.

         array_name = 'filter: data%filter1'
         CALL FILTER_dealloc_filter( data%filter1, inform%status,              &
                inform%alloc_status, array_name = array_name,                  &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) RETURN
       ELSE
         array_name = 'filter: data%filter1'
         CALL FILTER_resize_filter( data%max_filter,                           &
                data%filter1, inform%status,                                   &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) RETURN

         data%filter1( : data%n_filter ) = data%filter2( : data%n_filter ) 
         data%filter => data%filter1
         data%first_filter_in_use = .TRUE.

         array_name = 'filter: data%filter2'
         CALL FILTER_dealloc_filter( data%filter2, inform%status,              &
                inform%alloc_status, array_name = array_name,                  &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) RETURN
       END IF
     END IF

!  add the new filter entry

     data%n_filter = data%n_filter + 1 ; inform%filter_size = data%n_filter
     data%filter( data%n_filter )%o = o_new
     data%filter( data%n_filter )%v = v_new

     RETURN

!  End of subroutine FILTER_update_filter

     END SUBROUTINE FILTER_update_filter

!  -*-*-  F I L T E R _ R E S I Z E _ F I L T E R   S U B R O U T I N E   -*-*-

     SUBROUTINE FILTER_resize_filter( len, point, status, alloc_status,        &
        deallocate_error_fatal, array_name, exact_size, bad_alloc, out )

!  ensure that the filter_type pointer array "point" is of lenth at least len.

!  if exact_size is present and true, point is reallocated to be of size len. 
!  Otherwise point is only reallocated if its length is currently smaller 
!  than len

!  Dummy arguments

     INTEGER, INTENT( IN ) :: len
     INTEGER, INTENT( OUT ) :: status, alloc_status
     TYPE ( FILTER_filter_type ), POINTER, DIMENSION( : ) :: point
     INTEGER, OPTIONAL :: out
     LOGICAL, OPTIONAL :: deallocate_error_fatal, exact_size
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

!  Local variable

     LOGICAL :: reallocate

!  check to see if a reallocation (or initial allocation) is needed

     status = GALAHAD_ok ; alloc_status = 0 ; reallocate = .TRUE.
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point ) ) THEN
       IF ( PRESENT( exact_size ) ) THEN
         IF ( exact_size ) THEN
           IF ( SIZE( point ) /= len ) THEN 
             CALL FILTER_dealloc_filter( point, status, alloc_status,          &
                                         array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         ELSE
           IF ( SIZE( point ) < len ) THEN 
             CALL FILTER_dealloc_filter( point, status, alloc_status,          &
                                         array_name, bad_alloc, out )
           ELSE ; reallocate = .FALSE.
           END IF
         END IF
       ELSE
         IF ( SIZE( point ) < len ) THEN 
           CALL FILTER_dealloc_filter( point, status, alloc_status,            &
                                       array_name, bad_alloc, out )
           DEALLOCATE( point, STAT = alloc_status )
         ELSE ; reallocate = .FALSE.
         END IF
       END IF
     END IF

!  if a deallocation error occured, return if desired

     IF ( PRESENT( deallocate_error_fatal ) ) THEN
       IF ( deallocate_error_fatal .AND. alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     ELSE
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate ; RETURN
       END IF
     END IF

!  reallocate point to be of length len, checking for error returns

     IF ( reallocate ) ALLOCATE( point( len ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_deallocate
       IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )                 &
         bad_alloc = array_name
       IF ( PRESENT( out ) ) THEN
         IF ( PRESENT( array_name ) ) THEN
           IF ( out > 0 ) WRITE( out,                                          &
             "( ' ** Allocation error for ', A, ', status = ', I6 )" )         &
               TRIM( array_name ), alloc_status
         ELSE
           IF ( out > 0 ) WRITE( out,                                          &
             "( ' ** Allocation error status = ', I6 )" ) alloc_status
         END IF
       END IF
     END IF
     RETURN

!  End of FILTER_resize_filter

     END SUBROUTINE FILTER_resize_filter

! -*-*-  F I L T E R _ D E A L L O C _ F I L T E R   S U B R O U T I N E  -*-*-
 
    SUBROUTINE FILTER_dealloc_filter( point, status, alloc_status,             &
                                      array_name, bad_alloc, out )

!  deallocate the filter_type pointer array "point"

!  Dummy arguments

     INTEGER, INTENT( OUT ) :: status, alloc_status
     TYPE ( FILTER_filter_type ), POINTER, DIMENSION( : ) :: point
     INTEGER, OPTIONAL :: out
     CHARACTER ( LEN = 80 ), OPTIONAL :: array_name
     CHARACTER ( LEN = 80 ), OPTIONAL :: bad_alloc

     status = GALAHAD_ok ; alloc_status = 0
     IF ( PRESENT( bad_alloc ) ) bad_alloc = ''
     IF ( ASSOCIATED( point) ) THEN
       DEALLOCATE( point, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         status = GALAHAD_error_deallocate
         IF ( PRESENT( bad_alloc ) .AND. PRESENT( array_name ) )               &
           bad_alloc = array_name
         IF ( PRESENT( out ) ) THEN
           IF ( PRESENT( array_name ) ) THEN
             IF ( out > 0 ) WRITE( out,                                        &
               "( ' ** Deallocation error for ', A, ', status = ', I6 )" )     &
                 TRIM( array_name ), alloc_status
           ELSE
             IF ( out > 0 ) WRITE( out,                                        &
               "( ' ** Deallocation error status = ', I6 )" ) alloc_status
           END IF
         END IF
       END IF
     END IF
     RETURN

!  End of subroutine FILTER_dealloc_filter

     END SUBROUTINE FILTER_dealloc_filter

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-*-*- G A L A H A D -  F A S T R _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE FILTER_terminate( data, control, inform )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FILTER_data_type ), INTENT( INOUT ) :: data
     TYPE ( FILTER_control_type ), INTENT( IN ) :: control
     TYPE ( FILTER_inform_type ), INTENT( INOUT ) :: inform
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

     array_name = 'filter: data%filter1'
     CALL FILTER_dealloc_filter( data%filter1,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'filter: data%filter2'
     CALL FILTER_dealloc_filter( data%filter2,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     data%filter => NULL( ) ; data%filter1 => NULL( ) ; data%filter2 => NULL( )

     RETURN

!  End of subroutine FILTER_terminate

     END SUBROUTINE FILTER_terminate

!  End of module GALAHAD_FILTER

   END MODULE GALAHAD_FILTER_double
