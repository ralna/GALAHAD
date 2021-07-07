! THIS VERSION: GALAHAD 3.3 - 05/07/2021 AT 16:15 GMT.
   PROGRAM GALAHAD_HASH_EXAMPLE
   USE GALAHAD_HASH
   IMPLICIT NONE
   INTEGER :: degree, nhash
   TYPE ( HASH_data_type ) :: data
   TYPE ( HASH_control_type ) :: control        
   TYPE ( HASH_inform_type ) :: inform
   INTEGER, PARAMETER :: nchar = 10
   INTEGER, PARAMETER :: length = 100
   INTEGER, PARAMETER :: new_length = 200 
   INTEGER, PARAMETER :: nkeys1 = 8
   INTEGER, PARAMETER :: nkeys2 = 10
   INTEGER, PARAMETER :: nkeys3 = 3
   INTEGER, PARAMETER :: nkeys4 = 3
   INTEGER, PARAMETER :: nkeys5 = 11
   INTEGER :: i, position, inewem
   INTEGER :: MOVED_TO( length )
   CHARACTER ( LEN = 10 ) :: FIELD1( nkeys1 ) =                                &
       (/ 'ALPHA     ', 'BETA      ', 'GAMMA     ', 'DELTA     ',              &
          'X111111111', 'X111111112', 'X111111111', 'X111111114' /)
   CHARACTER ( LEN = 10 ) :: FIELD2( nkeys2 ) =                                &
       (/ 'ALPHA     ', 'BETA      ', 'GAMMA     ', 'DELTA     ',              &
          'EPSILON   ', 'X11111112 ', 'X111111113', 'X111111114',              &
          'X111111111', 'OMEGA     ' /)
   CHARACTER ( LEN = 10 ) :: FIELD3( nkeys3 ) =                                &
       (/ 'BETA      ', 'X111111112', 'OMEGA     ' /)
   CHARACTER ( LEN = 10 ) :: FIELD4( nkeys4 ) =                                &
       (/ 'OMEGA     ', 'A111111111', 'P110111111' /)
   CHARACTER ( LEN = 10 ) :: FIELD5( nkeys5 ) =                                &
       (/ 'ALPHA     ', 'BETA      ', 'GAMMA     ',                            &
          'DELTA     ', 'EPSILON   ', 'X111111112', 'X111111113',              &
          'X111111114', 'X111111111', 'OMEGA     ', 'X1111 1111' /)
!  set up the initial table
      CALL HASH_initialize( nchar, length, data, control, inform )
!  store a set of words in the table
      WRITE( 6, "( /, ' initial insertion', / )" )
      DO i = 1, nkeys1
        CALL HASH_insert( nchar, FIELD1( i ), position, data, control, inform )
        IF ( position > 0 ) THEN
          WRITE( 6, "( ' word ', A10, ' inserted  in table position ', I3 )" ) &
            FIELD1( i ), position
        ELSE
          WRITE(6, "(' word ', A10, ' already   in table position ', I3 )" )   &
            FIELD1( i ), - position
        END IF
      END DO
!  search the table for a second set of words
      DO i = 1, nkeys2
        CALL HASH_search( nchar, FIELD2( i ), position, data, control, inform )
        IF ( position > 0 ) THEN
          WRITE( 6, "( ' word ', A10, ' found     in table position ', I3 )" ) &
            FIELD2( i ), position
        ELSE
          WRITE( 6, "( ' word ', A10,' absent  from table')" ) FIELD2( i )
        END IF
      END DO
!  remove a third set of words from the table
      WRITE( 6, "( /, ' word removal', / )" )
      DO i = 1, nkeys3
        CALL HASH_remove( nchar, FIELD3( i ), position, data, control, inform )
        IF ( position > 0 ) THEN
          WRITE( 6, "( ' word ', A10,' removed from table position ', I3 )" )  &
            FIELD3( i ), position
        ELSE
          WRITE(6, "( ' word ', A10,' absent  from table' )" ) FIELD3( i )
        END IF
      END DO
!  re-search the table for the second set of words
      DO i = 1, nkeys2
        CALL HASH_search( nchar, FIELD2( i ), position, data, control, inform )
        IF ( position > 0 ) THEN
          WRITE( 6, "( ' word ', A10, ' found     in table position ', I3 )" ) &
            FIELD2( i ), position
        ELSE
          WRITE( 6, "( ' word ', A10,' absent  from table')" ) FIELD2( i )
        END IF
     END DO
!  increase the table size
      WRITE( 6, "( /, ' increase table size', / ) " )
      CALL HASH_rebuild( length, new_length, MOVED_TO, data, control, inform )
      DO i = 1, length
        IF ( MOVED_TO( i ) > 0 ) WRITE( 6, "( ' table entry in position ',     &
       &  I3, ' moved to position ', I3 )" ) i, MOVED_TO( i )
      END DO
!  store a fourth set of words in the table
      WRITE( 6, "( /, ' further insertion', / )" )
      DO i = 1, nkeys4
        CALL HASH_insert( nchar, FIELD4( i ), position, data, control, inform )
        IF ( position > 0 ) THEN
          WRITE( 6, "( ' word ', A12, ' inserted  in table position ', I3 )" ) &
            FIELD4( i ), position
        ELSE
          WRITE( 6, "( ' word ', A12, ' already   in table position ', I3 )" ) &
            FIELD4( i ), - position
        END IF
      END DO
!  re-search the table for the second set of words augmented with a further word
      DO i = 1, nkeys5
        CALL HASH_search( nchar, FIELD5( i ), position, data, control, inform )
        IF ( position > 0 ) THEN
          WRITE(6, "( ' word ', A12, ' found     in table position ', I3 )" )  &
            FIELD5( i ), position
        ELSE
          WRITE(6, "( ' word ', A12, ' absent  from table' )" ) FIELD5( i )
        END IF
      END DO
!  deallocate internal arrays
      CALL HASH_terminate( data, control, inform )
      STOP
   END PROGRAM GALAHAD_HASH_EXAMPLE
