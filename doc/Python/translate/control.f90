program control
! aid to convert the control types from ciface files to that 
! required by pyiface files
! nick gould, september 21st 2022
  integer :: in = 5
  integer :: out = 6
  integer :: i, j, line_length
  integer :: name_start
  character ( len = 1 ) :: type
  character ( len = 120 ) :: name
  character ( len = 100 ) :: line

  open( in )
  open( out )
  do
! read in next line
    line =  REPEAT( ' ', 100 )
    read( in, "( A )", end = 9, err = 9 ) line
    line = ADJUSTL( line ) 
    line_length = len( trim( line ) ) 
! interpret line
    if ( len( trim( line ) ) == 0 ) cycle ! blank line
    do i = 1, line_length
      if ( line( i : i + 3 ) == 'REAL' ) THEN
        type = 'r'
      else if ( line( i : i + 6 ) == 'LOGICAL' ) THEN
        type = 'l'
      else if ( line( i : i + 6 ) == 'INTEGER' ) THEN
        type = 'i'
      else if ( line( i : i + 8 ) == 'CHARACTER' ) THEN
        type = 'c'
      else if ( line( i : i + 3 ) == 'TYPE' ) THEN
        type = 't'
      else if ( line( i : i + 2 ) == ':: ' ) THEN
        do j = i + 3, line_length
          if ( line( j : j ) == ' ' ) then
            line_length = j - 1
            exit
          end if
        end do
        name =  REPEAT( ' ', 120 )
        name( 1 : line_length - i - 2 ) = line( i + 3 : line_length )
!       write( out, "( A, ' ', A )" ) type, trim( name )
        select case ( type )
        case ( 'i' )
          write( out, 1 ) trim( name ), trim( name ), trim( name )
1 format( '        if(strcmp(key_name, "', A, '") == 0){', /,                  &
          '            if(!parse_int_option(value, "', A, '",', /,             &
         '                                  &control->', A, '))', /,           &
          '                return false;', /,                                  &
          '            continue;', /,                                          &
          '        }')
        case ( 'r' )
          write( out, 2 ) trim( name ), trim( name ), trim( name )
2 format( '        if(strcmp(key_name, "', A, '") == 0){', /,                  &
          '            if(!parse_double_option(value, "', A, '",', /,          &
            '                                  &control->', A, '))', /,        &
          '                return false;', /,                                  &
          '            continue;', /,                                          &
          '        }')
        case ( 'l' )
          write( out, 3 ) trim( name ), trim( name ), trim( name )
3 format( '        if(strcmp(key_name, "', A, '") == 0){', /,                  &
          '            if(!parse_bool_option(value, "', A, '",', /,            &
           '                                  &control->', A, '))', /,         &
          '                return false;', /,                                  &
          '            continue;', /,                                          &
          '        }')
        case ( 'c' )
          write( out, 4 ) trim( name ), trim( name ), trim( name ), trim( name )
4 format( '        if(strcmp(key_name, "', A, '") == 0){', /,                  &
          '            if(!parse_char_option(value, "', A, '",', /,            &
          '                                  control->', A, ',', /,            &
          '                                  sizeof(control->', A, ')))', /,   &
          '                return false;', /,                                  &
          '            continue;', /,                                          &
          '        }')
        case ( 't' )
          name =  REPEAT( ' ', 120 )
          name( 1 : line_length - i - 10 ) = line( i + 3 : line_length - 8 )
          write( out, 5 ) trim( name ), trim( name ), trim( name )
5 format( '        //if(strcmp(key_name, "', A, '_options") == 0){', /,        &
          '        //    if(!', A, '_update_control(&control->', A,            &
          '_control, value))', /,                                              &
          '        //        return false;', /,                                &
          '        //    continue;', /,                                        &
          '        //}')
        end select
        exit
      end if
    end do
  end do
9 continue
  close( in )
  close( out )
  stop
end program control
