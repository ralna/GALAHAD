program time
! aid to convert the control and inform types from ciface files to that 
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
          write( out, 1 ) trim( name ), trim( name )
1 format( '    PyDict_SetItemString(py_time, "', A, '",', /,                   &
          '                         PyLong_FromLong(time->', A, '));' )
        case ( 'r' )
          write( out, 2 ) trim( name ), trim( name )
2 format( '    PyDict_SetItemString(py_time, "', A, '",', /,                   &
          '                         PyFloat_FromDouble(time->', A, '));' )
        case ( 'l' )
          write( out, 3 ) trim( name ), trim( name )
3 format( '    PyDict_SetItemString(py_time, "', A, '",', /,                   &
          '                         PyBool_FromLong(time->', A, '));' )
        case ( 'c' )
          write( out, 4 ) trim( name ), trim( name )
4 format( '    PyDict_SetItemString(py_time, "', A, '",', /,                   &
          '                         PyUnicode_FromString(time->', A, '));' )
        case ( 't' )
          name =  REPEAT( ' ', 120 )
          name( 1 : line_length - i - 7 ) = line( i + 3 : line_length - 5 )
          write( out, 5 ) trim( name ), trim( name ), trim( name )
5 format( '    PyDict_SetItemString(py_time, "', A, '_time",', /,              &
          '                         ', A, '_make_inform_dict(&time->',         &
                                       A, '_time));' )
        end select
        exit
      end if
    end do
  end do
9 continue
  close( in )
  close( out )
  stop
end program time
