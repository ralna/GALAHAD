program translate
! aid to convert the control and inform types from ciface .h files to that 
! required by pyiface files
! nick gould, august 30th 2022
  integer :: in = 5
  integer :: out = 6
  integer :: i, j, pos, line_length, total_line_length, output_line_length
  integer :: name_start, word_length, word_start, word_end, subword_length
  logical :: got_type, beyond, maths
  character ( len = 1 ) :: type
  character ( len = 120 ) :: name
  character ( len = 100 ) :: line
  character ( len = 10000 ) :: total_line
  character ( len = 72 ) :: output_line, word, subword

  open( in )
  open( out )
  pos = 1
  total_line =  REPEAT( ' ', 10000 )
  do
! read in next line
    line =  REPEAT( ' ', 100 )
    read( in, "( A )", end = 1, err = 1 ) line
    line = ADJUSTL( line ) 
    line_length = len( trim( line ) ) 
!write(6,*) 'in - ', trim( line )
! interpret line
    if ( len( trim( line ) ) == 0 ) then ! blank line
      cycle
    else if ( line( 1 : 10 ) == '/// \brief' ) then ! brief line
      cycle
    else if ( line( 1 : 4 ) == '/// ' ) then ! description line
! append line to total_line
      do i = 5, line_length
        if ( i < line_length .and. line( i : i + 1 ) == '  ' ) cycle
        total_line( pos : pos ) = line( i : i )
        pos = pos + 1
      end do
      total_line( pos : pos ) = ' '
      pos = pos + 1
!write(6,*) 't * ', trim( total_line )
    else ! definition line
! extract variable type and name
      name =  REPEAT( ' ', 120 )
      got_type = .false.
      do i = 1, line_length
!  find type
        if ( .NOT. got_type ) then
          if ( line( i : i ) == ' ' ) then
            if ( line( 1 : 7 ) == 'int64_t' ) then
              type = 'l'
            else if ( line( 1 : 3 ) == 'int' .or. line( 1 : 4 ) == 'long' ) then
              type = 'i'
            else if ( line( 1 : 4 ) == 'real' ) then
              type = 'r'
            else if ( line( 1 : 4 ) == 'bool' ) then
              type = 'b'
            else if ( line( 1 : 4 ) == 'char' ) then
              type = 'c'
            else if ( line( 1 : 6 ) == 'struct' ) then
              type = 's'
            else
!write(6,*) TRIM( line )
              write(6,*) ' unknown type ', line( 1 : i ), ' stopping'
              stop
            end if
            got_type = .true.
            name_start = i + 1
            cycle
          end if
!  find name
        else
          if ( line( i : i ) == ' ' .or. line( i : i ) == '[' .or.           &
               line( i : i ) == ';' ) then
            name( 1 : i - name_start ) = line( name_start : i - 1 )
!if ( trim( name ) == 'lbfgs_vectors' ) stop
            if ( type == 'l' ) then
!             write( out, "( A, A, A )" ) '"    ', TRIM( name ), ' : long\n"'
              write( out, "( A, A, A )" ) '          ', TRIM( name ), ' : long'
            else if ( type == 'i' ) then
!             write( out, "( A, A, A )" ) '"    ', TRIM( name ), ' : int\n"'
              write( out, "( A, A, A )" ) '          ', TRIM( name ), ' : int'
            else if ( type == 'r' ) then
!             write( out, "( A, A, A )" ) '"    ', TRIM( name ), ' : float\n"'
              write( out, "( A, A, A )" ) '          ', TRIM( name ), ' : float'
            else if ( type == 'b' ) then
!             write( out, "( A, A, A )" ) '"    ', TRIM( name ), ' : bool\n"'
              write( out, "( A, A, A )" ) '          ', TRIM( name ), ' : bool'
            else if ( type == 'c' ) then
!             write( out, "( A, A, A )" ) '"    ', TRIM( name ), ' : str\n"'
              write( out, "( A, A, A )" ) '          ', TRIM( name ), ' : str'
            else if ( type == 's' ) then
              name_start = i + 1
              do j = name_start + 1,  line_length
                if ( line( j : j ) == ' ' .or. line( j : j ) == '[' .or.       &
                     line( j : j ) == ';' ) exit
              end do
              name =  REPEAT( ' ', 120 )
              name( 1 : j - name_start ) = line( name_start : j - 1 )
!             write( out, "( A, A, A )" ) '"    ', TRIM( name ), ' : dict\n"'
              write( out, "( A, A, A )" ) '          ', TRIM( name ), ' : dict'
            end if
            output_line = REPEAT( ' ', 72 )
!           output_line( 1 : 6 ) = '"     '
            output_line( 1 : 12 ) = '      '
!           output_line_length = 7
            output_line_length = 13
!  format the output lines
            maths = .false.
            word = REPEAT( ' ', 72 )
            word_start = 1
            total_line_length = len( trim( total_line ) )
            do j = 1, total_line_length
              if ( j + 2 <= total_line_length ) then
                if ( total_line( j : j + 2 ) == '\f$' ) maths = .not. maths
              end if
              beyond = total_line( j : j ) == ' ' .and. .not. maths
              if ( beyond ) then ! beyond end of a not maths word
                word_end = j - 1
                if ( total_line( word_start : word_start ) == '.' ) then
                 word_length = word_end - word_start + 5
                  word( 1 : word_length )                                      &
                    = ' ``' // total_line( word_start + 1 : word_end ) // '``' 
                else
                  word_length = word_end - word_start + 2
                  word( 1 : word_length  )                                     &
                    = ' ' // total_line( word_start : word_end ) 
                end if
!write(6,*) word( 1 : word_length )
                word_start = j + 1

                if ( word_length == 4 ) then
                  if ( word( 1 : 4 ) == ' \li' ) then
!                   if ( output_line_length > 7 ) then
                    if ( output_line_length > 13 ) then
!                     output_line( output_line_length :                        &
!                                  output_line_length + 2 ) = '\n"'
                      write( out, "( A )" ) trim( output_line )
                      output_line = REPEAT( ' ', 72 )
!                     output_line( 1 : 8 ) = '"      *'
                      output_line( 1 : 14 ) = '             *'
                    else
                      output_line( 7 : 14 ) = '       *'
                    end if
                    output_line_length = 15
                    cycle
                  end if
                end if

                call modify_word( word, word_length )

!               if ( output_line_length + word_length > 69 ) then
                if ( output_line_length + word_length > 72 ) then
!                 output_line( output_line_length :                            &
!                              output_line_length + 2 ) = '\n"'
                  write( out, "( A )" ) trim( output_line )
                  output_line = REPEAT( ' ', 72 )
!                 output_line( 1 : 6 ) = '"     '
!                 output_line( 1 : 6 ) = '      '
                  output_line( 1 : 12 ) = '            '
!                 output_line_length = 7
                  output_line_length = 13
                end if

                output_line( output_line_length :                              &
                             output_line_length + word_length ) =              &
                  word( 1 : word_length )
                output_line_length = output_line_length + word_length
              end if
            end do
!  format the final line
            word_end = total_line_length
            if ( total_line( word_start : word_start ) == '.' ) then
             word_length = word_end - word_start + 5
              word( 1 : word_length )                                          &
                = ' ``' // total_line( word_start + 1 : word_end ) // '``' 
            else
              word_length = word_end - word_start + 2
              word( 1 : word_length  )                                         &
                = ' ' // total_line( word_start : word_end ) 
            end if

            call modify_word( word, word_length )

!           if ( output_line_length + word_length > 69 ) then
            if ( output_line_length + word_length > 72 ) then
!             output_line( output_line_length :                                &
!                          output_line_length + 2 ) = '\n"'
              write( out, "( A )" ) trim( output_line )
              output_line = REPEAT( ' ', 72 )
!             output_line( 1 : 6 ) = '"     '
              output_line( 1 : 12 ) = '            '
!             output_line_length = 7
              output_line_length = 13
            end if

            output_line( output_line_length :                                  &
                         output_line_length + word_length - 1 ) =              &
              word( 1 : word_length )
            output_line_length = output_line_length + word_length
!           if ( output_line_length > 7 ) then
            if ( output_line_length > 13 ) then
              if ( output_line( output_line_length - 1 :                       &
                                output_line_length - 1 ) == '.' ) then
!               output_line( output_line_length :                              &
!                            output_line_length + 2 ) = '\n"'
              else
!               output_line( output_line_length :                              &
!                            output_line_length + 3 ) = '.\n"'
                output_line( output_line_length :                              &
                             output_line_length ) = '.'
              end if
              write( out, "( A )" ) trim( output_line )
            end if
            total_line = REPEAT( ' ', 10000 )
            pos = 1
            exit
          end if
        end if
      end do
    end if
  end do
1 continue
  close( in )
  close( out )
  stop
contains
! modify key words so that sphinx understands
  subroutine modify_word( word, word_length )
  integer, intent( inout ) :: word_length
  character ( len = 72 ), intent( inout ) :: word
  character ( len = 72 ) :: subword
  integer :: subword_length
! change true and fale to True and False
  if ( word_length == 5 ) then
    if ( word( 1 : 5 ) == ' true' ) word( 1 : 5 ) = ' True'
  end if
  if ( word_length == 6 ) then
    if ( word( 1 : 6 ) == ' false' ) then
      word( 1 : 5 ) = ' False'
    else if ( word( 1 : 6 ) == ' true,' ) then 
      word( 1 : 6 ) = ' True,'
    else if ( word( 1 : 6 ) == ' true.' ) then 
      word( 1 : 6 ) = ' True.'
    end if
  end if
  if ( word_length == 7 ) then
    if ( word( 1 : 7 ) == ' false,' ) then 
      word( 1 : 7 ) = ' False,'
    else if ( word( 1 : 7 ) == ' false.' ) then 
      word( 1 : 7 ) = ' False.'
    end if
  end if
!  change latex comparison operators to typewriter ones
  if ( word_length == 11 ) then
    if ( word( 1 : 11 ) == ' \f$\leq\f$' ) then
      word( 1 : 3 ) = ' <='
      word_length = 3
    else if ( word( 1 : 11 ) == ' \f$\geq\f$' ) then
      word( 1 : 3 ) = ' >='
      word_length = 3
    else if ( word( 1 : 11 ) == ' \f$\neq\f$' ) then
      word( 1 : 3 ) = ' /='
      word_length = 3
    end if
  end if
!  change doxygen maths markers to sphinx ones
  if ( word_length > 7 ) then
    if ( word( 2 : 4 ) == '\f$' .and.                                          &
         word( word_length - 2 : word_length ) == '\f$' ) then
     subword_length = word_length - 7
     subword( 1 : subword_length ) = word( 5 : word_length - 3 )
!    word_length = subword_length + 9
!    word( 1 : word_length )                                                   &
!      = ' :math:`' // subword( 1 : subword_length ) // '`'
     word_length = subword_length + 3
     word( 1 : word_length ) = ' $' // subword( 1 : subword_length ) // '$'
    end if
  end if
  if ( word_length > 8 ) then
    if ( word( 2 : 4 ) == '\f$' .and.                                          &
         word( word_length - 3 : word_length ) == '\f$,' ) then
     subword_length = word_length - 8
     subword( 1 : subword_length ) = word( 5 : word_length - 4 )
!    word_length = subword_length + 10
!    word( 1 : word_length )                                                   &
!      = ' :math:`' // subword( 1 : subword_length ) // '`,'
     word_length = subword_length + 4
     word( 1 : word_length ) = ' $' // subword( 1 : subword_length ) // '$,'
    end if
    if ( word( 2 : 4 ) == '\f$' .and.                                          &
         word( word_length - 3 : word_length ) == '\f$.' ) then
     subword_length = word_length - 8
     subword( 1 : subword_length ) = word( 5 : word_length - 4 )
!    word_length = subword_length + 10
!    word( 1 : word_length )                                                   &
!      = ' :math:`' // subword( 1 : subword_length ) // '`.'
     word_length = subword_length + 4
     word( 1 : word_length )                                                   &
       = ' $' // subword( 1 : subword_length ) // '$.'
    end if
  end if
  end subroutine modify_word

end program translate
