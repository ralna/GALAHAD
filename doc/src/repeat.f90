   PROGRAM REPEAT
   INTEGER :: i, lend, l, l1, l2
   CHARACTER ( len = 100 ) :: lcpu, lits
   open( unit = 11, file = "cpu" )
   open( unit = 12, file = "its" )
   DO 
     lcpu = BLANK_FIELD( )
     lits = BLANK_FIELD( )
     READ( 11, "(A100)", end = 990 ) lcpu
     READ( 12, "(A100)", end = 990 ) lits
     i = 1
     lend = len_trim( lits )
     l1 = 1
     l2 = len_trim( lcpu )
 100 CONTINUE   
     IF ( lits(i:i) == '(' ) THEN
       DO l = l1, l2
         IF ( lcpu(l:l) == '(' ) THEN
           l1 = l + 3
           lits(i:i+2) = lcpu(l:l+2)
         END IF
       END DO
       i = i + 2
     END IF
     i = i + 1
     IF ( i <= lend ) GO TO 100
     WRITE( 6, * ) lits( : lend )
   END DO
 990 CONTINUE
    STOP 
   CONTAINS
     FUNCTION BLANK_FIELD( )
     CHARACTER ( len = 100 ) :: BLANK_FIELD
     DO i = 1, 100
       BLANK_FIELD( i: i ) = ' '
     END DO
     END FUNCTION BLANK_FIELD
   END PROGRAM REPEAT
