s/ c_double / c_float128 /g
s/ C_DOUBLE / C_FLOAT128 /g
s/ c_double[ ]*$/ c_float128/g
s/ C_DOUBLE[ ]*$/ C_FLOAT128/g
s/(c_double)/(c_float128)/g
s/(C_DOUBLE)/(C_FLOAT128)/g
s/_double[ ]*$/_quadruple/g
s/_double,/_quadruple,/g
s/_double /_quadruple /g
s/_double(/_quadruple(/g
s/_DOUBLE[ ]*$/_QUADRUPLE/g
s/_DOUBLE,/_QUADRUPLE,/g
s/_DOUBLE /_QUADRUPLE /g
s/_DOUBLE(/_QUADRUPLE(/g
s/dvalues/qvalues/g
s/D+0/E+0/g
s/0d0/0e0/g
s/real_bytes = 8/real_bytes = 16/g
s/MC13ED/MC13EQ/g
s/MA61AD/MA61AQ/g
s/MA77AD/MA77AQ/g
s/MC61AD/MC61AQ/g
s/MC77AD/MC77AQ/g
s/MC21BD/MC21BQ/g
s/dmumps/qmumps/g
s/DMUMPS/QMUMPS/g
s/symmetric_linear_solver = "ssids"/symmetric_linear_solver = "sytr "/g
s/definite_linear_solver = "ssids"/definite_linear_solver = "sytr "/g
