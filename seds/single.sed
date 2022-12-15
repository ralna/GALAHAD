s/ c_double / c_float /g
s/ C_DOUBLE / C_FLOAT /g
s/ c_double[ ]*$/ c_float/g
s/ C_DOUBLE[ ]*$/ C_FLOAT/g
s/(c_double)/(c_float)/g
s/(C_DOUBLE)/(C_FLOAT)/g
s/_double[ ]*$/_single/g
s/_double,/_single,/g
s/_double /_single /g
s/_double(/_single(/g
s/_DOUBLE[ ]*$/_SINGLE/g
s/_DOUBLE,/_SINGLE,/g
s/_DOUBLE /_SINGLE /g
s/_DOUBLE(/_SINGLE(/g
s/dvalues/svalues/g
s/D+0/E+0/g
s/0d0/0e0/g
s/real_bytes = 8/real_bytes = 4/g
s/MC13ED/MC13E/g
s/MA61AD/MA61A/g
s/MA77AD/MA77A/g
s/MC61AD/MC61A/g
s/MC77AD/MC77A/g
s/MC21BD/MC21B/g
s/dmumps/smumps/g
s/DMUMPS/SMUMPS/g
s/symmetric_linear_solver = "ssids"/symmetric_linear_solver = "sytr "/g
s/definite_linear_solver = "ssids"/definite_linear_solver = "sytr "/g
