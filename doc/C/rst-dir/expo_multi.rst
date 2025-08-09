.. _global:

.. _expo-multi-calls:

an example of a multi-precision call
------------------------------------

This is an example of how to use the ``expo`` package to solve a nonlinearly
constrained optimization problem with data provided both as
32-bit integers and double precision reals, and then again as
64-bit integers and single precision reals; the code is available in 
$GALAHAD/src/expo/C/expotm.c .

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The appropriate real and integer
types are supplied to each set of data. The structures and calls 
for the first set of data takes ``double`` reals and ``int32_t`` integers,
while for the second the data uses 
``float`` reals and ``int64_t`` integers, and appends ``_s_64`` to all
structure and function names.

.. include :: ../../../src/forthcoming/expo/C/expotm.c
   :code: C
