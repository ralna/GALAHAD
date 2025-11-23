.. _doxid-index_nrek:

GALAHAD NREK package
====================

.. module:: galahad.nrek

.. include:: ../../Python/nrek_intro.rst

.. include:: ../../Python/nrek_storage.rst

.. toctree::
	:hidden:

	nrek_functions.rst

introduction to function calls
------------------------------

To solve a given problem, functions from the nrek package must be 
called in the following order:

* :ref:`nrek_initialize <doxid-galahad__nrek_8h_1acb066d992c4ec394402bc7b7317e1163>` - provide default control parameters and set up initial data structures

* :ref:`nrek_read_specfile <doxid-galahad__nrek_8h_1adc7c56e7be2f7cc9d32921582d379b13>` (optional) - override control values by reading replacement values from a file

* :ref:`nrek_import <doxid-galahad__nrek_8h_1a4becded30e9b95fe7028b7799292c0af>` - set up problem data structures and fixed values

* :ref:`nrek_import_s <doxid-galahad__nrek_8h_1a427420b6025d522bb7b3c652e8c2be48>` - (optional) set up problem data structures and fixed values for the scaling matrix $S$, if any

* :ref:`nrek_reset_control <doxid-galahad__nrek_8h_1aae677e64bacb35354f49326815b694c3>` (optional) - possibly change control parameters if a sequence of problems are being solved

* :ref:`nrek_solve_problem <doxid-galahad__nrek_8h_1aadb8a751c29efcef663bf9560a1f9a8e>` - solve the trust-region problem

* :ref:`nrek_information <doxid-galahad__nrek_8h_1a3dda24010e564e2d6536cc7ea518451e>` (optional) - recover information about the solution and solution process

* :ref:`nrek_terminate <doxid-galahad__nrek_8h_1ab5cf0077db0631814fdd03599a585376>` - deallocate data structures

See the :ref:`examples <doxid-index_nrek_examples>` section for 
illustrations of use.

callable functions
------------------

.. include:: nrek_functions.rst

available structures
--------------------

.. include :: struct_nrek_control_type.rst

.. include :: struct_nrek_time_type.rst

.. include :: struct_nrek_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_nrek_examples:

example calls
-------------

This is an example of how to use the package to solve a trust-region subproblem;
the code is available in $GALAHAD/src/nrek/C/nrekt.c .
A variety of supported Hessian and constraint matrix storage formats are shown.

Notice that C-style indexing is used, and that this is flagged by setting 
``control.f_indexing`` to ``false``. The floating-point type ``rpc_``
is set in ``galahad_precision.h`` to ``double`` by default, but to ``float``
if the preprocessor variable ``SINGLE`` is defined. Similarly, the integer
type ``ipc_`` from ``galahad_precision.h`` is set to ``int`` by default, 
but to ``int64_t`` if the preprocessor variable ``INTEGER_64`` is defined.

.. include :: ../../../src/nrek/C/nrekt.c
   :code: C

This is the same example, but now fortran-style indexing is used;
the code is available in $GALAHAD/src/nrek/C/nrektf.c .

.. include :: ../../../src/nrek/C/nrektf.c
   :code: C

