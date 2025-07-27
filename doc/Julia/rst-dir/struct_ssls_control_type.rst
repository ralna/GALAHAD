.. index:: pair: table; ssls_control_type
.. _doxid-structssls__control__type:

ssls_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ssls_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          space_critical::Bool
          deallocate_error_fatal::Bool
          symmetric_linear_solver::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          sls_control::sls_control_type{T,INT}
	
.. _details-structssls__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structssls__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structssls__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structssls__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structssls__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

.. index:: pair: variable; space_critical
.. _doxid-structssls__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structssls__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structssls__control__type_symmetric_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

indefinite linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structssls__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structssls__control__type_sls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS
