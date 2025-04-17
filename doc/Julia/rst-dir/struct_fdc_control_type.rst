.. index:: pair: table; fdc_control_type
.. _doxid-structfdc__control__type:

fdc_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct fdc_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          indmin::INT
          valmin::INT
          pivot_tol::T
          zero_pivot::T
          max_infeas::T
          use_sls::Bool
          scale::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          symmetric_linear_solver::NTuple{31,Cchar}
          unsymmetric_linear_solver::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          sls_control::sls_control_type{T,INT}
          uls_control::uls_control_type{T,INT}
	
.. _details-structfdc__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structfdc__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structfdc__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structfdc__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structfdc__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; indmin
.. _doxid-structfdc__control__type_indmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT indmin

initial estimate of integer workspace for sls (obsolete)

.. index:: pair: variable; valmin
.. _doxid-structfdc__control__type_valmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT valmin

initial estimate of real workspace for sls (obsolete)

.. index:: pair: variable; pivot_tol
.. _doxid-structfdc__control__type_pivot_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol

the relative pivot tolerance (obsolete)

.. index:: pair: variable; zero_pivot
.. _doxid-structfdc__control__type_zero_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_pivot

the absolute pivot tolerance used (obsolete)

.. index:: pair: variable; max_infeas
.. _doxid-structfdc__control__type_max_infeas:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T max_infeas

the largest permitted residual

.. index:: pair: variable; use_sls
.. _doxid-structfdc__control__type_use_sls:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool use_sls

choose whether SLS or ULS is used to determine dependencies

.. index:: pair: variable; scale
.. _doxid-structfdc__control__type_scale:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool scale

should the rows of A be scaled to have unit infinity norm or should no scaling be applied

.. index:: pair: variable; space_critical
.. _doxid-structfdc__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structfdc__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structfdc__control__type_symmetric_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

symmetric (indefinite) linear equation solver

.. index:: pair: variable; unsymmetric_linear_solver
.. _doxid-structfdc__control__type_unsymmetric_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char unsymmetric_linear_solver[31]

unsymmetric linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structfdc__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structfdc__control__type_sls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; uls_control
.. _doxid-structfdc__control__type_uls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`uls_control_type<doxid-structuls__control__type>` uls_control

control parameters for ULS

