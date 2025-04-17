.. index:: pair: struct; eqp_control_type
.. _doxid-structeqp__control__type:

eqp_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct eqp_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          factorization::INT
          max_col::INT
          indmin::INT
          valmin::INT
          len_ulsmin::INT
          itref_max::INT
          cg_maxit::INT
          preconditioner::INT
          semi_bandwidth::INT
          new_a::INT
          new_h::INT
          sif_file_device::INT
          pivot_tol::T
          pivot_tol_for_basis::T
          zero_pivot::T
          inner_fraction_opt::T
          radius::T
          min_diagonal::T
          max_infeasibility_relative::T
          max_infeasibility_absolute::T
          inner_stop_relative::T
          inner_stop_absolute::T
          inner_stop_inter::T
          find_basis_by_transpose::Bool
          remove_dependencies::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          generate_sif_file::Bool
          sif_file_name::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          fdc_control::fdc_control_type{T,INT}
          sbls_control::sbls_control_type{T,INT}
          gltr_control::gltr_control_type{T,INT}
	
.. _details-structeqp__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structeqp__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structeqp__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structeqp__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structeqp__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level

.. index:: pair: variable; factorization
.. _doxid-structeqp__control__type_factorization:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization

the factorization to be used. Possible values are /li 0 automatic /li 1 Schur-complement factorization /li 2 augmented-system factorization (OBSOLETE)

.. index:: pair: variable; max_col
.. _doxid-structeqp__control__type_max_col:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_col

the maximum number of nonzeros in a column of A which is permitted with the Schur-complement factorization (OBSOLETE)

.. index:: pair: variable; indmin
.. _doxid-structeqp__control__type_indmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT indmin

an initial guess as to the integer workspace required by SBLS (OBSOLETE)

.. index:: pair: variable; valmin
.. _doxid-structeqp__control__type_valmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT valmin

an initial guess as to the real workspace required by SBLS (OBSOLETE)

.. index:: pair: variable; len_ulsmin
.. _doxid-structeqp__control__type_len_ulsmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT len_ulsmin

an initial guess as to the workspace required by ULS (OBSOLETE)

.. index:: pair: variable; itref_max
.. _doxid-structeqp__control__type_itref_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT itref_max

the maximum number of iterative refinements allowed (OBSOLETE)

.. index:: pair: variable; cg_maxit
.. _doxid-structeqp__control__type_cg_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_maxit

the maximum number of CG iterations allowed. If cg_maxit < 0, this number will be reset to the dimension of the system + 1

.. index:: pair: variable; preconditioner
.. _doxid-structeqp__control__type_preconditioner:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT preconditioner

the preconditioner to be used for the CG. Possible values are

* 0 automatic

* 1 no preconditioner, i.e, the identity within full factorization

* 2 full factorization

* 3 band within full factorization

* 4 diagonal using the barrier terms within full factorization (OBSOLETE)

* 5 optionally supplied diagonal, G = D

.. index:: pair: variable; semi_bandwidth
.. _doxid-structeqp__control__type_semi_bandwidth:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth

the semi-bandwidth of a band preconditioner, if appropriate (OBSOLETE)

.. index:: pair: variable; new_a
.. _doxid-structeqp__control__type_new_a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_a

how much has A changed since last problem solved: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; new_h
.. _doxid-structeqp__control__type_new_h:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_h

how much has H changed since last problem solved: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; sif_file_device
.. _doxid-structeqp__control__type_sif_file_device:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; pivot_tol
.. _doxid-structeqp__control__type_pivot_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol

the threshold pivot used by the matrix factorization. See the documentation for SBLS for details (OBSOLETE)

.. index:: pair: variable; pivot_tol_for_basis
.. _doxid-structeqp__control__type_pivot_tol_for_basis:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol_for_basis

the threshold pivot used by the matrix factorization when finding the ba See the documentation for ULS for details (OBSOLETE)

.. index:: pair: variable; zero_pivot
.. _doxid-structeqp__control__type_zero_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_pivot

any pivots smaller than zero_pivot in absolute value will be regarded to zero when attempting to detect linearly dependent constraints (OBSOLETE)

.. index:: pair: variable; inner_fraction_opt
.. _doxid-structeqp__control__type_inner_fraction_opt:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T inner_fraction_opt

the computed solution which gives at least inner_fraction_opt times the optimal value will be found (OBSOLETE)

.. index:: pair: variable; radius
.. _doxid-structeqp__control__type_radius:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T radius

an upper bound on the permitted step (-ve will be reset to an appropriat large value by eqp_solve)

.. index:: pair: variable; min_diagonal
.. _doxid-structeqp__control__type_min_diagonal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T min_diagonal

diagonal preconditioners will have diagonals no smaller than min_diagonal (OBSOLETE)

.. index:: pair: variable; max_infeasibility_relative
.. _doxid-structeqp__control__type_max_infeasibility_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T max_infeasibility_relative

if the constraints are believed to be rank defficient and the residual at a "typical" feasible point is larger than max( max_infeasibility_relative \* norm A, max_infeasibility_absolute ) the problem will be marked as infeasible

.. index:: pair: variable; max_infeasibility_absolute
.. _doxid-structeqp__control__type_max_infeasibility_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T max_infeasibility_absolute

see max_infeasibility_relative

.. index:: pair: variable; inner_stop_relative
.. _doxid-structeqp__control__type_inner_stop_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T inner_stop_relative

the computed solution is considered as an acceptable approximation to th minimizer of the problem if the gradient of the objective in the preconditioning(inverse) norm is less than max( inner_stop_relative \* initial preconditioning(inverse) gradient norm, inner_stop_absolute )

.. index:: pair: variable; inner_stop_absolute
.. _doxid-structeqp__control__type_inner_stop_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T inner_stop_absolute

see inner_stop_relative

.. index:: pair: variable; inner_stop_inter
.. _doxid-structeqp__control__type_inner_stop_inter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T inner_stop_inter

see inner_stop_relative

.. index:: pair: variable; find_basis_by_transpose
.. _doxid-structeqp__control__type_find_basis_by_transpose:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool find_basis_by_transpose

if .find_basis_by_transpose is true, implicit factorization precondition will be based on a basis of A found by examining A's transpose (OBSOLETE)

.. index:: pair: variable; remove_dependencies
.. _doxid-structeqp__control__type_remove_dependencies:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool remove_dependencies

if .remove_dependencies is true, the equality constraints will be preprocessed to remove any linear dependencies

.. index:: pair: variable; space_critical
.. _doxid-structeqp__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structeqp__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structeqp__control__type_generate_sif_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structeqp__control__type_sif_file_name:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} sif_file_name

name of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structeqp__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; fdc_control
.. _doxid-structeqp__control__type_fdc_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sbls_control
.. _doxid-structeqp__control__type_sbls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; gltr_control
.. _doxid-structeqp__control__type_gltr_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>` gltr_control

control parameters for GLTR

