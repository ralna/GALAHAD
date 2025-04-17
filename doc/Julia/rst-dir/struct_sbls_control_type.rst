.. index:: pair: table; sbls_control_type
.. _doxid-structsbls__control__type:

sbls_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sbls_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          indmin::INT
          valmin::INT
          len_ulsmin::INT
          itref_max::INT
          maxit_pcg::INT
          new_a::INT
          new_h::INT
          new_c::INT
          preconditioner::INT
          semi_bandwidth::INT
          factorization::INT
          max_col::INT
          scaling::INT
          ordering::INT
          pivot_tol::T
          pivot_tol_for_basis::T
          zero_pivot::T
          static_tolerance::T
          static_level::T
          min_diagonal::T
          stop_absolute::T
          stop_relative::T
          remove_dependencies::Bool
          find_basis_by_transpose::Bool
          affine::Bool
          allow_singular::Bool
          perturb_to_make_definite::Bool
          get_norm_residual::Bool
          check_basis::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          symmetric_linear_solver::NTuple{31,Cchar}
          definite_linear_solver::NTuple{31,Cchar}
          unsymmetric_linear_solver::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          sls_control::sls_control_type{T,INT}
          uls_control::uls_control_type{T,INT}
	
.. _details-structsbls__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsbls__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsbls__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structsbls__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structsbls__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; indmin
.. _doxid-structsbls__control__type_indmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT indmin

initial estimate of integer workspace for SLS (obsolete)

.. index:: pair: variable; valmin
.. _doxid-structsbls__control__type_valmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT valmin

initial estimate of real workspace for SLS (obsolete)

.. index:: pair: variable; len_ulsmin
.. _doxid-structsbls__control__type_len_ulsmin:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT len_ulsmin

initial estimate of workspace for ULS (obsolete)

.. index:: pair: variable; itref_max
.. _doxid-structsbls__control__type_itref_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT itref_max

maximum number of iterative refinements with preconditioner allowed

.. index:: pair: variable; maxit_pcg
.. _doxid-structsbls__control__type_maxit_pcg:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit_pcg

maximum number of projected CG iterations allowed

.. index:: pair: variable; new_a
.. _doxid-structsbls__control__type_new_a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_a

how much has $A$ changed since last factorization: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; new_h
.. _doxid-structsbls__control__type_new_h:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_h

how much has $H$ changed since last factorization: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; new_c
.. _doxid-structsbls__control__type_new_c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_c

how much has $C$ changed since last factorization: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; preconditioner
.. _doxid-structsbls__control__type_preconditioner:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT preconditioner

which preconditioner to use:

* 0 selected automatically

* 1 explicit with $G = I$

* 2 explicit with $G = H$

* 3 explicit with $G =$ diag(max($H$,min_diag))

* 4 explicit with $G =$ band $(H)$

* 5 explicit with $G =$ (optional, diagonal) $D$

* 11 explicit with $G_{11} = 0$, $G_{21} = 0$, $G_{22} = H_{22}$

* 12 explicit with $G_{11} = 0$, $G_{21} = H_{21}$, $G_{22} = H_{22}$

* -1 implicit with $G_{11} = 0$, $G_{21} = 0$, $G_{22} = I$

* -2 implicit with $G_{11} = 0$, $G_{21} = 0$, $G_{22} = H_{22}$

.. index:: pair: variable; semi_bandwidth
.. _doxid-structsbls__control__type_semi_bandwidth:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth

the semi-bandwidth for band(H)

.. index:: pair: variable; factorization
.. _doxid-structsbls__control__type_factorization:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization

the explicit factorization used:

* 0 selected automatically

* 1 Schur-complement if $G$ is diagonal and successful otherwise augmented system

* 2 augmented system

* 3 C_NULL-space

* 4 Schur-complement if $G$ is diagonal and successful otherwise failure

* 5 Schur-complement with pivoting if $G$ is diagonal and successful otherwise failure

.. index:: pair: variable; max_col
.. _doxid-structsbls__control__type_max_col:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_col

maximum number of nonzeros in a column of $A$ for Schur-complement factorization

.. index:: pair: variable; scaling
.. _doxid-structsbls__control__type_scaling:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT scaling

not used at present

.. index:: pair: variable; ordering
.. _doxid-structsbls__control__type_ordering:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ordering

see scaling

.. index:: pair: variable; pivot_tol
.. _doxid-structsbls__control__type_pivot_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol

the relative pivot tolerance used by ULS (obsolete)

.. index:: pair: variable; pivot_tol_for_basis
.. _doxid-structsbls__control__type_pivot_tol_for_basis:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol_for_basis

the relative pivot tolerance used by ULS when determining the basis matrix

.. index:: pair: variable; zero_pivot
.. _doxid-structsbls__control__type_zero_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_pivot

the absolute pivot tolerance used by ULS (obsolete)

.. index:: pair: variable; static_tolerance
.. _doxid-structsbls__control__type_static_tolerance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T static_tolerance

not used at present

.. index:: pair: variable; static_level
.. _doxid-structsbls__control__type_static_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T static_level

see static_tolerance

.. index:: pair: variable; min_diagonal
.. _doxid-structsbls__control__type_min_diagonal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T min_diagonal

the minimum permitted diagonal in diag(max($H$,min_diag))

.. index:: pair: variable; stop_absolute
.. _doxid-structsbls__control__type_stop_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_absolute

the required absolute and relative accuracies

.. index:: pair: variable; stop_relative
.. _doxid-structsbls__control__type_stop_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_relative

see stop_absolute

.. index:: pair: variable; remove_dependencies
.. _doxid-structsbls__control__type_remove_dependencies:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool remove_dependencies

preprocess equality constraints to remove linear dependencies

.. index:: pair: variable; find_basis_by_transpose
.. _doxid-structsbls__control__type_find_basis_by_transpose:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool find_basis_by_transpose

determine implicit factorization preconditioners using a basis of A found by examining A's transpose

.. index:: pair: variable; affine
.. _doxid-structsbls__control__type_affine:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool affine

can the right-hand side $c$ be assumed to be zero?

.. index:: pair: variable; allow_singular
.. _doxid-structsbls__control__type_allow_singular:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool allow_singular

do we tolerate "singular" preconditioners?

.. index:: pair: variable; perturb_to_make_definite
.. _doxid-structsbls__control__type_perturb_to_make_definite:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool perturb_to_make_definite

if the initial attempt at finding a preconditioner is unsuccessful, should the diagonal be perturbed so that a second attempt succeeds?

.. index:: pair: variable; get_norm_residual
.. _doxid-structsbls__control__type_get_norm_residual:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool get_norm_residual

compute the residual when applying the preconditioner?

.. index:: pair: variable; check_basis
.. _doxid-structsbls__control__type_check_basis:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool check_basis

if an implicit or C_NULL-space preconditioner is used, assess and correct for ill conditioned basis matrices

.. index:: pair: variable; space_critical
.. _doxid-structsbls__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structsbls__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structsbls__control__type_symmetric_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

indefinite linear equation solver

.. index:: pair: variable; definite_linear_solver
.. _doxid-structsbls__control__type_definite_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

definite linear equation solver

.. index:: pair: variable; unsymmetric_linear_solver
.. _doxid-structsbls__control__type_unsymmetric_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char unsymmetric_linear_solver[31]

unsymmetric linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structsbls__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structsbls__control__type_sls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; uls_control
.. _doxid-structsbls__control__type_uls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`uls_control_type<doxid-structuls__control__type>` uls_control

control parameters for ULS

