.. index:: pair: table; sbls_control_type
.. _doxid-structsbls__control__type:

sbls_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sbls.h>
	
	struct sbls_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structsbls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structsbls__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structsbls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structsbls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`indmin<doxid-structsbls__control__type_1a5031bbc31f94e4cba6a540a3182b6d80>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`valmin<doxid-structsbls__control__type_1a0e142fa8dc9c363c3c2993b6129b0955>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`len_ulsmin<doxid-structsbls__control__type_1a600c95211b782597cd1b2475bb2c54c6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itref_max<doxid-structsbls__control__type_1a903ba4ef0869186a65d4c32459a6a0ed>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit_pcg<doxid-structsbls__control__type_1aac98c23514fe50d29412bb0ecfacc8f2>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_a<doxid-structsbls__control__type_1a7bea45d51fd9384037bbbf82f7750ce6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_h<doxid-structsbls__control__type_1ae60c5b5b987dd62f25253ba4164813f5>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_c<doxid-structsbls__control__type_1a9eed9b4ef920c669df5ff1a7c67e3047>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`preconditioner<doxid-structsbls__control__type_1adf7719f1a4491459e361e80a00c55656>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth<doxid-structsbls__control__type_1abf884043df0f9c0d95bcff6fae1bf9bb>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization<doxid-structsbls__control__type_1a108359f1209601e6c6074c215e3abd8b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_col<doxid-structsbls__control__type_1abca2db33b9520095e98790d45a1be93f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scaling<doxid-structsbls__control__type_1a26f0572eeeaa419eabb09dc89c00b89d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ordering<doxid-structsbls__control__type_1a4175ebe476addcfc3433fc97c19e0708>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol<doxid-structsbls__control__type_1a133347eb5f45a24a77b63b4afd4212e8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol_for_basis<doxid-structsbls__control__type_1a1912d9ec51c4e88125762b7d03ef31a6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_pivot<doxid-structsbls__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`static_tolerance<doxid-structsbls__control__type_1a3ce9c9cc8dd0f7c4684ea0bd80cc5946>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`static_level<doxid-structsbls__control__type_1ae7faebd3367553993434f6a03e65502d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`min_diagonal<doxid-structsbls__control__type_1a984528c49e15a61a1d30fc8fa2d166cc>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_absolute<doxid-structsbls__control__type_1a16e43fc1e4c1e1b4c671a9b1fbbcd3e6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_relative<doxid-structsbls__control__type_1ae3103abf29cabc33010d53428da2f2fc>`;
		bool :ref:`remove_dependencies<doxid-structsbls__control__type_1ae17a6b550239434c639239ddf45bc1ad>`;
		bool :ref:`find_basis_by_transpose<doxid-structsbls__control__type_1aa88001d7f86370d329247cf28f8ff499>`;
		bool :ref:`affine<doxid-structsbls__control__type_1ad036b5eb236ab3725436ece9cbf93e57>`;
		bool :ref:`allow_singular<doxid-structsbls__control__type_1a2ea4ee3b5051d51642fae65d6ba75e7d>`;
		bool :ref:`perturb_to_make_definite<doxid-structsbls__control__type_1aa811deca1c703ca2ab1c43fefffa6fbd>`;
		bool :ref:`get_norm_residual<doxid-structsbls__control__type_1acdcd8a05219b5c554c279137cb409a03>`;
		bool :ref:`check_basis<doxid-structsbls__control__type_1afc637bcbceea5647ce8f349e109cb979>`;
		bool :ref:`space_critical<doxid-structsbls__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structsbls__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`symmetric_linear_solver<doxid-structsbls__control__type_1af297ace351b9307640715643cde57384>`[31];
		char :ref:`definite_linear_solver<doxid-structsbls__control__type_1a9b46b7a8e0af020499e645bef711f634>`[31];
		char :ref:`unsymmetric_linear_solver<doxid-structsbls__control__type_1aef6da6b715a0f41983c2a62397104eec>`[31];
		char :ref:`prefix<doxid-structsbls__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structsbls__control__type_1a31b308b91955ee385daacc3de00f161b>`;
		struct :ref:`uls_control_type<doxid-structuls__control__type>` :ref:`uls_control<doxid-structsbls__control__type_1ac6782df4602dd9c04417e2554d72bb00>`;
	};
.. _details-structsbls__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsbls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsbls__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structsbls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structsbls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; indmin
.. _doxid-structsbls__control__type_1a5031bbc31f94e4cba6a540a3182b6d80:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` indmin

initial estimate of integer workspace for SLS (obsolete)

.. index:: pair: variable; valmin
.. _doxid-structsbls__control__type_1a0e142fa8dc9c363c3c2993b6129b0955:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` valmin

initial estimate of real workspace for SLS (obsolete)

.. index:: pair: variable; len_ulsmin
.. _doxid-structsbls__control__type_1a600c95211b782597cd1b2475bb2c54c6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` len_ulsmin

initial estimate of workspace for ULS (obsolete)

.. index:: pair: variable; itref_max
.. _doxid-structsbls__control__type_1a903ba4ef0869186a65d4c32459a6a0ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itref_max

maximum number of iterative refinements with preconditioner allowed

.. index:: pair: variable; maxit_pcg
.. _doxid-structsbls__control__type_1aac98c23514fe50d29412bb0ecfacc8f2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit_pcg

maximum number of projected CG iterations allowed

.. index:: pair: variable; new_a
.. _doxid-structsbls__control__type_1a7bea45d51fd9384037bbbf82f7750ce6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_a

how much has $A$ changed since last factorization: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; new_h
.. _doxid-structsbls__control__type_1ae60c5b5b987dd62f25253ba4164813f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_h

how much has $H$ changed since last factorization: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; new_c
.. _doxid-structsbls__control__type_1a9eed9b4ef920c669df5ff1a7c67e3047:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_c

how much has $C$ changed since last factorization: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; preconditioner
.. _doxid-structsbls__control__type_1adf7719f1a4491459e361e80a00c55656:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` preconditioner

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
.. _doxid-structsbls__control__type_1abf884043df0f9c0d95bcff6fae1bf9bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth

the semi-bandwidth for band(H)

.. index:: pair: variable; factorization
.. _doxid-structsbls__control__type_1a108359f1209601e6c6074c215e3abd8b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization

the explicit factorization used:

* 0 selected automatically

* 1 Schur-complement if $G$ is diagonal and successful otherwise augmented system

* 2 augmented system

* 3 null-space

* 4 Schur-complement if $G$ is diagonal and successful otherwise failure

* 5 Schur-complement with pivoting if $G$ is diagonal and successful otherwise failure

.. index:: pair: variable; max_col
.. _doxid-structsbls__control__type_1abca2db33b9520095e98790d45a1be93f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_col

maximum number of nonzeros in a column of $A$ for Schur-complement factorization

.. index:: pair: variable; scaling
.. _doxid-structsbls__control__type_1a26f0572eeeaa419eabb09dc89c00b89d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scaling

not used at present

.. index:: pair: variable; ordering
.. _doxid-structsbls__control__type_1a4175ebe476addcfc3433fc97c19e0708:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ordering

see scaling

.. index:: pair: variable; pivot_tol
.. _doxid-structsbls__control__type_1a133347eb5f45a24a77b63b4afd4212e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol

the relative pivot tolerance used by ULS (obsolete)

.. index:: pair: variable; pivot_tol_for_basis
.. _doxid-structsbls__control__type_1a1912d9ec51c4e88125762b7d03ef31a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol_for_basis

the relative pivot tolerance used by ULS when determining the basis matrix

.. index:: pair: variable; zero_pivot
.. _doxid-structsbls__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_pivot

the absolute pivot tolerance used by ULS (obsolete)

.. index:: pair: variable; static_tolerance
.. _doxid-structsbls__control__type_1a3ce9c9cc8dd0f7c4684ea0bd80cc5946:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` static_tolerance

not used at present

.. index:: pair: variable; static_level
.. _doxid-structsbls__control__type_1ae7faebd3367553993434f6a03e65502d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` static_level

see static_tolerance

.. index:: pair: variable; min_diagonal
.. _doxid-structsbls__control__type_1a984528c49e15a61a1d30fc8fa2d166cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` min_diagonal

the minimum permitted diagonal in diag(max($H$,min_diag))

.. index:: pair: variable; stop_absolute
.. _doxid-structsbls__control__type_1a16e43fc1e4c1e1b4c671a9b1fbbcd3e6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_absolute

the required absolute and relative accuracies

.. index:: pair: variable; stop_relative
.. _doxid-structsbls__control__type_1ae3103abf29cabc33010d53428da2f2fc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_relative

see stop_absolute

.. index:: pair: variable; remove_dependencies
.. _doxid-structsbls__control__type_1ae17a6b550239434c639239ddf45bc1ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool remove_dependencies

preprocess equality constraints to remove linear dependencies

.. index:: pair: variable; find_basis_by_transpose
.. _doxid-structsbls__control__type_1aa88001d7f86370d329247cf28f8ff499:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool find_basis_by_transpose

determine implicit factorization preconditioners using a basis of A found by examining A's transpose

.. index:: pair: variable; affine
.. _doxid-structsbls__control__type_1ad036b5eb236ab3725436ece9cbf93e57:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool affine

can the right-hand side $c$ be assumed to be zero?

.. index:: pair: variable; allow_singular
.. _doxid-structsbls__control__type_1a2ea4ee3b5051d51642fae65d6ba75e7d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool allow_singular

do we tolerate "singular" preconditioners?

.. index:: pair: variable; perturb_to_make_definite
.. _doxid-structsbls__control__type_1aa811deca1c703ca2ab1c43fefffa6fbd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool perturb_to_make_definite

if the initial attempt at finding a preconditioner is unsuccessful, should the diagonal be perturbed so that a second attempt succeeds?

.. index:: pair: variable; get_norm_residual
.. _doxid-structsbls__control__type_1acdcd8a05219b5c554c279137cb409a03:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool get_norm_residual

compute the residual when applying the preconditioner?

.. index:: pair: variable; check_basis
.. _doxid-structsbls__control__type_1afc637bcbceea5647ce8f349e109cb979:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool check_basis

if an implicit or null-space preconditioner is used, assess and correct for ill conditioned basis matrices

.. index:: pair: variable; space_critical
.. _doxid-structsbls__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structsbls__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structsbls__control__type_1af297ace351b9307640715643cde57384:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

the name of the symmetric-indefinite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', and 'sytr', although only 'sytr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; definite_linear_solver
.. _doxid-structsbls__control__type_1a9b46b7a8e0af020499e645bef711f634:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

the name of the definite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', 'potr', 'sytr' and 'pbtr', although only 'potr',  'sytr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; unsymmetric_linear_solver
.. _doxid-structsbls__control__type_1aef6da6b715a0f41983c2a62397104eec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char unsymmetric_linear_solver[31]

the name of the unsymmetric linear equation solver used. Possible choices are currently: 'gls', 'ma48' and 'getr', although only 'getr' is installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_uls<details-uls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structsbls__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structsbls__control__type_1a31b308b91955ee385daacc3de00f161b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; uls_control
.. _doxid-structsbls__control__type_1ac6782df4602dd9c04417e2554d72bb00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`uls_control_type<doxid-structuls__control__type>` uls_control

control parameters for ULS

