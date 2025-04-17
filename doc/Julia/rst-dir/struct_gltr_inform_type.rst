.. index:: pair: table; gltr_inform_type
.. _doxid-structgltr__inform__type:

gltr_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct gltr_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          iter_pass2::INT
          obj::T
          multiplier::T
          mnormx::T
          piv::T
          curv::T
          rayleigh::T
          leftmost::T
          negative_curvature::Bool
          hard_case::Bool

.. _details-structgltr__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structgltr__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See :ref:`gltr_solve_problem <doxid-galahad__gltr_8h_1ad77040d245e6bc307d13ea0cec355f18>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structgltr__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structgltr__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structgltr__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structgltr__inform__type_iter_pass2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter_pass2

the total number of pass-2 iterations required if the solution lies on the trust-region boundary

.. index:: pair: variable; obj
.. _doxid-structgltr__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the quadratic function

.. index:: pair: variable; multiplier
.. _doxid-structgltr__inform__type_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the Lagrange multiplier corresponding to the trust-region constraint

.. index:: pair: variable; mnormx
.. _doxid-structgltr__inform__type_mnormx:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mnormx

the $M$ -norm of $x$

.. index:: pair: variable; piv
.. _doxid-structgltr__inform__type_piv:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T piv

the latest pivot in the Cholesky factorization of the Lanczos tridiagona

.. index:: pair: variable; curv
.. _doxid-structgltr__inform__type_curv:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T curv

the most negative cuurvature encountered

.. index:: pair: variable; rayleigh
.. _doxid-structgltr__inform__type_rayleigh:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T rayleigh

the current Rayleigh quotient

.. index:: pair: variable; leftmost
.. _doxid-structgltr__inform__type_leftmost:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T leftmost

an estimate of the leftmost generalized eigenvalue of the pencil $(H,M)$

.. index:: pair: variable; negative_curvature
.. _doxid-structgltr__inform__type_negative_curvature:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool negative_curvature

was negative curvature encountered ?

.. index:: pair: variable; hard_case
.. _doxid-structgltr__inform__type_hard_case:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hard_case

did the hard case occur ?

