.. index:: pair: table; glrt_inform_type
.. _doxid-structglrt__inform__type:

glrt_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block
	
        struct glrt_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          iter_pass2::INT
          obj::T
          obj_regularized::T
          multiplier::T
          xpo_norm::T
          leftmost::T
          negative_curvature::Bool
          hard_case::Bool

.. _details-structglrt__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structglrt__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See :ref:`glrt_solve_problem <doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structglrt__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structglrt__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structglrt__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structglrt__inform__type_iter_pass2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter_pass2

the total number of pass-2 iterations required

.. index:: pair: variable; obj
.. _doxid-structglrt__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the quadratic function

.. index:: pair: variable; obj_regularized
.. _doxid-structglrt__inform__type_obj_regularized:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_regularized

the value of the regularized quadratic function

.. index:: pair: variable; multiplier
.. _doxid-structglrt__inform__type_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the multiplier, $\sigma \|x\|^{p-2}$

.. index:: pair: variable; xpo_norm
.. _doxid-structglrt__inform__type_xpo_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T xpo_norm

the value of the norm $\|x\|_M$

.. index:: pair: variable; leftmost
.. _doxid-structglrt__inform__type_leftmost:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T leftmost

an estimate of the leftmost generalized eigenvalue of the pencil $(H,M)$

.. index:: pair: variable; negative_curvature
.. _doxid-structglrt__inform__type_negative_curvature:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool negative_curvature

was negative curvature encountered ?

.. index:: pair: variable; hard_case
.. _doxid-structglrt__inform__type_hard_case:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hard_case

did the hard case occur ?

