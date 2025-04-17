.. index:: pair: table; l2rt_inform_type
.. _doxid-structl2rt__inform__type:

l2rt_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct l2rt_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          iter_pass2::INT
          biters::INT
          biter_min::INT
          biter_max::INT
          obj::T
          multiplier::T
          x_norm::T
          r_norm::T
          Atr_norm::T
          biter_mean::T

.. _details-structl2rt__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structl2rt__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See :ref:`l2rt_solve_problem <doxid-galahad__l2rt_8h_1a53042b19cef3a62c34631b00111ce754>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structl2rt__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structl2rt__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structl2rt__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structl2rt__inform__type_iter_pass2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter_pass2

the total number of pass-2 iterations required

.. index:: pair: variable; biters
.. _doxid-structl2rt__inform__type_biters:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT biters

the total number of inner iterations performed

.. index:: pair: variable; biter_min
.. _doxid-structl2rt__inform__type_biter_min:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT biter_min

the smallest number of inner iterations performed during an outer iteration

.. index:: pair: variable; biter_max
.. _doxid-structl2rt__inform__type_biter_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT biter_max

the largest number of inner iterations performed during an outer iteration

.. index:: pair: variable; obj
.. _doxid-structl2rt__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function

.. index:: pair: variable; multiplier
.. _doxid-structl2rt__inform__type_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the multiplier, $\lambda = \mu + \sigma \|x\|^{p-2} * \sqrt{\|Ax-b\|^2 + \mu \|x\|^2}$

.. index:: pair: variable; x_norm
.. _doxid-structl2rt__inform__type_x_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the Euclidean norm of $x$

.. index:: pair: variable; r_norm
.. _doxid-structl2rt__inform__type_r_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T r_norm

the Euclidean norm of $Ax-b$

.. index:: pair: variable; Atr_norm
.. _doxid-structl2rt__inform__type_Atr_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T Atr_norm

the Euclidean norm of $A^T (Ax-b) + \lambda x$

.. index:: pair: variable; biter_mean
.. _doxid-structl2rt__inform__type_biter_mean:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T biter_mean

the average number of inner iterations performed during an outer iteration

