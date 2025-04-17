.. index:: pair: table; dps_inform_type
.. _doxid-structdps__inform__type:

dps_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct dps_inform_type{T,INT}
          status::INT
          alloc_status::INT
          mod_1by1::INT
          mod_2by2::INT
          obj::T
          obj_regularized::T
          x_norm::T
          multiplier::T
          pole::T
          hard_case::Bool
          bad_alloc::NTuple{81,Cchar}
          time::dps_time_type{T}
          sls_inform::sls_inform_type{T,INT}
	
.. _details-structdps__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structdps__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See DPS_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structdps__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; mod_1by1
.. _doxid-structdps__inform__type_mod_1by1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mod_1by1

the number of 1 by 1 blocks from the factorization of H that were modified when constructing $M$

.. index:: pair: variable; mod_2by2
.. _doxid-structdps__inform__type_mod_2by2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mod_2by2

the number of 2 by 2 blocks from the factorization of H that were modified when constructing $M$

.. index:: pair: variable; obj
.. _doxid-structdps__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the quadratic function

.. index:: pair: variable; obj_regularized
.. _doxid-structdps__inform__type_obj_regularized:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_regularized

the value of the regularized quadratic function

.. index:: pair: variable; x_norm
.. _doxid-structdps__inform__type_x_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the M-norm of the solution

.. index:: pair: variable; multiplier
.. _doxid-structdps__inform__type_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the Lagrange multiplier associated with the constraint/regularization

.. index:: pair: variable; pole
.. _doxid-structdps__inform__type_pole:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pole

a lower bound max(0,-lambda_1), where lambda_1 is the left-most eigenvalue of $(H,M)$

.. index:: pair: variable; hard_case
.. _doxid-structdps__inform__type_hard_case:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hard_case

has the hard case occurred?

.. index:: pair: variable; bad_alloc
.. _doxid-structdps__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array that provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structdps__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`dps_time_type<doxid-structdps__time__type>` time

time information

.. index:: pair: variable; sls_inform
.. _doxid-structdps__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from SLS

