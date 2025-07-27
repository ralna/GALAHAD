.. index:: pair: table; ssls_inform_type
.. _doxid-structssls__inform__type:

ssls_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ssls_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          factorization_integer::Int64
          factorization_real::Int64
          rank::INT
          rank_def::Bool
          time::ssls_time_type{T}
          sls_inform::sls_inform_type{T,INT}

.. _details-structssls__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structssls__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See SSLS_form_and_factorize for details

.. index:: pair: variable; alloc_status
.. _doxid-structssls__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structssls__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

.. index:: pair: variable; factorization_integer
.. _doxid-structssls__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structssls__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; rank
.. _doxid-structssls__inform__type_rank:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT rank

the computed rank of $K$

.. index:: pair: variable; rank_def
.. _doxid-structssls__inform__type_rank_def:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool rank_def

is the matrix $K$ rank defficient?

.. index:: pair: variable; time
.. _doxid-structssls__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ssls_time_type<doxid-structssls__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structssls__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters from the GALAHAD package SLS used
