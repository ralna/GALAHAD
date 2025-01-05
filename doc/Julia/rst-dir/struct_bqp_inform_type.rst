.. index:: pair: struct; bqp_inform_type
.. _doxid-structbqp__inform__type:

bqp_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct bqp_inform_type{T,INT}
          status::INT
          alloc_status::INT
          factorization_status::INT
          iter::INT
          cg_iter::INT
          obj::T
          norm_pg::T
          bad_alloc::NTuple{81,Cchar}
          time::bqp_time_type
          sbls_inform::sbls_inform_type{T,INT}
	
.. _details-structbqp__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structbqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

reported return status:

* **0**

  success

* **-1**

  allocation error

* **-2**

  deallocation error

* **-3**

  matrix data faulty (.n < 1, .ne < 0)

* **-20**

  alegedly +ve definite matrix is not

.. index:: pair: variable; alloc_status
.. _doxid-structbqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

Fortran STAT value after allocate failure.

.. index:: pair: variable; factorization_status
.. _doxid-structbqp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

status return from factorization

.. index:: pair: variable; iter
.. _doxid-structbqp__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structbqp__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

number of CG iterations required

.. index:: pair: variable; obj
.. _doxid-structbqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

current value of the objective function

.. index:: pair: variable; norm_pg
.. _doxid-structbqp__inform__type_1acb02a4d1ae275a55874bb9897262b1fe:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_pg

current value of the projected gradient

.. index:: pair: variable; bad_alloc
.. _doxid-structbqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structbqp__inform__type_1a7f44be002389597b3f6c06e9a9b6eefa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`bqp_time_type<doxid-structbqp__time__type>` time

times for various stages

.. index:: pair: variable; sbls_inform
.. _doxid-structbqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform values from SBLS

