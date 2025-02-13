.. index:: pair: table; fdc_inform_type
.. _doxid-structfdc__inform__type:

fdc_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct fdc_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          factorization_status::INT
          factorization_integer::Int64
          factorization_real::Int64
          non_negligible_pivot::T
          time::fdc_time_type{T}
          sls_inform::sls_inform_type{T,INT}
          uls_inform::uls_inform_type{T,INT}
	
.. _details-structfdc__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structfdc__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See FDC_find_dependent for details

.. index:: pair: variable; alloc_status
.. _doxid-structfdc__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structfdc__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; factorization_status
.. _doxid-structfdc__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structfdc__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structfdc__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structfdc__inform__type_1a827ddb7fead8e375404c9b770b67e771:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; time
.. _doxid-structfdc__inform__type_1af77e8a375712c74c7ec4cb4d6ee89826:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_time_type<doxid-structfdc__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structfdc__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

SLS inform type.

.. index:: pair: variable; uls_inform
.. _doxid-structfdc__inform__type_1aa39eb0d7b50d4a858849f8ef652ae84c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`uls_inform_type<doxid-structuls__inform__type>` uls_inform

ULS inform type.

