.. index:: pair: struct; slls_inform_type
.. _doxid-structslls__inform__type:

slls_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct slls_inform_type{T,INT}
          status::INT
          alloc_status::INT
          factorization_status::INT
          iter::INT
          cg_iter::INT
          obj::T
          norm_pg::T
          bad_alloc::NTuple{81,Cchar}
          time::slls_time_type{T}
          sbls_inform::sbls_inform_type{T,INT}
          convert_inform::convert_inform_type{T,INT}
	
.. _details-structslls__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structslls__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

reported return status.

.. index:: pair: variable; alloc_status
.. _doxid-structslls__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

Fortran STAT value after allocate failure.

.. index:: pair: variable; factorization_status
.. _doxid-structslls__inform__type_factorization_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

status return from factorization

.. index:: pair: variable; iter
.. _doxid-structslls__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structslls__inform__type_cg_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

number of CG iterations required

.. index:: pair: variable; obj
.. _doxid-structslls__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

current value of the objective function, r(x).

.. index:: pair: variable; norm_pg
.. _doxid-structslls__inform__type_norm_pg:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_pg

current value of the Euclidean norm of projected gradient of r(x).

.. index:: pair: variable; bad_alloc
.. _doxid-structslls__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structslls__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`slls_time_type<doxid-structslls__time__type>` time

times for various stages

.. index:: pair: variable; sbls_inform
.. _doxid-structslls__inform__type_sbls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform values from SBLS

.. index:: pair: variable; convert_inform
.. _doxid-structslls__inform__type_convert_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`convert_inform_type<doxid-structconvert__inform__type>` convert_inform

inform values for CONVERT

