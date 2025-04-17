.. index:: pair: struct; cro_inform_type
.. _doxid-structcro__inform__type:

cro_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct cro_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          dependent::INT
          time::cro_time_type{T}
          sls_inform::sls_inform_type{T,INT}
          sbls_inform::sbls_inform_type{T,INT}
          uls_inform::uls_inform_type{T,INT}
          scu_status::INT
          scu_inform::scu_inform_type
          ir_inform::ir_inform_type{T,INT}

.. _details-structcro__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structcro__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See CRO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structcro__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structcro__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; dependent
.. _doxid-structcro__inform__type_dependent:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT dependent

the number of dependent active constraints

.. index:: pair: variable; time
.. _doxid-structcro__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`cro_time_type<doxid-structcro__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structcro__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from SLS

.. index:: pair: variable; sbls_inform
.. _doxid-structcro__inform__type_sbls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

information from SBLS

.. index:: pair: variable; uls_inform
.. _doxid-structcro__inform__type_uls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`uls_inform_type<doxid-structuls__inform__type>` uls_inform

information from ULS

.. index:: pair: variable; scu_status
.. _doxid-structcro__inform__type_scu_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT scu_status

information from SCU

.. index:: pair: variable; scu_inform
.. _doxid-structcro__inform__type_scu_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`scu_inform_type<doxid-structscu__inform__type>` scu_inform

see scu_status

.. index:: pair: variable; ir_inform
.. _doxid-structcro__inform__type_ir_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

information from IR

