.. index:: pair: struct; lpa_inform_type
.. _doxid-structlpa__inform__type:

lpa_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lpa_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          la04_job::INT
          la04_job_info::INT
          obj::T
          primal_infeasibility::T
          feasible::Bool
          RINFO::NTuple{40,T}
          time::lpa_time_type{T}
          rpd_inform::rpd_inform_type{INT}
	
.. _details-structlpa__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlpa__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See LPA_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structlpa__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlpa__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlpa__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; la04_job
.. _doxid-structlpa__inform__type_la04_job:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT la04_job

the final value of la04's job argument

.. index:: pair: variable; la04_job_info
.. _doxid-structlpa__inform__type_la04_job_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT la04_job_info

any extra information from an unsuccessfull call to LA04 (LA04's RINFO(35)

.. index:: pair: variable; obj
.. _doxid-structlpa__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by LPA_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structlpa__inform__type_primal_infeasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T primal_infeasibility

the value of the primal infeasibility

.. index:: pair: variable; feasible
.. _doxid-structlpa__inform__type_feasible:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; RINFO
.. _doxid-structlpa__inform__type_RINFO:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T RINFO[40]

the information array from LA04

.. index:: pair: variable; time
.. _doxid-structlpa__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lpa_time_type<doxid-structlpa__time__type>` time

timings (see above)

.. index:: pair: variable; rpd_inform
.. _doxid-structlpa__inform__type_rpd_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rpd_inform_type<doxid-structrpd__inform__type>` rpd_inform

inform parameters for RPD

