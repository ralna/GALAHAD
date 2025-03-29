.. index:: pair: table; nodend_control_type
.. _doxid-structnodend__control__type:

nodend_control_type structure
-----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct nodend_control_type{T,INT}
          f_indexing::Bool
          version::NTuple{31,Cchar}
          error::INT
          out::INT
          print_level::INT
          no_metis_4_use_5_instead::Bool
          prefix::NTuple{31,Cchar}
          metis4_ptype::INT
          metis4_ctype::INT
          metis4_itype::INT
          metis4_rtype::INT
          metis4_dbglvl::INT
          metis4_oflags::INT
          metis4_pfactor::INT
          metis4_nseps::INT
          metis5_ptype::INT
          metis5_objtype::INT
          metis5_ctype::INT
          metis5_iptype::INT
          metis5_rtype::INT
          metis5_dbglvl::INT
          metis5_niter::INT
          metis5_ncuts::INT
          metis5_seed::INT
          metis5_no2hop::INT
          metis5_minconn::INT
          metis5_contig::INT
          metis5_compress::INT
          metis5_ccorder::INT
          metis5_pfactor::INT
          metis5_nseps::INT
          metis5_ufactor::INT
          metis5_niparts::INT
          metis5_ondisk::INT
          metis5_dropedges::INT
          metis5_twohop::INT
          metis5_fast::INT

.. _details-structnodend__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structnodend__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; version
.. _doxid-structnodend__control__type_version:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char version[31]

specify the version of METIS to be used. Possible values are 4.0, 5.1 and 5.2

.. index:: pair: variable; error
.. _doxid-structnodend__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structnodend__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structnodend__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

.. index:: pair: variable; no_metis_4_use_5_instead
.. _doxid-structnodend__control__type_no_metis_4_use_5_instead:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool no_metis_4_use_5_instead

if .no_metis_4_use_5_instead is true, and METIS 4.0 is not availble, use Metis 5.2 instead

.. index:: pair: variable; prefix
.. _doxid-structnodend__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; metis4_ptype 
.. _doxid-structnodend__control__type_metis4_ptype:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis4_ptype

the partitioning method employed. 0 = multilevel recursive  bisectioning: 1 = multilevel k-way partitioning

.. index:: pair: variable; metis4_ctype 
.. _doxid-structnodend__control__type_metis4_ctype:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis4_ctype

the matching scheme to be used during coarsening: 1 = random matching, 2 = heavy-edge matching, 3 = sorted heavy-edge matching, and 4 = k-way sorted heavy-edge matching.

.. index:: pair: variable; metis4_itype 
.. _doxid-structnodend__control__type_metis4_itype:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis4_itype

the algorithm used during initial partitioning: 1 = edge-based region growing and 2 = node-based region growing.

.. index:: pair: variable; metis4_rtype 
.. _doxid-structnodend__control__type_metis4_rtype:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis4_rtype

the algorithm used for refinement: 1 = two-sided node Fiduccia-Mattheyses (FM) refinement, and 2 = one-sided node FM refinement.

.. index:: pair: variable; metis4_dbglvl 
.. _doxid-structnodend__control__type_metis4_dbglvl:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis4_dbglvl

the amount of progress/debugging information printed: 0 = nothing, 1 = timings, and $>$ 1 increasingly more.

.. index:: pair: variable; metis4_oflags 
.. _doxid-structnodend__control__type_metis4_oflags:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis4_oflags

select whether or not to compress the graph, and to order connected components separately: 0 = do neither, 1 = try to compress the graph, 2 = order each connected component separately, and 3 = do both.

.. index:: pair: variable; metis4_pfactor 
.. _doxid-structnodend__control__type_metis4_pfactor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis4_pfactor

the minimum degree of the vertices that will be ordered last. More specifically, any vertices with a degree greater than 0.1 metis4_pfactor times the average degree are removed from the graph, an ordering of the rest of the vertices is computed, and an overall ordering is computed by ordering the removed vertices at the end of the overall ordering. Any value smaller than 1 means that no vertices will be ordered last.

.. index:: pair: variable; metis4_nseps 
.. _doxid-structnodend__control__type_metis4_nseps:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis4_nseps

the number of different separators that the algorithm will compute at each level of nested dissection.

.. index:: pair: variable; metis5_ptype 
.. _doxid-structnodend__control__type_metis5_ptype:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_ptype

the partitioning method. The value 0 gives multilevel recursive bisectioning, while 1 corresponds to multilevel $k$-way partitioning.

.. index:: pair: variable; metis5_objtype 
.. _doxid-structnodend__control__type_metis5_objtype:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_objtype

the type of the objective. Currently the only and default value metis5_objtype = 2, specifies node-based nested dissection, and any invalid value will be replaced by this default.

.. index:: pair: variable; metis5_ctype 
.. _doxid-structnodend__control__type_metis5_ctype:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_ctype

the matching scheme to be used during coarsening: 0 = random matching,  and 1 = sorted heavy-edge matching.

.. index:: pair: variable; metis5_iptype 
.. _doxid-structnodend__control__type_metis5_iptype:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_iptype

the algorithm used during initial partitioning: 2 = derive separators from edge cuts, and 3 = grow bisections using a greedy node-based strategy.

.. index:: pair: variable; metis5_rtype 
.. _doxid-structnodend__control__type_metis5_rtype:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_rtype

the algorithm used for refinement: 2 = Two-sided node FM refinement,  and 3 = One-sided node FM refinement.

.. index:: pair: variable; metis5_dbglvl 
.. _doxid-structnodend__control__type_metis5_dbglvl:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_dbglvl

the amount of progress/debugging information printed: 0 = nothing, 1 = diagnostics, 2 = plus timings, and $>$ 2 plus more.

.. index:: pair: variable; metis5_niparts 
.. _doxid-structnodend__control__type_metis5_niparts:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_niparts

the number of initial partitions used by MeTiS 5.2. 

.. index:: pair: variable; metis5_niter 
.. _doxid-structnodend__control__type_metis5_niter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_niter

the number of iterations used by the refinement algorithm.

.. index:: pair: variable; metis5_ncuts 
.. _doxid-structnodend__control__type_metis5_ncuts:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_ncuts

the number of different partitionings that it will compute: -1 = not used.

.. index:: pair: variable; metis5_seed 
.. _doxid-structnodend__control__type_metis5_seed:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_seed

the seed for the random number generator.

.. index:: pair: variable; metis5_ondisk 
.. _doxid-structnodend__control__type_metis5_ondisk:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_ondisk

whether on-disk storage is used (0 = no, 1 = yes) by MeTiS 5.2.

.. index:: pair: variable; metis5_minconn 
.. _doxid-structnodend__control__type_metis5_minconn:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_minconn

specify that the partitioning routines should try to minimize the maximum degree of the subdomain graph: 0 = no, 1 = yes, and -1 = not used. 

.. index:: pair: variable; metis5_contig 
.. _doxid-structnodend__control__type_metis5_contig:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_contig

specify that the partitioning routines should try to produce partitions that are contiguous: 0 = no, 1 = yes, and -1 = not used.

.. index:: pair: variable; metis5_compress 
.. _doxid-structnodend__control__type_metis5_compress:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_compress

specify that the graph should be compressed by combining together vertices that have identical adjacency lists: 0 = no, and 1 = yes.

.. index:: pair: variable; metis5_ccorder 
.. _doxid-structnodend__control__type_metis5_ccorder:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_ccorder

specify if the connected components of the graph should first be identified and ordered separately: 0 = no, and 1 = yes.

.. index:: pair: variable; metis5_pfactor 
.. _doxid-structnodend__control__type_metis5_pfactor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_pfactor

the minimum degree of the vertices that will be ordered last. More specifically, any vertices with a degree greater than 0.1 metis4_pfactor times the average degree are removed from the graph, an ordering of the rest of the vertices is computed, and an overall ordering is computed by ordering the removed vertices at the end of the overall ordering.

.. index:: pair: variable; metis5_nseps 
.. _doxid-structnodend__control__type_metis5_nseps:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_nseps

the number of different separators that the algorithm will compute at each level of nested dissection.

.. index:: pair: variable; metis5_ufactor 
.. _doxid-structnodend__control__type_metis5_ufactor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_ufactor

the maximum allowed load imbalance (1 +metis5_ufactor)/1000 among the partitions.

.. index:: pair: variable; metis5_dropedges 
.. _doxid-structnodend__control__type_metis5_dropedges:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_dropedges

will edges be dropped (0 = no, 1 = yes) by MeTiS 5.2.

.. index:: pair: variable; metis5_no2hop 
.. _doxid-structnodend__control__type_metis5_no2hop:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_no2hop

specify that the coarsening will not perform any 2–hop matchings when the standard matching approach fails to sufficiently coarsen the graph: 0 = no, and 1 = yes.

.. index:: pair: variable; metis5_twohop 
.. _doxid-structnodend__control__type_metis5_twohop:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_twohop

reserved for future use but ignored at present.

.. index:: pair: variable; metis5_fast 
.. _doxid-structnodend__control__type_metis5_fast:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT metis5_fast

reserved for future use but ignored at present.
