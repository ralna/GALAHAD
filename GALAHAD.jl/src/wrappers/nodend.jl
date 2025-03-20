export nodend_control_type

struct nodend_control_type{INT}
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
end

export nodend_inform_type

struct nodend_inform_type{INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  version::NTuple{4,Cchar}
end
