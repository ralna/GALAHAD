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

export nodend_initialize

function nodend_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{nodend_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function nodend_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{nodend_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
end

function nodend_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{nodend_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function nodend_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{nodend_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
end

function nodend_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{nodend_control_type{Int32}},
                                                status::Ptr{Int32})::Cvoid
end

function nodend_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{nodend_control_type{Int64}},
                                                   status::Ptr{Int64})::Cvoid
end

export nodend_read_specfile

function nodend_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.nodend_read_specfile(control::Ptr{nodend_control_type{Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function nodend_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.nodend_read_specfile(control::Ptr{nodend_control_type{Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function nodend_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.nodend_read_specfile(control::Ptr{nodend_control_type{Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function nodend_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.nodend_read_specfile(control::Ptr{nodend_control_type{Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function nodend_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.nodend_read_specfile(control::Ptr{nodend_control_type{Int32}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function nodend_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.nodend_read_specfile(control::Ptr{nodend_control_type{Int64}},
                                                      specfile::Ptr{Cchar})::Cvoid
end

export nodend_information

function nodend_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.nodend_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{nodend_inform_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function nodend_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.nodend_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{nodend_inform_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function nodend_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.nodend_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{nodend_inform_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function nodend_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.nodend_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{nodend_inform_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function nodend_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.nodend_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{nodend_inform_type{Int32}},
                                                 status::Ptr{Int32})::Cvoid
end

function nodend_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.nodend_information(data::Ptr{Ptr{Cvoid}},
                                                    inform::Ptr{nodend_inform_type{Int64}},
                                                    status::Ptr{Int64})::Cvoid
end
