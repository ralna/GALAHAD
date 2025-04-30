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

export nodend_time_type

struct nodend_time_type{T}
  total::T
  metis::T
  clock_total::T
  clock_metis::T
end

export nodend_inform_type

struct nodend_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  version::NTuple{4,Cchar}
  time::nodend_time_type{T}
end

export nodend_initialize

function nodend_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{nodend_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nodend_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{nodend_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nodend_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{nodend_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nodend_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{nodend_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nodend_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{nodend_control_type{Int32}},
                                                status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nodend_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.nodend_initialize(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{nodend_control_type{Int64}},
                                                   status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
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

export nodend_order

function nodend_order(::Type{Float32}, ::Type{Int32}, control, data, status, n, perm,
                      A_type, ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.nodend_order(control::Ptr{nodend_control_type{Int32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        perm::Ptr{Int32}, A_type::Ptr{Cchar}, ne::Int32,
                                        A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                        A_ptr::Ptr{Int32})::Cvoid
end

function nodend_order(::Type{Float32}, ::Type{Int64}, control, data, status, n, perm,
                      A_type, ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single_64.nodend_order(control::Ptr{nodend_control_type{Int64}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, perm::Ptr{Int64}, A_type::Ptr{Cchar},
                                           ne::Int64, A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                           A_ptr::Ptr{Int64})::Cvoid
end

function nodend_order(::Type{Float64}, ::Type{Int32}, control, data, status, n, perm,
                      A_type, ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.nodend_order(control::Ptr{nodend_control_type{Int32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        perm::Ptr{Int32}, A_type::Ptr{Cchar}, ne::Int32,
                                        A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                        A_ptr::Ptr{Int32})::Cvoid
end

function nodend_order(::Type{Float64}, ::Type{Int64}, control, data, status, n, perm,
                      A_type, ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double_64.nodend_order(control::Ptr{nodend_control_type{Int64}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, perm::Ptr{Int64}, A_type::Ptr{Cchar},
                                           ne::Int64, A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                           A_ptr::Ptr{Int64})::Cvoid
end

function nodend_order(::Type{Float128}, ::Type{Int32}, control, data, status, n, perm,
                      A_type, ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple.nodend_order(control::Ptr{nodend_control_type{Int32}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, perm::Ptr{Int32}, A_type::Ptr{Cchar},
                                           ne::Int32, A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                           A_ptr::Ptr{Int32})::Cvoid
end

function nodend_order(::Type{Float128}, ::Type{Int64}, control, data, status, n, perm,
                      A_type, ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple_64.nodend_order(control::Ptr{nodend_control_type{Int64}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, perm::Ptr{Int64},
                                              A_type::Ptr{Cchar}, ne::Int64,
                                              A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                              A_ptr::Ptr{Int64})::Cvoid
end

export nodend_information

function nodend_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.nodend_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{nodend_inform_type{Float32,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function nodend_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.nodend_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{nodend_inform_type{Float32,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function nodend_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.nodend_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{nodend_inform_type{Float64,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function nodend_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.nodend_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{nodend_inform_type{Float64,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function nodend_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.nodend_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{nodend_inform_type{Float128,
                                                                                Int32}},
                                                 status::Ptr{Int32})::Cvoid
end

function nodend_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.nodend_information(data::Ptr{Ptr{Cvoid}},
                                                    inform::Ptr{nodend_inform_type{Float128,
                                                                                   Int64}},
                                                    status::Ptr{Int64})::Cvoid
end

export nodend_terminate

function nodend_terminate(::Type{Float32}, ::Type{Int32}, data)
  @ccall libgalahad_single.nodend_terminate(data::Ptr{Ptr{Cvoid}})::Cvoid
end

function nodend_terminate(::Type{Float32}, ::Type{Int64}, data)
  @ccall libgalahad_single_64.nodend_terminate(data::Ptr{Ptr{Cvoid}})::Cvoid
end

function nodend_terminate(::Type{Float64}, ::Type{Int32}, data)
  @ccall libgalahad_double.nodend_terminate(data::Ptr{Ptr{Cvoid}})::Cvoid
end

function nodend_terminate(::Type{Float64}, ::Type{Int64}, data)
  @ccall libgalahad_double_64.nodend_terminate(data::Ptr{Ptr{Cvoid}})::Cvoid
end

function nodend_terminate(::Type{Float128}, ::Type{Int32}, data)
  @ccall libgalahad_quadruple.nodend_terminate(data::Ptr{Ptr{Cvoid}})::Cvoid
end

function nodend_terminate(::Type{Float128}, ::Type{Int64}, data)
  @ccall libgalahad_quadruple_64.nodend_terminate(data::Ptr{Ptr{Cvoid}})::Cvoid
end

const runnodend_sif_single = joinpath(galahad_bindir, "runnodend_sif_single$(exeext)")

function run_sif(::Val{:nodend}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runnodend_sif_single $path_libsif $path_outsdif`)
end

const runnodend_sif_double = joinpath(galahad_bindir, "runnodend_sif_double$(exeext)")

function run_sif(::Val{:nodend}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runnodend_sif_double $path_libsif $path_outsdif`)
end

const runnodend_sif_quadruple = joinpath(galahad_bindir, "runnodend_sif_quadruple$(exeext)")

function run_sif(::Val{:nodend}, ::Val{:quadruple}, path_libsif::String,
                 path_outsdif::String)
  return run(`$runnodend_sif_quadruple $path_libsif $path_outsdif`)
end
