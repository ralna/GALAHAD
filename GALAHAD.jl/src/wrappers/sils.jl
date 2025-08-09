export sils_control_type

struct sils_control_type{T,INT}
  f_indexing::Bool
  ICNTL::NTuple{30,INT}
  lp::INT
  wp::INT
  mp::INT
  sp::INT
  ldiag::INT
  la::INT
  liw::INT
  maxla::INT
  maxliw::INT
  pivoting::INT
  nemin::INT
  factorblocking::INT
  solveblocking::INT
  thresh::INT
  ordering::INT
  scaling::INT
  CNTL::NTuple{5,T}
  multiplier::T
  reduce::T
  u::T
  static_tolerance::T
  static_level::T
  tolerance::T
  convergence::T
end

export sils_ainfo_type

struct sils_ainfo_type{T,INT}
  flag::INT
  more::INT
  nsteps::INT
  nrltot::INT
  nirtot::INT
  nrlnec::INT
  nirnec::INT
  nrladu::INT
  niradu::INT
  ncmpa::INT
  oor::INT
  dup::INT
  maxfrt::INT
  stat::INT
  faulty::INT
  opsa::T
  opse::T
end

export sils_finfo_type

struct sils_finfo_type{T,INT}
  flag::INT
  more::INT
  maxfrt::INT
  nebdu::INT
  nrlbdu::INT
  nirbdu::INT
  nrltot::INT
  nirtot::INT
  nrlnec::INT
  nirnec::INT
  ncmpbr::INT
  ncmpbi::INT
  ntwo::INT
  neig::INT
  delay::INT
  signc::INT
  nstatic::INT
  modstep::INT
  rank::INT
  stat::INT
  faulty::INT
  step::INT
  opsa::T
  opse::T
  opsb::T
  maxchange::T
  smin::T
  smax::T
end

export sils_sinfo_type

struct sils_sinfo_type{T,INT}
  flag::INT
  stat::INT
  cond::T
  cond2::T
  berr::T
  berr2::T
  error::T
end

export sils_initialize

function sils_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.sils_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sils_control_type{Float32,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function sils_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.sils_initialize_s_64(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{sils_control_type{Float32,
                                                                                  Int64}},
                                                   status::Ptr{Int64})::Cvoid
end

function sils_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.sils_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sils_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sils_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.sils_initialize_64(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{sils_control_type{Float64,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function sils_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.sils_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{sils_control_type{Float128,
                                                                               Int32}},
                                                status::Ptr{Int32})::Cvoid
end

function sils_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.sils_initialize_q_64(data::Ptr{Ptr{Cvoid}},
                                                      control::Ptr{sils_control_type{Float128,
                                                                                     Int64}},
                                                      status::Ptr{Int64})::Cvoid
end

export sils_read_specfile

function sils_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.sils_read_specfile_s(control::Ptr{sils_control_type{Float32,
                                                                               Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function sils_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.sils_read_specfile_s_64(control::Ptr{sils_control_type{Float32,
                                                                                     Int64}},
                                                      specfile::Ptr{Cchar})::Cvoid
end

function sils_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.sils_read_specfile(control::Ptr{sils_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function sils_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.sils_read_specfile_64(control::Ptr{sils_control_type{Float64,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

function sils_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.sils_read_specfile_q(control::Ptr{sils_control_type{Float128,
                                                                                  Int32}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function sils_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.sils_read_specfile_q_64(control::Ptr{sils_control_type{Float128,
                                                                                        Int64}},
                                                         specfile::Ptr{Cchar})::Cvoid
end

export sils_import

function sils_import(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.sils_import_s(control::Ptr{sils_control_type{Float32,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32})::Cvoid
end

function sils_import(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.sils_import_s_64(control::Ptr{sils_control_type{Float32,
                                                                              Int64}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int64})::Cvoid
end

function sils_import(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.sils_import(control::Ptr{sils_control_type{Float64,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32})::Cvoid
end

function sils_import(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.sils_import_64(control::Ptr{sils_control_type{Float64,Int64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int64})::Cvoid
end

function sils_import(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.sils_import_q(control::Ptr{sils_control_type{Float128,Int32}},
                                            data::Ptr{Ptr{Cvoid}},
                                            status::Ptr{Int32})::Cvoid
end

function sils_import(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.sils_import_q_64(control::Ptr{sils_control_type{Float128,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int64})::Cvoid
end

export sils_reset_control

function sils_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.sils_reset_control_s(control::Ptr{sils_control_type{Float32,
                                                                               Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function sils_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.sils_reset_control_s_64(control::Ptr{sils_control_type{Float32,
                                                                                     Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64})::Cvoid
end

function sils_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.sils_reset_control(control::Ptr{sils_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function sils_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.sils_reset_control_64(control::Ptr{sils_control_type{Float64,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

function sils_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.sils_reset_control_q(control::Ptr{sils_control_type{Float128,
                                                                                  Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32})::Cvoid
end

function sils_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.sils_reset_control_q_64(control::Ptr{sils_control_type{Float128,
                                                                                        Int64}},
                                                         data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64})::Cvoid
end

export sils_information

function sils_information(::Type{Float32}, ::Type{Int32}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_single.sils_information_s(data::Ptr{Ptr{Cvoid}},
                                              ainfo::Ptr{sils_ainfo_type{Float32,Int32}},
                                              finfo::Ptr{sils_finfo_type{Float32,Int32}},
                                              sinfo::Ptr{sils_sinfo_type{Float32,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function sils_information(::Type{Float32}, ::Type{Int64}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_single_64.sils_information_s_64(data::Ptr{Ptr{Cvoid}},
                                                    ainfo::Ptr{sils_ainfo_type{Float32,
                                                                               Int64}},
                                                    finfo::Ptr{sils_finfo_type{Float32,
                                                                               Int64}},
                                                    sinfo::Ptr{sils_sinfo_type{Float32,
                                                                               Int64}},
                                                    status::Ptr{Int64})::Cvoid
end

function sils_information(::Type{Float64}, ::Type{Int32}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_double.sils_information(data::Ptr{Ptr{Cvoid}},
                                            ainfo::Ptr{sils_ainfo_type{Float64,Int32}},
                                            finfo::Ptr{sils_finfo_type{Float64,Int32}},
                                            sinfo::Ptr{sils_sinfo_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function sils_information(::Type{Float64}, ::Type{Int64}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_double_64.sils_information_64(data::Ptr{Ptr{Cvoid}},
                                                  ainfo::Ptr{sils_ainfo_type{Float64,Int64}},
                                                  finfo::Ptr{sils_finfo_type{Float64,Int64}},
                                                  sinfo::Ptr{sils_sinfo_type{Float64,Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

function sils_information(::Type{Float128}, ::Type{Int32}, data, ainfo, finfo, sinfo,
                          status)
  @ccall libgalahad_quadruple.sils_information_q(data::Ptr{Ptr{Cvoid}},
                                                 ainfo::Ptr{sils_ainfo_type{Float128,Int32}},
                                                 finfo::Ptr{sils_finfo_type{Float128,Int32}},
                                                 sinfo::Ptr{sils_sinfo_type{Float128,Int32}},
                                                 status::Ptr{Int32})::Cvoid
end

function sils_information(::Type{Float128}, ::Type{Int64}, data, ainfo, finfo, sinfo,
                          status)
  @ccall libgalahad_quadruple_64.sils_information_q_64(data::Ptr{Ptr{Cvoid}},
                                                       ainfo::Ptr{sils_ainfo_type{Float128,
                                                                                  Int64}},
                                                       finfo::Ptr{sils_finfo_type{Float128,
                                                                                  Int64}},
                                                       sinfo::Ptr{sils_sinfo_type{Float128,
                                                                                  Int64}},
                                                       status::Ptr{Int64})::Cvoid
end

export sils_finalize

function sils_finalize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.sils_finalize_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sils_control_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sils_finalize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.sils_finalize_s_64(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{sils_control_type{Float32,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function sils_finalize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.sils_finalize(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sils_control_type{Float64,Int32}},
                                         status::Ptr{Int32})::Cvoid
end

function sils_finalize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.sils_finalize_64(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{sils_control_type{Float64,
                                                                              Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function sils_finalize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.sils_finalize_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{sils_control_type{Float128,
                                                                             Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function sils_finalize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.sils_finalize_q_64(data::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{sils_control_type{Float128,
                                                                                   Int64}},
                                                    status::Ptr{Int64})::Cvoid
end

function run_sif(::Val{:sils}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runsils_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:sils}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runsils_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
