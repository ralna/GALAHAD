export sils_control_type

struct sils_control_type{T}
  f_indexing::Bool
  ICNTL::NTuple{30,Cint}
  lp::Cint
  wp::Cint
  mp::Cint
  sp::Cint
  ldiag::Cint
  la::Cint
  liw::Cint
  maxla::Cint
  maxliw::Cint
  pivoting::Cint
  nemin::Cint
  factorblocking::Cint
  solveblocking::Cint
  thresh::Cint
  ordering::Cint
  scaling::Cint
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

struct sils_ainfo_type{T}
  flag::Cint
  more::Cint
  nsteps::Cint
  nrltot::Cint
  nirtot::Cint
  nrlnec::Cint
  nirnec::Cint
  nrladu::Cint
  niradu::Cint
  ncmpa::Cint
  oor::Cint
  dup::Cint
  maxfrt::Cint
  stat::Cint
  faulty::Cint
  opsa::T
  opse::T
end

export sils_finfo_type

struct sils_finfo_type{T}
  flag::Cint
  more::Cint
  maxfrt::Cint
  nebdu::Cint
  nrlbdu::Cint
  nirbdu::Cint
  nrltot::Cint
  nirtot::Cint
  nrlnec::Cint
  nirnec::Cint
  ncmpbr::Cint
  ncmpbi::Cint
  ntwo::Cint
  neig::Cint
  delay::Cint
  signc::Cint
  nstatic::Cint
  modstep::Cint
  rank::Cint
  stat::Cint
  faulty::Cint
  step::Cint
  opsa::T
  opse::T
  opsb::T
  maxchange::T
  smin::T
  smax::T
end

export sils_sinfo_type

struct sils_sinfo_type{T}
  flag::Cint
  stat::Cint
  cond::T
  cond2::T
  berr::T
  berr2::T
  error::T
end

export sils_initialize

function sils_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.sils_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sils_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function sils_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.sils_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sils_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function sils_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.sils_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{sils_control_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export sils_read_specfile

function sils_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.sils_read_specfile_s(control::Ptr{sils_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function sils_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.sils_read_specfile(control::Ptr{sils_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function sils_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.sils_read_specfile_q(control::Ptr{sils_control_type{Float128}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export sils_import

function sils_import(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.sils_import_s(control::Ptr{sils_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint})::Cvoid
end

function sils_import(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.sils_import(control::Ptr{sils_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint})::Cvoid
end

function sils_import(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.sils_import_q(control::Ptr{sils_control_type{Float128}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint})::Cvoid
end

export sils_reset_control

function sils_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.sils_reset_control_s(control::Ptr{sils_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function sils_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.sils_reset_control(control::Ptr{sils_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

function sils_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.sils_reset_control_q(control::Ptr{sils_control_type{Float128}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Cint})::Cvoid
end

export sils_information

function sils_information(::Type{Float32}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_single.sils_information_s(data::Ptr{Ptr{Cvoid}},
                                              ainfo::Ptr{sils_ainfo_type{Float32}},
                                              finfo::Ptr{sils_finfo_type{Float32}},
                                              sinfo::Ptr{sils_sinfo_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function sils_information(::Type{Float64}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_double.sils_information(data::Ptr{Ptr{Cvoid}},
                                            ainfo::Ptr{sils_ainfo_type{Float64}},
                                            finfo::Ptr{sils_finfo_type{Float64}},
                                            sinfo::Ptr{sils_sinfo_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

function sils_information(::Type{Float128}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_quadruple.sils_information_q(data::Ptr{Ptr{Cvoid}},
                                                 ainfo::Ptr{sils_ainfo_type{Float128}},
                                                 finfo::Ptr{sils_finfo_type{Float128}},
                                                 sinfo::Ptr{sils_sinfo_type{Float128}},
                                                 status::Ptr{Cint})::Cvoid
end

export sils_finalize

function sils_finalize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.sils_finalize_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sils_control_type{Float32}},
                                           status::Ptr{Cint})::Cvoid
end

function sils_finalize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.sils_finalize(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sils_control_type{Float64}},
                                         status::Ptr{Cint})::Cvoid
end

function sils_finalize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.sils_finalize_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{sils_control_type{Float128}},
                                              status::Ptr{Cint})::Cvoid
end
