export sils_control_type

mutable struct sils_control_type{T}
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

  sils_control_type{T}() where T = new()
end

export sils_ainfo_type

mutable struct sils_ainfo_type{T}
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

  sils_ainfo_type{T}() where T = new()
end

export sils_finfo_type

mutable struct sils_finfo_type{T}
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

  sils_finfo_type{T}() where T = new()
end

export sils_sinfo_type

mutable struct sils_sinfo_type{T}
  flag::Cint
  stat::Cint
  cond::T
  cond2::T
  berr::T
  berr2::T
  error::T

  sils_sinfo_type{T}() where T = new()
end

export sils_initialize_s

function sils_initialize_s(data, control, status)
  @ccall libgalahad_single.sils_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{sils_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export sils_initialize

function sils_initialize(data, control, status)
  @ccall libgalahad_double.sils_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{sils_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export sils_read_specfile_s

function sils_read_specfile_s(control, specfile)
  @ccall libgalahad_single.sils_read_specfile_s(control::Ref{sils_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export sils_read_specfile

function sils_read_specfile(control, specfile)
  @ccall libgalahad_double.sils_read_specfile(control::Ref{sils_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export sils_import_s

function sils_import_s(control, data, status)
  @ccall libgalahad_single.sils_import_s(control::Ref{sils_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint})::Cvoid
end

export sils_import

function sils_import(control, data, status)
  @ccall libgalahad_double.sils_import(control::Ref{sils_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint})::Cvoid
end

export sils_reset_control_s

function sils_reset_control_s(control, data, status)
  @ccall libgalahad_single.sils_reset_control_s(control::Ref{sils_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

export sils_reset_control

function sils_reset_control(control, data, status)
  @ccall libgalahad_double.sils_reset_control(control::Ref{sils_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

export sils_information_s

function sils_information_s(data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_single.sils_information_s(data::Ptr{Ptr{Cvoid}},
                                              ainfo::Ref{sils_ainfo_type{Float32}},
                                              finfo::Ref{sils_finfo_type{Float32}},
                                              sinfo::Ref{sils_sinfo_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export sils_information

function sils_information(data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_double.sils_information(data::Ptr{Ptr{Cvoid}},
                                            ainfo::Ref{sils_ainfo_type{Float64}},
                                            finfo::Ref{sils_finfo_type{Float64}},
                                            sinfo::Ref{sils_sinfo_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export sils_finalize_s

function sils_finalize_s(data, control, status)
  @ccall libgalahad_single.sils_finalize_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{sils_control_type{Float32}},
                                           status::Ptr{Cint})::Cvoid
end

export sils_finalize

function sils_finalize(data, control, status)
  @ccall libgalahad_double.sils_finalize(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{sils_control_type{Float64}},
                                         status::Ptr{Cint})::Cvoid
end
