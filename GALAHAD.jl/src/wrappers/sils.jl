mutable struct sils_control_type
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
    CNTL::NTuple{5,Float64}
    multiplier::Float64
    reduce::Float64
    u::Float64
    static_tolerance::Float64
    static_level::Float64
    tolerance::Float64
    convergence::Float64
end

mutable struct sils_ainfo_type
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
    opsa::Float64
    opse::Float64
end

mutable struct sils_finfo_type
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
    opsa::Float64
    opse::Float64
    opsb::Float64
    maxchange::Float64
    smin::Float64
    smax::Float64
end

mutable struct sils_sinfo_type
    flag::Cint
    stat::Cint
    cond::Float64
    cond2::Float64
    berr::Float64
    berr2::Float64
    error::Float64
end

function sils_initialize(data, control, status)
    @ccall libgalahad_double.sils_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{sils_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function sils_read_specfile(control, specfile)
    @ccall libgalahad_double.sils_read_specfile(control::Ref{sils_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function sils_import(control, data, status)
    @ccall libgalahad_double.sils_import(control::Ref{sils_control_type},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint})::Cvoid
end

function sils_reset_control(control, data, status)
    @ccall libgalahad_double.sils_reset_control(control::Ref{sils_control_type},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function sils_information(data, ainfo, finfo, sinfo, status)
    @ccall libgalahad_double.sils_information(data::Ptr{Ptr{Cvoid}},
                                              ainfo::Ref{sils_ainfo_type},
                                              finfo::Ref{sils_finfo_type},
                                              sinfo::Ref{sils_sinfo_type},
                                              status::Ptr{Cint})::Cvoid
end

function sils_finalize(data, control, status)
    @ccall libgalahad_double.sils_finalize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{sils_control_type},
                                           status::Ptr{Cint})::Cvoid
end
