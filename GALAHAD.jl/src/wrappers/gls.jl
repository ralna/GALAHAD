mutable struct gls_control
    f_indexing::Bool
    lp::Cint
    wp::Cint
    mp::Cint
    ldiag::Cint
    btf::Cint
    maxit::Cint
    factor_blocking::Cint
    solve_blas::Cint
    la::Cint
    la_int::Cint
    maxla::Cint
    pivoting::Cint
    fill_in::Cint
    multiplier::Float64
    reduce::Float64
    u::Float64
    switch_full::Float64
    drop::Float64
    tolerance::Float64
    cgce::Float64
    diagonal_pivoting::Bool
    struct_abort::Bool
end

mutable struct gls_ainfo
    flag::Cint
    more::Cint
    len_analyse::Cint
    len_factorize::Cint
    ncmpa::Cint
    rank::Cint
    drop::Cint
    struc_rank::Cint
    oor::Cint
    dup::Cint
    stat::Cint
    lblock::Cint
    sblock::Cint
    tblock::Cint
    ops::Float64
end

mutable struct gls_finfo
    flag::Cint
    more::Cint
    size_factor::Cint
    len_factorize::Cint
    drop::Cint
    rank::Cint
    stat::Cint
    ops::Float64
end

mutable struct gls_sinfo
    flag::Cint
    more::Cint
    stat::Cint
end

function gls_initialize(data, control)
    @ccall libgalahad_double.gls_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{gls_control})::Cvoid
end

function gls_read_specfile(control, specfile)
    @ccall libgalahad_double.gls_read_specfile(control::Ptr{gls_control},
                                               specfile::Ptr{Cchar})::Cvoid
end

function gls_import(control, data, status)
    @ccall libgalahad_double.gls_import(control::Ptr{gls_control}, data::Ptr{Ptr{Cvoid}},
                                        status::Ptr{Cint})::Cvoid
end

function gls_reset_control(control, data, status)
    @ccall libgalahad_double.gls_reset_control(control::Ptr{gls_control},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function gls_information(data, ainfo, finfo, sinfo, status)
    @ccall libgalahad_double.gls_information(data::Ptr{Ptr{Cvoid}}, ainfo::Ptr{gls_ainfo},
                                             finfo::Ptr{gls_finfo}, sinfo::Ptr{gls_sinfo},
                                             status::Ptr{Cint})::Cvoid
end

function gls_finalize(data, control, status)
    @ccall libgalahad_double.gls_finalize(data::Ptr{Ptr{Cvoid}}, control::Ptr{gls_control},
                                          status::Ptr{Cint})::Cvoid
end
