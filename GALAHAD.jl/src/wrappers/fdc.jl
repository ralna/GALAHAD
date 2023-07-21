mutable struct fdc_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    indmin::Cint
    valmin::Cint
    pivot_tol::Float64
    zero_pivot::Float64
    max_infeas::Float64
    use_sls::Bool
    scale::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    symmetric_linear_solver::NTuple{31,Cchar}
    unsymmetric_linear_solver::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    sls_control::sls_control_type
    uls_control::uls_control_type
end

mutable struct fdc_time_type
    total::Float64
    analyse::Float64
    factorize::Float64
    clock_total::Float64
    clock_analyse::Float64
    clock_factorize::Float64
end

mutable struct fdc_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    factorization_status::Cint
    factorization_integer::Int64
    factorization_real::Int64
    non_negligible_pivot::Float64
    time::fdc_time_type
    sls_inform::sls_inform_type
    uls_inform::uls_inform_type
end

function fdc_initialize(data, control, status)
    @ccall libgalahad_double.fdc_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{fdc_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function fdc_read_specfile(control, specfile)
    @ccall libgalahad_double.fdc_read_specfile(control::Ref{fdc_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function fdc_find_dependent_rows(control, data, inform, status, m, n, A_ne, A_col, A_ptr,
                                 A_val, b, n_depen, depen)
    @ccall libgalahad_double.fdc_find_dependent_rows(control::Ref{fdc_control_type},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     inform::Ref{fdc_inform_type},
                                                     status::Ptr{Cint}, m::Cint, n::Cint,
                                                     A_ne::Cint, A_col::Ptr{Cint},
                                                     A_ptr::Ptr{Cint}, A_val::Ptr{Float64},
                                                     b::Ptr{Float64}, n_depen::Ptr{Cint},
                                                     depen::Ptr{Cint})::Cvoid
end

function fdc_terminate(data, control, inform)
    @ccall libgalahad_double.fdc_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{fdc_control_type},
                                           inform::Ref{fdc_inform_type})::Cvoid
end
