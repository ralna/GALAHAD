mutable struct eqp_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    factorization::Cint
    max_col::Cint
    indmin::Cint
    valmin::Cint
    len_ulsmin::Cint
    itref_max::Cint
    cg_maxit::Cint
    preconditioner::Cint
    semi_bandwidth::Cint
    new_a::Cint
    new_h::Cint
    sif_file_device::Cint
    pivot_tol::Float64
    pivot_tol_for_basis::Float64
    zero_pivot::Float64
    inner_fraction_opt::Float64
    radius::Float64
    min_diagonal::Float64
    max_infeasibility_relative::Float64
    max_infeasibility_absolute::Float64
    inner_stop_relative::Float64
    inner_stop_absolute::Float64
    inner_stop_inter::Float64
    find_basis_by_transpose::Bool
    remove_dependencies::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    generate_sif_file::Bool
    sif_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    fdc_control::fdc_control_type
    sbls_control::sbls_control_type
    gltr_control::gltr_control_type
end

mutable struct eqp_time_type
    total::Float64
    find_dependent::Float64
    factorize::Float64
    solve::Float64
    solve_inter::Float64
    clock_total::Float64
    clock_find_dependent::Float64
    clock_factorize::Float64
    clock_solve::Float64
end

mutable struct eqp_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    cg_iter::Cint
    cg_iter_inter::Cint
    factorization_integer::Int64
    factorization_real::Int64
    obj::Float64
    time::eqp_time_type
    fdc_inform::fdc_inform_type
    sbls_inform::sbls_inform_type
    gltr_inform::gltr_inform_type
end

function eqp_initialize(data, control, status)
    @ccall libgalahad_double.eqp_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{eqp_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function eqp_read_specfile(control, specfile)
    @ccall libgalahad_double.eqp_read_specfile(control::Ptr{eqp_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function eqp_import(control, data, status, n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type,
                    A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.eqp_import(control::Ptr{eqp_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function eqp_reset_control(control, data, status)
    @ccall libgalahad_double.eqp_reset_control(control::Ptr{eqp_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function eqp_solve_qp(data, status, n, m, h_ne, H_val, g, f, a_ne, A_val, c, x, y)
    @ccall libgalahad_double.eqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                          g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                          A_val::Ptr{Float64}, c::Ptr{Float64},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function eqp_solve_sldqp(data, status, n, m, w, x0, g, f, a_ne, A_val, c, x, y)
    @ccall libgalahad_double.eqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, w::Ptr{Float64},
                                             x0::Ptr{Float64}, g::Ptr{Float64},
                                             f::Float64, a_ne::Cint, A_val::Ptr{Float64},
                                             c::Ptr{Float64}, x::Ptr{Float64},
                                             y::Ptr{Float64})::Cvoid
end

function eqp_resolve_qp(data, status, n, m, g, f, c, x, y)
    @ccall libgalahad_double.eqp_resolve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, m::Cint, g::Ptr{Float64}, f::Float64,
                                            c::Ptr{Float64}, x::Ptr{Float64},
                                            y::Ptr{Float64})::Cvoid
end

function eqp_information(data, inform, status)
    @ccall libgalahad_double.eqp_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{eqp_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function eqp_terminate(data, control, inform)
    @ccall libgalahad_double.eqp_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{eqp_control_type},
                                           inform::Ptr{eqp_inform_type})::Cvoid
end
