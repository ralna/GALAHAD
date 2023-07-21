mutable struct bgo_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    attempts_max::Cint
    max_evals::Cint
    sampling_strategy::Cint
    hypercube_discretization::Cint
    alive_unit::Cint
    alive_file::NTuple{31,Cchar}
    infinity::Float64
    obj_unbounded::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    random_multistart::Bool
    hessian_available::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
    ugo_control::ugo_control_type
    lhs_control::lhs_control_type
    trb_control::trb_control_type
end

mutable struct bgo_time_type
    total::Float32
    univariate_global::Float32
    multivariate_local::Float32
    clock_total::Float64
    clock_univariate_global::Float64
    clock_multivariate_local::Float64
end

mutable struct bgo_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    f_eval::Cint
    g_eval::Cint
    h_eval::Cint
    obj::Float64
    norm_pg::Float64
    time::bgo_time_type
    ugo_inform::ugo_inform_type
    lhs_inform::lhs_inform_type
    trb_inform::trb_inform_type
end

function bgo_initialize(data, control, status)
    @ccall libgalahad_double.bgo_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{bgo_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function bgo_read_specfile(control, specfile)
    @ccall libgalahad_double.bgo_read_specfile(control::Ref{bgo_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function bgo_import(control, data, status, n, x_l, x_u, H_type, ne, H_row, H_col, H_ptr)
    @ccall libgalahad_double.bgo_import(control::Ref{bgo_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function bgo_reset_control(control, data, status)
    @ccall libgalahad_double.bgo_reset_control(control::Ref{bgo_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function bgo_solve_with_mat(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                            eval_hprod, eval_prec)
    @ccall libgalahad_double.bgo_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint,
                                                x::Ptr{Float64}, g::Ptr{Float64},
                                                ne::Cint, eval_f::Ptr{Cvoid},
                                                eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                eval_hprod::Ptr{Cvoid},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_without_mat(data, userdata, status, n, x, g, eval_f, eval_g, eval_hprod,
                               eval_shprod, eval_prec)
    @ccall libgalahad_double.bgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                   n::Cint, x::Ptr{Float64},
                                                   g::Ptr{Float64}, eval_f::Ptr{Cvoid},
                                                   eval_g::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid},
                                                   eval_shprod::Ptr{Cvoid},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
    @ccall libgalahad_double.bgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint},
                                                        eval_status::Ptr{Cint}, n::Cint,
                                                        x::Ptr{Float64}, f::Float64,
                                                        g::Ptr{Float64}, ne::Cint,
                                                        H_val::Ptr{Float64},
                                                        u::Ptr{Float64},
                                                        v::Ptr{Float64})::Cvoid
end

function bgo_solve_reverse_without_mat(data, status, eval_status, n, x, f, g, u, v,
                                       index_nz_v, nnz_v, index_nz_u, nnz_u)
    @ccall libgalahad_double.bgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Cint},
                                                           eval_status::Ptr{Cint}, n::Cint,
                                                           x::Ptr{Float64}, f::Float64,
                                                           g::Ptr{Float64},
                                                           u::Ptr{Float64},
                                                           v::Ptr{Float64},
                                                           index_nz_v::Ptr{Cint},
                                                           nnz_v::Ptr{Cint},
                                                           index_nz_u::Ptr{Cint},
                                                           nnz_u::Cint)::Cvoid
end

function bgo_information(data, inform, status)
    @ccall libgalahad_double.bgo_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{bgo_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function bgo_terminate(data, control, inform)
    @ccall libgalahad_double.bgo_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{bgo_control_type},
                                           inform::Ref{bgo_inform_type})::Cvoid
end
