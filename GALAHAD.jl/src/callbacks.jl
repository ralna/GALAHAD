export galahad_f, galahad_g, galahad_h
export galahad_hprod, galahad_shprod
export galahad_prec, galahad_constant_prec
export galahad_fgh, galahad_fc, galahad_gj, galahad_hl
export galahad_r, galahad_jr, galahad_hr
export galahad_jrprod, galahad_hrprod, galahad_shrprod
export galahad_jr_prod, galahad_jr_sprod, galahad_jr_prods
export galahad_jr_scol

for (T, INT, CT, suffix) in ((:Float32 , :Int32, :Float32  , "_s"   ),
                             (:Float32 , :Int64, :Float32  , "_s_64"),
                             (:Float64 , :Int32, :Float64  , "_d"   ),
                             (:Float64 , :Int64, :Float64  , "_d_64"),
                             (:Float128, :Int32, :Cfloat128, "_q"   ),
                             (:Float128, :Int64, :Cfloat128, "_q_64"))

  julia_galahad_f_r = Symbol(:julia_galahad_f, suffix)
  julia_galahad_g_r = Symbol(:julia_galahad_g, suffix)
  julia_galahad_h_r = Symbol(:julia_galahad_h, suffix)
  julia_galahad_hprod_r = Symbol(:julia_galahad_hprod, suffix)
  julia_galahad_shprod_r = Symbol(:julia_galahad_shprod, suffix)
  julia_galahad_fgh_r = Symbol(:julia_galahad_fgh, suffix)
  julia_galahad_fc_r = Symbol(:julia_galahad_fc, suffix)
  julia_galahad_gj_r = Symbol(:julia_galahad_gj, suffix)
  julia_galahad_hl_r = Symbol(:julia_galahad_hl, suffix)
  julia_galahad_prec_r = Symbol(:julia_galahad_prec, suffix)
  julia_galahad_constant_prec_r = Symbol(:julia_galahad_constant_prec, suffix)
  julia_galahad_r_r = Symbol(:julia_galahad_r, suffix)
  julia_galahad_jr_r = Symbol(:julia_galahad_jr, suffix)
  julia_galahad_hr_r = Symbol(:julia_galahad_hr, suffix)
  julia_galahad_jrprod_r = Symbol(:julia_galahad_jrprod, suffix)
  julia_galahad_hrprod_r = Symbol(:julia_galahad_hrprod, suffix)
  julia_galahad_shrprod_r = Symbol(:julia_galahad_shrprod, suffix)
  julia_galahad_jr_prod_r = Symbol(:julia_galahad_jr_prod, suffix)
  julia_galahad_jr_sprod_r = Symbol(:julia_galahad_jr_sprod, suffix)
  julia_galahad_jr_prods_r = Symbol(:julia_galahad_jr_prods, suffix)
  julia_galahad_jr_scol_r = Symbol(:julia_galahad_jr_scol, suffix)

  @eval begin
    function $julia_galahad_f_r(n::$INT, x::Ptr{$T}, f::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _f = unsafe_wrap(Vector{$T}, f, 1)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_f(_x, _f, _userdata)
      return $INT(0)
    end

    galahad_f(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_f_r)), $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_g_r(n::$INT, x::Ptr{$T}, g::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _g = unsafe_wrap(Vector{$T}, g, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_g(_x, _g, _userdata)
      return $INT(0)
    end

    galahad_g(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_g_r)), $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_h_r(n::$INT, ne::$INT, x::Ptr{$T}, hval::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _hval = unsafe_wrap(Vector{$T}, hval, ne)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_h(_x, _hval, _userdata)
      return $INT(0)
    end

    galahad_h(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_h_r)), $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_hprod_r(n::$INT, x::Ptr{$T}, u::Ptr{$T}, v::Ptr{$T}, got_h::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _u = unsafe_wrap(Vector{$T}, u, n)
      _v = unsafe_wrap(Vector{$T}, v, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_hprod(_x, _u, _v, got_h, _userdata)
      return $INT(0)
    end

    galahad_hprod(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_hprod_r)), $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

    function $julia_galahad_shprod_r(n::$INT, x::Ptr{$T}, nnz_v::$INT, index_nz_v::Ptr{$INT},
                                     v::Ptr{$T}, nnz_u::Ptr{$INT}, index_nz_u::Ptr{$INT},
                                     u::Ptr{$T}, got_h::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _u = unsafe_wrap(Vector{$T}, u, n)
      _v = unsafe_wrap(Vector{$T}, v, n)
      _index_nz_v = unsafe_wrap(Vector{$INT}, index_nz_v, nnz_v)
      _nnz_u = unsafe_wrap(Vector{$INT}, nnz_u, 1)
      _index_nz_u = unsafe_wrap(Vector{$INT}, index_nz_u, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_shprod(_x, nnz_v, _index_nz_v, _v, _nnz_u, _index_nz_u, _u, got_h, _userdata)
      return $INT(0)
    end

    galahad_shprod(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_shprod_r)), $INT, ($INT, Ptr{$T}, $INT, Ptr{$INT}, Ptr{$T}, Ptr{$INT}, Ptr{$INT}, Ptr{$T}, Bool, Ptr{Cvoid}))

    function $julia_galahad_fgh_r(x::$CT, f::Ptr{$T}, g::Ptr{$T}, h::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _f = unsafe_wrap(Vector{$T}, f, 1)
      _g = unsafe_wrap(Vector{$T}, g, 1)
      _h = unsafe_wrap(Vector{$T}, h, 1)
      _userdata = unsafe_pointer_to_objref(userdata)
      _f[1] = _userdata.eval_f(x |> $T, _userdata)
      _g[1] = _userdata.eval_g(x |> $T, _userdata)
      _h[1] = _userdata.eval_h(x |> $T, _userdata)
      return $INT(0)
    end

    galahad_fgh(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_fgh_r)), $INT, ($CT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_fc_r(n::$INT, m::$INT, x::Ptr{$T}, f::Ptr{$T}, c::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _c = unsafe_wrap(Vector{$T}, c, m)
      _f = unsafe_wrap(Vector{$T}, f, 1)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_fc(_x, _f, _c, _userdata)
      return $INT(0)
    end

    galahad_fc(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_fc_r)), $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_gj_r(n::$INT, m::$INT, J_ne::$INT, x::Ptr{$T}, g::Ptr{$T}, jval::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _g = unsafe_wrap(Vector{$T}, g, n)
      _jval = unsafe_wrap(Vector{$T}, jval, J_ne)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_gj(_x, _g, _jval, _userdata)
      return $INT(0)
    end

    galahad_gj(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_gj_r)), $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_hl_r(n::$INT, m::$INT, H_ne::$INT, x::Ptr{$T}, y::Ptr{$T}, hval::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _y = unsafe_wrap(Vector{$T}, y, m)
      _hval = unsafe_wrap(Vector{$T}, hval, H_ne)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_hl(_x, _y, _hval, _userdata)
      return $INT(0)
    end

    galahad_hl(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_hl_r)), $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_prec_r(n::$INT, x::Ptr{$T}, u::Ptr{$T}, v::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _u = unsafe_wrap(Vector{$T}, u, n)
      _v = unsafe_wrap(Vector{$T}, v, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_prec(_x, _u, _v, _userdata)
      return $INT(0)
    end

    galahad_prec(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_prec_r)), $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_constant_prec_r(n::$INT, v::Ptr{$T}, p::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _v = unsafe_wrap(Vector{$T}, v, n)
      _p = unsafe_wrap(Vector{$T}, p, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_prec(_v, _p, _userdata)
      return $INT(0)
    end

    galahad_constant_prec(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_constant_prec_r)), $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_r_r(n::$INT, m::$INT, x::Ptr{$T}, r::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _r = unsafe_wrap(Vector{$T}, r, m)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_r(_x, _r, _userdata)
      return $INT(0)
    end

    galahad_r(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_r_r)), $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_jr_r(n::$INT, m::$INT, jne::$INT, x::Ptr{$T}, jr::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _jr = unsafe_wrap(Vector{$T}, jr, jne)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_jr(_x, _jr, _userdata)
      return $INT(0)
    end

    galahad_jr(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_jr_r)), $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_hr_r(n::$INT, m::$INT, hne::$INT, x::Ptr{$T}, y::Ptr{$T}, hr::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _y = unsafe_wrap(Vector{$T}, y, m)
      _hr = unsafe_wrap(Vector{$T}, hr, hne)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_hr(_x, _y, _hr, _userdata)
      return $INT(0)
    end

    galahad_hr(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_hr_r)), $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_jrprod_r(n::$INT, m::$INT, x::Ptr{$T}, transpose::Bool, u::Ptr{$T}, v::Ptr{$T}, got_j::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _u = unsafe_wrap(Vector{$T}, u, transpose ? n : m)
      _v = unsafe_wrap(Vector{$T}, v, transpose ? m : n)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_jrprod(_x, transpose, _u, _v, got_j, _userdata)
      return $INT(0)
    end

    galahad_jrprod(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_jrprod_r)), $INT, ($INT, $INT, Ptr{$T}, Bool, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

    function $julia_galahad_hrprod_r(n::$INT, m::$INT, x::Ptr{$T}, y::Ptr{$T}, u::Ptr{$T}, v::Ptr{$T}, got_h::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _y = unsafe_wrap(Vector{$T}, y, m)
      _u = unsafe_wrap(Vector{$T}, u, n)
      _v = unsafe_wrap(Vector{$T}, v, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_hrprod(_x, _y, _u, _v, got_h, _userdata)
      return $INT(0)
    end

    galahad_hrprod(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_hrprod_r)), $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

    function $julia_galahad_shrprod_r(n::$INT, m::$INT, pne::$INT, x::Ptr{$T}, v::Ptr{$T}, pval::Ptr{$T}, got_h::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _v = unsafe_wrap(Vector{$T}, v, n)
      _pval = unsafe_wrap(Vector{$T}, pval, pne)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_shrprod(_x, _v, _pval, got_h, _userdata)
      return $INT(0)
    end

    galahad_shrprod(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_shrprod_r)), $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

    function $julia_galahad_jr_prod_r(n::$INT, m_r::$INT, x::Ptr{$T}, transpose::Bool, v::Ptr{$T}, p::Ptr{$T}, got_jr::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _v = unsafe_wrap(Vector{$T}, v, transpose ? m_r : n)
      _p = unsafe_wrap(Vector{$T}, p, transpose ? n : m_r)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_jr_prod(_x, transpose, _v, _p, got_jr, _userdata)
      return $INT(0)
    end

    galahad_jr_prod(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_jr_prod_r)), $INT, ($INT, $INT, Ptr{$T}, Bool, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

    function $julia_galahad_jr_sprod_r(n::$INT, m_r::$INT, x::Ptr{$T}, transpose::Bool, v::Ptr{$T}, p::Ptr{$T}, free::Ptr{$INT}, n_free::$INT, got_jr::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _v = unsafe_wrap(Vector{$T}, v, transpose ? m_r : n)
      _p = unsafe_wrap(Vector{$T}, p, transpose ? n : m_r)
      _free = unsafe_wrap(Vector{$INT}, free, n_free)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_jr_sprod(n, m_r, _x, transpose, _v, _p, _free, n_free, got_jr, _userdata)
      return $INT(0)
    end

    galahad_jr_sprod(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_jr_sprod_r)), $INT, ($INT, $INT, Ptr{$T}, Bool, Ptr{$T}, Ptr{$T}, Ptr{$INT}, $INT, Bool, Ptr{Cvoid}))

    function $julia_galahad_jr_prods_r(n::$INT, m_r::$INT, x::Ptr{$T}, v::Ptr{$T}, p::Ptr{$T}, iv::Ptr{$INT}, lvl::$INT, lvu::$INT, ip::Ptr{$INT}, lp::Ptr{$INT}, got_jr::Bool, userdata::Ptr{Cvoid})::$INT
      mnm = max(m_r, n)
      _x = unsafe_wrap(Vector{$T}, x, n)
      _v = unsafe_wrap(Vector{$T}, v, n)
      _p = unsafe_wrap(Vector{$T}, p, m_r)
      _iv = unsafe_wrap(Vector{$INT}, iv, mnm)
      _ip = unsafe_wrap(Vector{$INT}, ip, ip == C_NULL ? 0 : m_r)
      _lp = unsafe_wrap(Vector{$INT}, lp, lp == C_NULL ? 0 : 1)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_jr_prods(n, m_r, _x, _v, _p, _iv, lvl, lvu, _ip, _lp, got_jr, _userdata)
      return $INT(0)
    end

    galahad_jr_prods(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_jr_prods_r)), $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$INT}, $INT, $INT, Ptr{$INT}, Ptr{$INT}, Bool, Ptr{Cvoid}))

    function $julia_galahad_jr_scol_r(n::$INT, m_r::$INT, x::Ptr{$T}, index::$INT, val::Ptr{$T}, row::Ptr{$INT}, nz::Ptr{$INT}, got_jr::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _val = unsafe_wrap(Vector{$T}, val, n)
      _row = unsafe_wrap(Vector{$INT}, row, n)
      _nz = unsafe_wrap(Vector{$INT}, nz, 1)
      _userdata = unsafe_pointer_to_objref(userdata)
      _userdata.eval_jr_scol(n, _x, index, _val, _row, _nz, got_jr, _userdata)
      return $INT(0)
    end

    galahad_jr_scol(::Type{$T}, ::Type{$INT}) = @cfunction($(Expr(:$, julia_galahad_jr_scol_r)), $INT, ($INT, $INT, Ptr{$T}, $INT, Ptr{$T}, Ptr{$INT}, $Ptr{$INT}, Bool, Ptr{Cvoid}))
  end
end
