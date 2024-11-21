#ifdef INTEGER_64
#define ma48_control_r ma48_control_q_64
#define ma48_ainfo_r ma48_ainfo_q_64
#define ma48_finfo_r ma48_finfo_q_64
#define ma48_sinfo_r ma48_sinfo_q_64
#define ma48_initialize_r ma48_initialize_q_64
#define ma48_default_control_r ma48_default_control_q_64
#define ma48_analyse_r ma48_analyse_q_64
#define ma48_get_perm_r ma48_get_perm_q_64
#define ma48_factorize_r ma48_factorize_q_64
#define ma48_solve_r ma48_solve_q_64
#define ma48_finalize_r ma48_finalize_q_64
#define ma48_special_rows_and_cols_r ma48_special_rows_and_cols_q_64
#define ma48_determinant_r ma48_determinant_q_64
#else
#define ma48_control_r ma48_control_q
#define ma48_ainfo_r ma48_ainfo_q
#define ma48_finfo_r ma48_finfo_q
#define ma48_sinfo_r ma48_sinfo_q
#define ma48_initialize_r ma48_initialize_q
#define ma48_default_control_r ma48_default_control_q
#define ma48_analyse_r ma48_analyse_q
#define ma48_get_perm_r ma48_get_perm_q
#define ma48_factorize_r ma48_factorize_q
#define ma48_solve_r ma48_solve_q
#define ma48_finalize_r ma48_finalize_q
#define ma48_special_rows_and_cols_r ma48_special_rows_and_cols_q
#define ma48_determinant_r ma48_determinant_q
#endif

#ifdef INTEGER_64
#define ma57_default_control_r      ma57_default_control_q_64
#define ma57_init_factors_r         ma57_init_factors_q_64
#define ma57_control_r              ma57_control_q_64
#define ma57_ainfo_r                ma57_ainfo_q_64
#define ma57_finfo_r                ma57_finfo_q_64
#define ma57_sinfo_r                ma57_sinfo_q_64
#define ma57_analyse_r              ma57_analyse_q_64
#define ma57_factorize_r            ma57_factorize_q_64
#define ma57_solve_r                ma57_solve_q_64
#define ma57_finalize_r             ma57_finalize_q_64
#define ma57_enquire_perm_r    	  ma57_enquire_perm_q_64
#define ma57_enquire_pivots_r  	  ma57_enquire_pivots_q_64
#define ma57_enquire_d_r       	  ma57_enquire_d_q_64
#define ma57_enquire_perturbation_r ma57_enquire_perturbation_q_64
#define ma57_enquire_scaling_r      ma57_enquire_scaling_q_64
#define ma57_alter_d_r              ma57_alter_d_q_64
#define ma57_part_solve_r           ma57_part_solve_q_64
#define ma57_sparse_lsolve_r        ma57_sparse_lsolve_q_64
#define ma57_fredholm_alternative_r ma57_fredholm_alternative_q_64
#define ma57_lmultiply_r            ma57_lmultiply_q_64
#define ma57_get_factors_r          ma57_get_factors_q_64
#else
#define ma57_default_control_r      ma57_default_control_q
#define ma57_init_factors_r         ma57_init_factors_q
#define ma57_control_r              ma57_control_q
#define ma57_ainfo_r                ma57_ainfo_q
#define ma57_finfo_r                ma57_finfo_q
#define ma57_sinfo_r                ma57_sinfo_q
#define ma57_analyse_r              ma57_analyse_q
#define ma57_factorize_r            ma57_factorize_q
#define ma57_solve_r                ma57_solve_q
#define ma57_finalize_r             ma57_finalize_q
#define ma57_enquire_perm_r    	  ma57_enquire_perm_q
#define ma57_enquire_pivots_r  	  ma57_enquire_pivots_q
#define ma57_enquire_d_r       	  ma57_enquire_d_q
#define ma57_enquire_perturbation_r ma57_enquire_perturbation_q
#define ma57_enquire_scaling_r      ma57_enquire_scaling_q
#define ma57_alter_d_r              ma57_alter_d_q
#define ma57_part_solve_r           ma57_part_solve_q
#define ma57_sparse_lsolve_r        ma57_sparse_lsolve_q
#define ma57_fredholm_alternative_r ma57_fredholm_alternative_q
#define ma57_lmultiply_r            ma57_lmultiply_q
#define ma57_get_factors_r          ma57_get_factors_q
#endif

#ifdef INTEGER_64
#define ma77_control_r ma77_control_q_64
#define ma77_info_r ma77_info_q_64
#define ma77_default_control_r ma77_default_control_q_64
#define ma77_open_nelt_r ma77_open_nelt_q_64
#define ma77_open_r ma77_open_q_64
#define ma77_input_vars_r ma77_input_vars_q_64
#define ma77_input_reals_r ma77_input_reals_q_64
#define ma77_analyse_r ma77_analyse_q_64
#define ma77_factor_r ma77_factor_q_64
#define ma77_factor_solve_r ma77_factor_solve_q_64
#define ma77_solve_r ma77_solve_q_64
#define ma77_resid_r ma77_resid_q_64
#define ma77_scale_r ma77_scale_q_64
#define ma77_enquire_posdef_r ma77_enquire_posdef_q_64
#define ma77_enquire_indef_r ma77_enquire_indef_q_64
#define ma77_alter_r ma77_alter_q_64
#define ma77_restart_r ma77_restart_q_64
#define ma77_finalise_r ma77_finalise_q_64
#define ma77_solve_fredholm_r ma77_solve_fredholm_q_64
#define ma77_lmultiply_r ma77_lmultiply_q_64
#else
#define ma77_control_r ma77_control_q
#define ma77_info_r ma77_info_q
#define ma77_default_control_r ma77_default_control_q
#define ma77_open_nelt_r ma77_open_nelt_q
#define ma77_open_r ma77_open_q
#define ma77_input_vars_r ma77_input_vars_q
#define ma77_input_reals_r ma77_input_reals_q
#define ma77_analyse_r ma77_analyse_q
#define ma77_factor_r ma77_factor_q
#define ma77_factor_solve_r ma77_factor_solve_q
#define ma77_solve_r ma77_solve_q
#define ma77_resid_r ma77_resid_q
#define ma77_scale_r ma77_scale_q
#define ma77_enquire_posdef_r ma77_enquire_posdef_q
#define ma77_enquire_indef_r ma77_enquire_indef_q
#define ma77_alter_r ma77_alter_q
#define ma77_restart_r ma77_restart_q
#define ma77_finalise_r ma77_finalise_q
#define ma77_solve_fredholm_r ma77_solve_fredholm_q
#define ma77_lmultiply_r ma77_lmultiply_q
#endif

#ifdef INTEGER_64
#define ma86_control_r ma86_control_q_64
#define ma86_info_r ma86_info_q_64
#define ma86_default_control_r ma86_default_control_q_64
#define ma86_analyse_r ma86_analyse_q_64
#define ma86_factor_r ma86_factor_q_64
#define ma86_factor_solve_r ma86_factor_solve_q_64
#define ma86_solve_r ma86_solve_q_64
#define ma86_finalise_r ma86_finalise_q_64
#else
#define ma86_control_r ma86_control_q
#define ma86_info_r ma86_info_q
#define ma86_default_control_r ma86_default_control_q
#define ma86_analyse_r ma86_analyse_q
#define ma86_factor_r ma86_factor_q
#define ma86_factor_solve_r ma86_factor_solve_q
#define ma86_solve_r ma86_solve_q
#define ma86_finalise_r ma86_finalise_q
#endif

#ifdef INTEGER_64
#define ma87_control_r ma87_control_q_64
#define ma87_info_r ma87_info_q_64
#define ma87_default_control_r ma87_default_control_q_64
#define ma87_analyse_r ma87_analyse_q_64
#define ma87_factor_r ma87_factor_q_64
#define ma87_factor_solve_r ma87_factor_solve_q_64
#define ma87_solve_r ma87_solve_q_64
#define ma87_sparse_fwd_solve_r ma87_sparse_fwd_solve_q_64
#define ma87_finalise_r ma87_finalise_q_64
#else
#define ma87_control_r ma87_control_q
#define ma87_info_r ma87_info_q
#define ma87_default_control_r ma87_default_control_q
#define ma87_analyse_r ma87_analyse_q
#define ma87_factor_r ma87_factor_q
#define ma87_factor_solve_r ma87_factor_solve_q
#define ma87_solve_r ma87_solve_q
#define ma87_sparse_fwd_solve_r ma87_sparse_fwd_solve_q
#define ma87_finalise_r ma87_finalise_q
#endif

#ifdef INTEGER_64
#define ma97_control_r ma97_control_q_64
#define ma97_info_r ma97_info_q_64
#define ma97_default_control_r ma97_default_control_q_64
#define ma97_analyse_r ma97_analyse_q_64
#define ma97_analyse_coord_r ma97_analyse_coord_q_64
#define ma97_factor_r ma97_factor_q_64
#define ma97_factor_solve_r ma97_factor_solve_q_64
#define ma97_solve_r ma97_solve_q_64
#define ma97_free_akeep_r ma97_free_akeep_q_64
#define ma97_free_fkeep_r ma97_free_fkeep_q_64
#define ma97_finalise_r ma97_finalise_q_64
#define ma97_enquire_posdef_r ma97_enquire_posdef_q_64
#define ma97_enquire_indef_r ma97_enquire_indef_q_64
#define ma97_alter_r ma97_alter_q_64
#define ma97_solve_fredholm_r ma97_solve_fredholm_q_64
#define ma97_lmultiply_r ma97_lmultiply_q_64
#define ma97_sparse_fwd_solve_r ma97_sparse_fwd_solve_q_64
#else
#define ma97_control_r ma97_control_q
#define ma97_info_r ma97_info_q
#define ma97_default_control_r ma97_default_control_q
#define ma97_analyse_r ma97_analyse_q
#define ma97_analyse_coord_r ma97_analyse_coord_q
#define ma97_factor_r ma97_factor_q
#define ma97_factor_solve_r ma97_factor_solve_q
#define ma97_solve_r ma97_solve_q
#define ma97_free_akeep_r ma97_free_akeep_q
#define ma97_free_fkeep_r ma97_free_fkeep_q
#define ma97_finalise_r ma97_finalise_q
#define ma97_enquire_posdef_r ma97_enquire_posdef_q
#define ma97_enquire_indef_r ma97_enquire_indef_q
#define ma97_alter_r ma97_alter_q
#define ma97_solve_fredholm_r ma97_solve_fredholm_q
#define ma97_lmultiply_r ma97_lmultiply_q
#define ma97_sparse_fwd_solve_r ma97_sparse_fwd_solve_q
#endif

#ifdef INTEGER_64
#define mc64_control_r mc64_control_q_64
#define mc64_info_r mc64_info_q_64
#define mc64_default_control_r mc64_default_control_q_64
#define mc64_matching_r mc64_matching_q_64
#else
#define mc64_control_r mc64_control_q
#define mc64_info_r mc64_info_q
#define mc64_default_control_r mc64_default_control_q
#define mc64_matching_r mc64_matching_q
#endif

#ifdef INTEGER_64
#define mi20_default_control_r mi20_default_control_q_64
#define mi20_control_r mi20_control_q_64
#define mi20_default_solve_control_r mi20_default_solve_control_q_64
#define mi20_solve_control_r mi20_solve_control_q_64
#define mi20_info_r mi20_info_q_64
#define mi20_setup_r mi20_setup_q_64
#define mi20_setup_csr_r mi20_setup_csr_q_64
#define mi20_setup_csc_r mi20_setup_csc_q_64
#define mi20_setup_coord_r mi20_setup_coord_q_64
#define mi20_finalize_r mi20_finalize_q_64
#define mi20_precondition_r mi20_precondition_q_64
#define mi20_solve_r mi20_solve_q_64
#else
#define mi20_default_control_r mi20_default_control_q
#define mi20_control_r mi20_control_q
#define mi20_default_solve_control_r mi20_default_solve_control_q
#define mi20_solve_control_r mi20_solve_control_q
#define mi20_info_r mi20_info_q
#define mi20_setup_r mi20_setup_q
#define mi20_setup_csr_r mi20_setup_csr_q
#define mi20_setup_csc_r mi20_setup_csc_q
#define mi20_setup_coord_r mi20_setup_coord_q
#define mi20_finalize_r mi20_finalize_q
#define mi20_precondition_r mi20_precondition_q
#define mi20_solve_r mi20_solve_q
#endif

#ifdef INTEGER_64
#define mi28_control_r mi28_control_q_64
#define mi28_info_r mi28_info_q_64
#define mi28_default_control_r mi28_default_control_q_64
#define mi28_factorize_r mi28_factorize_q_64
#define mi28_precondition_r mi28_precondition_q_64
#define mi28_solve_r mi28_solve_q_64
#define mi28_finalise_r mi28_finalise_q_64
#else
#define mi28_control_r mi28_control_q
#define mi28_info_r mi28_info_q
#define mi28_default_control_r mi28_default_control_q
#define mi28_factorize_r mi28_factorize_q
#define mi28_precondition_r mi28_precondition_q
#define mi28_solve_r mi28_solve_q
#define mi28_finalise_r mi28_finalise_q
#endif
