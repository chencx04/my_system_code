from scipy.special import beta
import numpy as np
from scipy.integrate import quad
from constant import Const


def cal_density_temperature_profile(model):
    # 需要电子密度数据，电子温度数据，剖面因子，电子和离子的比值
    """
    the input below is need:
        ndensity information:
            prof_data.ndensity_e_on_axis: 轴上电子密度
            prof_data.ndensity_e_ped: 台基电子密度
            prof_data.ndensity_e_sep: 分离面电子密度
            prof_data.rho_ped_ndensity: 密度台基处归一化小半径
            prof_data.alpha_ndensity: 密度剖面因子
        
        temperature information:
            prof_data.temp_e_on_axis: 轴上电子温度
            prof_data.temp_e_ped: 台基电子温度
            prof_data.temp_e_sep: 分离面电子温度
            prof_data.rho_ped_temp: 温度台基处归一化小半径
            prof_data.alpha_temp: 温度剖面因子 alpha
            prof_data.beta_temp: 温度剖面因子 beta
        
        prof_data.i_plasma_pedestal: 是否为台基

        prof_data.f_temp_electron_ion: 电子和离子的温度比值
        comp_data.f_density_ion_total_electron: 离子总密度占电子密度比

    the part of output is: (change prof_data's attributes)
        the value of density and temperature:
            prof_data.ndensity_e_vol_avg
            prof_data.temp_e_vol_avg
            prof_data.ndensity_i_vol_avg
            prof_data.temp_i_vol_avg
            prof_data.ndensity_i_on_axis
            prof_data.temp_i_on_axis

        the profile function, normalized by average value:
            prof_data.density_func_norm_avg
            prof_data.temp_func_norm_avg

        the line-averaged density for fusion power calculation:
            prof_data.ndensity_e_line_avg
    """


    
    # calculate on-axis and vol-avg value for ion/electron
    # include ndensity_e_vol_avg, ndensity_e_on_axis, temp_e_vol_avg, temp_e_on_axis
    #         ndensity_i_vol_avg, ndensity_i_on_axis, temp_i_vol_avg, temp_i_on_axis
    model.prof_data.ndensity_e_vol_avg = cal_n_vol_avg(model.prof_data.ndensity_e_on_axis,
                                               model.prof_data.ndensity_e_ped,model.prof_data.ndensity_e_sep,
                                               model.prof_data.rho_ped_ndensity,
                                               model.prof_data.alpha_ndensity,
                                               model.prof_data.i_plasma_pedestal)
    model.prof_data.temp_e_vol_avg = cal_temp_vol_avg(model.prof_data.temp_e_on_axis,
                                              model.prof_data.temp_e_ped,model.prof_data.temp_e_sep,
                                              model.prof_data.rho_ped_temp,
                                              model.prof_data.alpha_temp,model.prof_data.beta_temp,
                                              model.prof_data.i_plasma_pedestal)
    model.prof_data.temp_i_on_axis = model.prof_data.temp_e_on_axis/model.prof_data.f_temp_electron_ion
    model.prof_data.temp_i_vol_avg = model.prof_data.temp_e_vol_avg/model.prof_data.f_temp_electron_ion
    model.prof_data.temp_i_ped = model.prof_data.temp_e_ped/model.prof_data.f_temp_electron_ion
    model.prof_data.temp_i_sep = model.prof_data.temp_E_sep/model.prof_data.f_temp_electron_ion
    model.prof_data.ndensity_i_vol_avg = model.prof_data.ndensity_e_vol_avg*model.comp_data.f_density_ion_total_electron
    model.prof_data.ndensity_i_on_axis = model.prof_data.ndensity_e_on_axis*model.comp_data.f_density_ion_total_electron
    model.prof_data.ndensity_i_ped = model.prof_data.ndensity_e_ped*model.comp_data.f_density_ion_total_electron
    model.prof_data.ndensity_i_sep = model.prof_data.ndensity_e_sep*model.comp_data.f_density_ion_total_electron

    # ---------------- #
    # calculate profile function, using vol-avg value for normalization
    model.prof_data.density_func_norm_avg=density_profile_func(model.prof_data.ndensity_e_on_axis,
                                                         model.prof_data.ndensity_e_vol_avg,
                                                         model.prof_data.ndensity_e_ped,
                                                         model.prof_data.ndensity_e_sep,
                                                         model.prof_data.rho_ped_ndensity,
                                                         model.prof_data.alpha_ndensity,
                                                         model.prof_data.i_plasma_pedestal)
    model.prof_data.temp_func_norm_avg=temp_profile_func(model.prof_data.temp_e_on_axis,
                                                   model.prof_data.temp_e_vol_avg,
                                                   model.prof_data.temp_e_ped,
                                                   model.prof_data.temp_e_sep,
                                                   model.prof_data.rho_ped_temp,
                                                   model.prof_data.alpha_temp, model.prof_data.beta_temp,
                                                   model.prof_data.i_plasma_pedestal)

    # line-averaged density for fusion power calculation
    navg, _=quad(model.prof_data.density_func_norm_avg,0,1)
    model.prof_data.ndensity_e_line_avg=navg * model.prof_data.ndensity_e_vol_avg
    

def cal_n_vol_avg(n0,n_ped,n_sep,rho_ped_n,alphan,i_plasma_pedestal):
        if i_plasma_pedestal==1:
            return (
                ((n0 + n_ped * alphan) * rho_ped_n**2 / (1 + alphan))
                + 1/3 * (1 - rho_ped_n) * (n_ped * (2 * rho_ped_n + 1) + n_sep * (rho_ped_n + 2))
            )
        elif i_plasma_pedestal==0:
            return n0 / (1 + alphan)
   

def cal_temp_vol_avg(t0,t_ped,t_sep,rho_ped_t,alphat,betat,i_plasma_pedestal):
    if i_plasma_pedestal==1:
        t_avg = (
            t_ped * rho_ped_t**2 + (t0 - t_ped) * 2 / betat * rho_ped_t**2 * beta(1 + alphat, 2 / betat)
            + 1/3 * (1 - rho_ped_t) * (t_ped * (2 * rho_ped_t + 1) + t_sep * (rho_ped_t + 2))
        )
    elif i_plasma_pedestal==0:
        t_avg = t0 / (1 + alphat)
    return t_avg


def density_profile_func(n0, n_avg, n_ped, n_sep, rho_ped_n, alphan, i_plasma_pedestal):
    def fun_ped(rho):      
        part1 = (n_ped + (n0 - n_ped) * (1 - rho**2 / rho_ped_n**2)**alphan)
        part2 = (n_sep + (n_ped - n_sep) * (1 - rho) / (1 - rho_ped_n))
        return np.where(rho <= rho_ped_n, part1, part2) / n_avg
    if i_plasma_pedestal == 1:
        return fun_ped
    elif i_plasma_pedestal==0:
        return lambda rho: n0 * (1 - rho**2)**alphan / n_avg


def temp_profile_func(t0, t_avg, t_ped, t_sep, rho_ped_t, alphat, betat, i_plasma_pedestal):
    def fun_ped(rho):
        part1 = (t_ped + (t0 - t_ped) * (1 - (rho**betat) / (rho_ped_t**betat))**alphat)
        part2 = (t_sep + (t_ped - t_sep) * (1 - rho) / (1 - rho_ped_t))
        return np.where(rho <= rho_ped_t, part1, part2) / t_avg
    if i_plasma_pedestal == 1:
        return fun_ped
    elif i_plasma_pedestal==0:
        return lambda rho: t0 * (1 - rho**2)**alphat / t_avg


def cal_pressure_profile(
    ne, te, 
    ne0, te0, f_den_ie, f_temp_ei, 
    ne_avg, te_avg, 
    i_dimension):
    """
    Calculate pressure profile and volume-averaged pressure profile.
    ne: density_func_norm_avg
    te: temp_func_norm_avg
    ne0: ndensity_e_on_axis
    te0: temp_e_on_axis
    ne_avg: ndensity_e_vol_avg
    te_avg: temp_e_vol_avg
    f_den_ie: f_density_ion_total_electron
    f_temp_ei: f_temp_electron_ion

    """
    pres0 = Const.MILL_QE * ne0 * te0 * (1.0 + f_den_ie / f_temp_ei)
    pres = Const.MILL_QE * ne_avg * te_avg * (1.0 + f_den_ie / f_temp_ei)

    if i_dimension == 0:
        fun = lambda rho: ne(rho) * te(rho) * rho * 2

        fun1 = lambda rho: ne(rho) * te(rho) * rho

        fun2 = lambda rho: ne(rho) * rho

        f_temp_avg_density_weighted_temp_vol_avg = quad(fun1, 0, 1)[0] / quad(fun2, 0, 1)[0]

        pres_thermal_profile = lambda rho:(ne(rho) * te(rho) * pres)

        pres_thermal_vol_avg = (quad(fun, 0, 1)[0] * pres)

        pres_thermal_vol_avg_profile = lambda rho: [(quad(fun, 0, x)[0] * pres) / x**2 for x in rho]


    te_adw = f_temp_avg_density_weighted_temp_vol_avg * te_avg
    ti_adw = te_adw / f_temp_ei

    return (
        pres0, 
        f_temp_avg_density_weighted_temp_vol_avg,
        pres_thermal_profile,
        pres_thermal_vol_avg,
        pres_thermal_vol_avg_profile,
        ti_adw,
        te_adw)


def cal_pres_and_pprime_func(prof):
    ne = prof.density_func_norm_avg
    te = prof.temp_func_norm_avg
    p0 = prof.pres_thermal_plasma_on_axis
    pres = lambda rho: ne(rho) * te(rho) / (ne(0.0) * te(0.0)) * p0
    pprime = lambda rho: np.gradient(pres(rho), rho)
    return pres, pprime


def cal_beta_thermal_plasma(model):
    model.prof_data.beta_thermal_plasma_total    = 2 * Const.MU0 * model.prof_data.pres_thermal_plasma_vol_avg / model.prof_data.b_total**2
    model.prof_data.beta_thermal_plasma_toroidal = 2 * Const.MU0 * model.prof_data.pres_thermal_plasma_vol_avg / model.prof_data.b_toroidal_rmajor**2
    model.prof_data.beta_thermal_plasma_poloidal = 2 * Const.MU0 * model.prof_data.pres_thermal_plasma_vol_avg / model.prof_data.b_poloidal_avg**2
    model.prof_data.beta_thermal_plasma_norm = (100 * model.prof_data.beta_thermal_plasma_toroidal * model.geom_data.rminor
                                            * model.prof_data.b_toroidal_rmajor / model.prof_data.current_plasma_total_ma)
    p_avg = model.prof_data.pres_thermal_plasma_vol_avg_profile(model.prof_data.rho)
    p_avg = np.concatenate([np.flip(p_avg),p_avg])
    model.prof_data.beta_thermal_plasma_toroidal_profile = 2 * Const.MU0 * p_avg / model.prof_data.b_toroidal_profile**2


