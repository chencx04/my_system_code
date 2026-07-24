import array
from constant import mass_ion_amu
from charge_ion import charge_ion
from constant import Const
from aux_functions import AuxFunctions
import numpy as np

def plasma_component_fraction(power_data,comp_data,i_fusion_reaction):
    """
    仅计算各部分密度和电子密度的比值
    
    需要注意的输入量: (非常规输入量,需要在结构体初始化时赋值)
    alpha_produce_rate: power_data, 不作为程序输入, 由初始化的值0.0提供
    p_produce_rate: power_data, 在计算了 power 后作为输入
    f_density_h_isotope_electron: comp_data, 后续计算会更新, 默认值需为0.0
    
    其他输入量:
    f_density_proton_electron_input: 多余质子电子密度占比输入（氢硼反应时）
    i_proton_pb11_input: 是否输入多余质子与电子密度比（氢硼反应时）
    
    f_density_alpha_electron: alpha 粒子电子密度占比
    f_density_beam_electron: 中性束电子密度占比

    f_density_p_beam: 中性束中p占比
    f_density_t_beam: 中性束中T占比

    f_density_d_fuel: D燃料电子密度占比
    f_density_t_fuel: T燃料电子密度占比
    f_density_he3_fuel: He3燃料电子密度占比
    f_density_b11_fuel: B11燃料电子密度占比
    f_density_p_fuel: p燃料电子密度占比
    
    imp_name: 杂质的名称列表，字符串数组
    f_density_imp_electron: 杂质i的电子密度占比, 数组, 长度与imp_name相同


    """

    # 质子密度比例
    if i_fusion_reaction == 1:
        # D & T
        if power_data.alpha_produce_rate < 1.0e-6:
            # 第一次调用模型，还未计算聚变功率
            comp_data.f_density_proton_electron = max(
                comp_data.f_density_proton_electron_input,
                comp_data.f_density_alpha_electron * (comp_data.f_density_he3_fuel + 1.0e-3)
            )
        else:
            comp_data.f_density_proton_electron = max(
                comp_data.f_density_proton_electron_input,
                comp_data.f_density_alpha_electron * power_data.p_produce_rate / power_data.alpha_produce_rate
            )
    elif i_fusion_reaction == 2:
        # p & B11
        if comp_data.i_proton_pb11_input == 1:
            comp_data.f_density_proton_electron = comp_data.f_density_proton_electron_input
        else:
            comp_data.f_density_proton_electron = 0.0


    # 杂质 sum(n*Z)，不包括 H,D,T,He3,He4,B11 （质子 + 燃料）
    if comp_data.imp_name is not None:
        comp_data.mass_imp_amu = np.array([mass_ion_amu(name) for name in comp_data.imp_name])
        comp_data.charge_imp = np.array([charge_ion(name) for name in comp_data.imp_name])
        comp_data.f_density_imp_total_electron = np.sum(comp_data.f_density_imp_electron)
        comp_data.f_density_imp_charge_weighted_electron = np.sum(np.array(comp_data.f_density_imp_electron) * np.array(comp_data.charge_imp))
    else:
        comp_data.mass_imp_amu = []
        comp_data.charge_imp = []
        comp_data.f_density_imp_total_electron = 0.0
        comp_data.f_density_imp_charge_weighted_electron = 0.0


    # 燃料 D,T,He3,B11,p
    if i_fusion_reaction == 1:
        comp_data.f_density_fuel_charge_weighted_electron = (
            1 - 2 * comp_data.f_density_alpha_electron
            - comp_data.f_density_proton_electron
            - comp_data.f_density_beam_electron
            - comp_data.f_density_imp_charge_weighted_electron
        )
        comp_data.f_density_fuel_electron = comp_data.f_density_fuel_charge_weighted_electron / (1.0 + comp_data.f_density_he3_fuel)
    else:
        comp_data.f_density_fuel_charge_weighted_electron = (
            1 - 2 * comp_data.f_density_alpha_electron
            - comp_data.f_density_proton_electron
            - comp_data.f_density_beam_electron * (comp_data.f_density_p_beam + (1.0 - comp_data.f_density_p_beam) * charge_ion("b11"))
            - comp_data.f_density_imp_charge_weighted_electron
        )
        comp_data.f_density_fuel_electron = comp_data.f_density_fuel_charge_weighted_electron / (1.0 + comp_data.f_density_b11_fuel * 4)


    comp_data.f_density_d_fuel_electron = comp_data.f_density_fuel_electron * comp_data.f_density_d_fuel
    comp_data.f_density_t_fuel_electron = comp_data.f_density_fuel_electron * comp_data.f_density_t_fuel
    comp_data.f_density_he3_fuel_electron = comp_data.f_density_fuel_electron * comp_data.f_density_he3_fuel
    comp_data.f_density_b11_fuel_electron = comp_data.f_density_fuel_electron * comp_data.f_density_b11_fuel
    comp_data.f_density_p_fuel_electron = comp_data.f_density_fuel_electron * comp_data.f_density_p_fuel
    
    # 中性束
    if i_fusion_reaction == 1:
        comp_data.f_density_d_beam_electron = comp_data.f_density_beam_electron * (1 - comp_data.f_density_t_beam)
        comp_data.f_density_t_beam_electron = comp_data.f_density_beam_electron * comp_data.f_density_t_beam
        comp_data.f_density_p_beam_electron = 0.0
        comp_data.f_density_b11_beam_electron = 0.0
    elif i_fusion_reaction == 2:
        comp_data.f_density_p_beam_electron = comp_data.f_density_beam_electron * comp_data.f_density_p_beam
        comp_data.f_density_b11_beam_electron = comp_data.f_density_beam_electron * (1 - comp_data.f_density_p_beam)
        comp_data.f_density_d_beam_electron = 0.0
        comp_data.f_density_t_beam_electron = 0.0

    # 设置 H 使其包括 D, T, p, 中性束
    if i_fusion_reaction == 1:
        comp_data.f_density_h_isotope_electron = comp_data.f_density_proton_electron + comp_data.f_density_fuel_electron * (1 - comp_data.f_density_he3_fuel) + comp_data.f_density_beam_electron
        comp_data.f_density_b_isotope_electron = 0.0
    elif i_fusion_reaction == 2:
        comp_data.f_density_h_isotope_electron = comp_data.f_density_proton_electron + comp_data.f_density_p_fuel_electron + comp_data.f_density_p_beam_electron
        comp_data.f_density_b_isotope_electron = comp_data.f_density_b11_fuel_electron + comp_data.f_density_b11_beam_electron
    # 设置 He 使其包括 He3 和 He4
    comp_data.f_density_he_isotope_electron = comp_data.f_density_alpha_electron + comp_data.f_density_he3_fuel_electron

    # 总离子 = 燃料(D,T,He3,p,B11) + 杂质(电荷数>2) + 氦灰(alpha,He4) + 中性束(D,T)或(p,B11) + 质子(H,p)
    comp_data.f_density_ion_total_electron = (
        comp_data.f_density_fuel_electron
        + comp_data.f_density_imp_total_electron
        + comp_data.f_density_alpha_electron
        + comp_data.f_density_beam_electron
        + comp_data.f_density_proton_electron
    )


    # 杂质特定元素查找
    if comp_data.imp_name is not None:
        try:
            idx = comp_data.imp_name.index("c")
            comp_data.f_density_imp_c_electron = comp_data.f_density_imp_electron[idx]
        except ValueError:
            comp_data.f_density_imp_c_electron = 0.0

        try:
            idx = comp_data.imp_name.index("o")
            comp_data.f_density_imp_o_electron = comp_data.f_density_imp_electron[idx]
        except ValueError:
            comp_data.f_density_imp_o_electron = 0.0

        try:
            idx = comp_data.imp_name.index("fe")
            comp_data.f_density_imp_fe_ar_electron = comp_data.f_density_imp_electron[idx]
        except ValueError:
            comp_data.f_density_imp_fe_ar_electron = 0.0

        try:
            idx = comp_data.imp_name.index("ar")
            comp_data.f_density_imp_fe_ar_electron += comp_data.f_density_imp_electron[idx]
        except ValueError:
            pass
    else:
        comp_data.f_density_imp_c_electron = 0.0
        comp_data.f_density_imp_o_electron = 0.0
        comp_data.f_density_imp_fe_ar_electron = 0.0


    comp_data.charge_eff_total_ion = (
        comp_data.f_density_h_isotope_electron * 1 
        + comp_data.f_density_he_isotope_electron * 4
        + (comp_data.f_density_b11_fuel_electron + comp_data.f_density_b11_beam_electron) * 25
    )
    if comp_data.imp_name is not None:
        comp_data.charge_eff_total_ion += np.sum(np.array(comp_data.f_density_imp_electron) * np.array(comp_data.charge_imp)**2)

    # alpha粒子的能量分配给离子和电子的比例，仅用于电子和离子平衡

    # 单粒子平均质量
    comp_data.m_fuel_avg_amu = (
        mass_ion_amu("d") * comp_data.f_density_d_fuel
        + mass_ion_amu("t") * comp_data.f_density_t_fuel
        + mass_ion_amu("he3") * comp_data.f_density_he3_fuel
        + mass_ion_amu("b11") * comp_data.f_density_b11_fuel
        + mass_ion_amu("p") * comp_data.f_density_p_fuel
    )
    
    if i_fusion_reaction == 1:
        comp_data.m_beam_avg_amu = (
            mass_ion_amu("d") * (1 - comp_data.f_density_t_beam)
            + mass_ion_amu("t") * comp_data.f_density_t_beam
        )
    elif i_fusion_reaction == 2:
        comp_data.m_beam_avg_amu = (
            mass_ion_amu("p") * comp_data.f_density_p_beam
            + mass_ion_amu("b11") * (1 - comp_data.f_density_p_beam)
        )

    comp_data.m_ion_total_avg_amu = (
        comp_data.m_fuel_avg_amu * comp_data.f_density_fuel_electron
        + mass_ion_amu("he4") * comp_data.f_density_alpha_electron
        + mass_ion_amu("proton") * comp_data.f_density_proton_electron
        + comp_data.m_beam_avg_amu * comp_data.f_density_beam_electron
    ) / comp_data.f_density_ion_total_electron
    if comp_data.imp_name is not None:
        comp_data.m_ion_total_avg_amu += np.sum(np.array(comp_data.f_density_imp_electron) * np.array(comp_data.mass_imp_amu)) / comp_data.f_density_ion_total_electron


def cal_composition_density(comp_data, ndensity_e_vol_avg, vol_plasma):
    """
    计算组成相关的密度值，更新 comp_data 结构体

    输入：
        comp_data 中包含以下输入变量：
            f_density_..._electron: 各部分电子密度占比
            charge_imp: 杂质电荷数
            mass_imp_amu: 杂质质量
        ndensity_e_vol_avg: 电子密度体积平均值
        vol_plasma: 等离子体体积
    """

    comp_data.ndensity_alpha_vol_avg = comp_data.f_density_alpha_electron * ndensity_e_vol_avg
    comp_data.ndensity_proton_vol_avg = comp_data.f_density_proton_electron * ndensity_e_vol_avg
    comp_data.ndensity_beam_ion_vol_avg = comp_data.f_density_beam_electron * ndensity_e_vol_avg 
    comp_data.ndensity_d_beam_vol_avg = comp_data.f_density_d_beam_electron * ndensity_e_vol_avg 
    comp_data.ndensity_t_beam_vol_avg = comp_data.f_density_t_beam_electron * ndensity_e_vol_avg  
    comp_data.ndensity_p_beam_vol_avg = comp_data.f_density_p_beam_electron * ndensity_e_vol_avg  
    comp_data.ndensity_b11_beam_vol_avg = comp_data.f_density_b11_beam_electron * ndensity_e_vol_avg  
    
    # impurity density
    comp_data.ndensitycharge_imp_total_vol_avg = comp_data.f_density_imp_charge_weighted_electron * ndensity_e_vol_avg
    comp_data.ndensity_imp_total_vol_avg = comp_data.f_density_imp_total_electron * ndensity_e_vol_avg
    if comp_data.imp_name is not None:
        comp_data.ndensity_imp_vol_avg = np.array(comp_data.f_density_imp_electron) * ndensity_e_vol_avg
    else:
        comp_data.ndensity_imp_vol_avg = 0.0
    comp_data.ndensity_imp_c_vol_avg = comp_data.f_density_imp_c_electron * ndensity_e_vol_avg
    comp_data.ndensity_imp_o_vol_avg = comp_data.f_density_imp_o_electron * ndensity_e_vol_avg
    comp_data.ndensity_imp_fe_ar_vol_avg = comp_data.f_density_imp_fe_ar_electron * ndensity_e_vol_avg

    # fuel density
    comp_data.ndensitycharge_fuel_vol_avg = comp_data.f_density_fuel_charge_weighted_electron * ndensity_e_vol_avg
    comp_data.ndensity_fuel_vol_avg = comp_data.f_density_fuel_electron * ndensity_e_vol_avg
    comp_data.ndensity_d_fuel_vol_avg = comp_data.f_density_d_fuel_electron * ndensity_e_vol_avg
    comp_data.ndensity_t_fuel_vol_avg = comp_data.f_density_t_fuel_electron * ndensity_e_vol_avg
    comp_data.ndensity_he3_fuel_vol_avg = comp_data.f_density_he3_fuel_electron * ndensity_e_vol_avg
    comp_data.ndensity_b11_fuel_vol_avg = comp_data.f_density_b11_fuel_electron * ndensity_e_vol_avg
    comp_data.ndensity_p_fuel_vol_avg = comp_data.f_density_p_fuel_electron * ndensity_e_vol_avg

    # 同位素密度
    comp_data.ndensity_h_isotope_vol_avg = comp_data.f_density_h_isotope_electron * ndensity_e_vol_avg
    comp_data.ndensity_he_isotope_vol_avg = comp_data.f_density_he_isotope_electron * ndensity_e_vol_avg
    comp_data.ndensity_b_isotope_vol_avg = comp_data.f_density_b_isotope_electron * ndensity_e_vol_avg

    # mass
    vol_amu = vol_plasma * Const.AMU
    comp_data.m_fuel_total_kg = comp_data.m_fuel_avg_amu * comp_data.ndensity_fuel_vol_avg * vol_amu
    comp_data.m_ion_total_kg = comp_data.m_ion_total_avg_amu * ndensity_e_vol_avg * comp_data.f_density_ion_total_electron * vol_amu
    comp_data.m_alpha_total_kg = mass_ion_amu("alpha") * comp_data.ndensity_alpha_vol_avg * vol_amu
    comp_data.m_electron_total_kg = ndensity_e_vol_avg * vol_plasma * Const.MASS_ELECTRON_KG
    comp_data.m_plasma_kg = comp_data.m_ion_total_kg + comp_data.m_electron_total_kg

    comp_data.charge_eff_mass_weighted = ( comp_data.ndensity_proton_vol_avg * 1 / mass_ion_amu("proton")
                                        + comp_data.ndensity_alpha_vol_avg * 4 / mass_ion_amu("alpha")
                                        + comp_data.ndensity_d_fuel_vol_avg * 1 / mass_ion_amu("d")
                                        + comp_data.ndensity_t_fuel_vol_avg * 1 / mass_ion_amu("t")
                                        + comp_data.ndensity_he3_fuel_vol_avg * 4 / mass_ion_amu("he3")
                                        + comp_data.ndensity_p_fuel_vol_avg * 1 / mass_ion_amu("proton")
                                        + comp_data.ndensity_b11_fuel_vol_avg * 25 / mass_ion_amu("b11")
                                        + comp_data.ndensity_d_beam_vol_avg * 1 / mass_ion_amu("d")
                                        + comp_data.ndensity_t_beam_vol_avg * 1 / mass_ion_amu("t")
                                        + comp_data.ndensity_p_beam_vol_avg * 1 / mass_ion_amu("proton")
                                        + comp_data.ndensity_b11_beam_vol_avg * 25 / mass_ion_amu("b11")
                                        )

    if comp_data.imp_name is not None:
        comp_data.charge_eff_mass_weighted += np.sum(comp_data.ndensity_imp_vol_avg * (comp_data.charge_imp ** 2) / comp_data.mass_imp_amu)
