import numpy as np
from scipy.integrate import quad

def plasma_geom(geom_data,i_plasma_geometry):
    # input:
    # aspect, rmajor, rminor, eps 中两个作为输入
    # i_plasma_geometry: 根据 i_plasma_geometry 的取值选择输入

    # rmajor, aspect, rminor, eps 中两个作为输入
    if geom_data.rmajor is not None and geom_data.rminor is not None:
        geom_data.aspect = geom_data.rmajor / geom_data.rminor
        geom_data.eps = 1.0 / geom_data.aspect
    else:
        if geom_data.aspect is not None:
            geom_data.eps = 1.0 / geom_data.aspect
        else:
            if geom_data.eps is None:
                raise ValueError("aspect and eps cannot be None at the same time")
            geom_data.aspect = 1.0 / geom_data.eps
        if geom_data.rmajor is not None:
            geom_data.rminor = geom_data.rmajor / geom_data.aspect
        else:
            if geom_data.rminor is None:
                raise ValueError("rminor and rmajor cannot be None at the same time")
            geom_data.rmajor = geom_data.rminor * geom_data.aspect

    
    # 根据 i_plasma_geometry 的取值计算： kappa95, delta95, kappa, delta
    match i_plasma_geometry:
        case 0:
            geom_data.kappa95=geom_data.kappa/1.12
            geom_data.delta95=geom_data.delta/1.5
        case _:
            raise ValueError("i_plasma_geometry is not valid")
    
    
    xi, thetai, xo, thetao = cal_geom_angles(geom_data.kappa,geom_data.delta,geom_data.rminor,
                                             i_plasma_geometry)
    s_in, s_out, s = cal_surface_plasma(geom_data.rmajor,geom_data.rminor,
                                        xi,xo,thetai,thetao,
                                        i_plasma_geometry)
    geom_data.length_plasma = cal_length_poloidal(xi,xo,thetai,thetao,i_plasma_geometry)
    geom_data.vol_plasma = cal_volume_plasma(geom_data.rmajor,geom_data.rminor,
                                             geom_data.kappa,geom_data.delta,
                                             xi,xo,thetai,thetao,i_plasma_geometry)
    geom_data.xi = xi
    geom_data.thetai = thetai
    geom_data.xo = xo
    geom_data.thetao = thetao
    geom_data.surface_area_plasma_inner = s_in
    geom_data.surface_area_plasma_outer = s_out
    geom_data.surface_area_plasma = s
    
    geom_data.area_cross_section_plasma = cal_area_cross_section(xi,xo,thetai,thetao,
                                                                 geom_data.kappa,geom_data.rminor,
                                                                 i_plasma_geometry)
    geom_data.kappa_ipb_scale = geom_data.vol_plasma / (
        2.0 * np.pi * geom_data.rmajor) / (np.pi * geom_data.rminor**2)
    

    
def cal_geom_angles(kappa,delta,rminor,i_plasma_geometry):
    match i_plasma_geometry:
        case 0:
            thetai=np.pi-2*np.atan(kappa/(1.0-delta))
            thetao=np.pi-2*np.atan(kappa/(1.0+delta))
            xi=rminor*kappa/np.sin(thetai)
            xo=rminor*kappa/np.sin(thetao)
        case _:
            raise ValueError("i_plasma_geometry is not valid")
    return xi,thetai,xo,thetao


def cal_surface_plasma(rmajor,rminor,xi,xo,thetai,thetao,i_plasma_geometry):
        # process 的算法
        # rc = geom_data.rmajor-geom_data.rminor+geom_data.xi
        # si = 4*pi*geom_data.xi*(rc*geom_data.thetai-geom_data.xi*sin(geom_data.thetai))
        # rc = geom_data.rmajor+geom_data.rminor-geom_data.xo
        # so = 4*pi*geom_data.xo*(rc*geom_data.thetao+geom_data.xo*sin(geom_data.thetao))
        # s=si+so

        # 积分
        # 2*pi*sum(R*dl)
        # dl**2=dR**2+dZ**2
    match i_plasma_geometry:
        case 0:
            # dl=xi*dtheta
            fun_i = lambda theta: (rmajor - rminor + xi * (1 - np.cos(theta))) * xi
            fun_o = lambda theta: (rmajor + rminor - xo * (1 - np.cos(theta))) * xo

            si,_=quad(fun_i,0,thetai)
            so,_=quad(fun_o,0,thetao)
        case _:
            raise ValueError("i_plasma_geometry is not valid")
    s=si+so
    return 4*np.pi*si,4*np.pi*so,4*np.pi*s


def cal_length_poloidal(xi,xo,thetai,thetao,i_plasma_geometry):
    match i_plasma_geometry:
        case 0:
                length_poloidal=(xi*thetai + xo*thetao)*2
        case _:
            raise ValueError("i_plasma_geometry is not valid")
    return length_poloidal

    
def cal_volume_plasma(rmajor,rminor,kappa,delta,
                   xi,xo,thetai,thetao,
                   i_plasma_geometry):
    match i_plasma_geometry:
        case 0:
            # 暂时不考虑缩放修正f_vol_plasma，公式和process中的有区别
            vol_plasma=(2*np.pi**2*kappa*(rmajor/rminor-delta)+16*np.pi*kappa*delta/3)*rminor**3
            
            # 积分
            # dR*dZ=r*dr*dtheta
            # R*dR*dZ=(rmajor-rminor+xi-r*cos(theta))*r*dr*dtheta
            # [(rmajor-rminor+xi)/2*xi**2*(1-cos(thetai)**2/cos(theta)**2)-cos(theta)/3*xi**3*(1-cos(thetai)**3/cos(theta)**3)]*dtheta
            # theta: 0 -> thetai, r: xi*cos(thetai)/cos(theta) -> xi
            fun_1 = lambda theta: ((rmajor-rminor+xi)/2*xi**2*(1-np.cos(thetai)**2/np.cos(theta)**2)
                                    -np.cos(theta)/3*xi**3*(1-np.cos(thetai)**3/np.cos(theta)**3))
            vol_1,_=quad(fun_1,0,thetai)
            fun_2 = lambda theta: ((rmajor+rminor-xo)/2*xo**2*(1-np.cos(thetao)**2/np.cos(theta)**2)
                                    +np.cos(theta)/3*xo**3*(1-np.cos(thetao)**3/np.cos(theta)**3))
            vol_2,_=quad(fun_2,0,thetao)
            vol_plasma=2*np.pi*(vol_1+vol_2)*2
        case _:
            raise ValueError("i_plasma_geometry is not valid")
    return vol_plasma

def cal_area_cross_section(xi,xo,thetai,thetao,kappa,rminor,i_plasma_geometry):
    match i_plasma_geometry:
        case 0:
            area_circle_in = xi**2 *thetai
            area_circle_out = xo**2 *thetao
            area_tri_in = xi *np.cos(thetai) *kappa*rminor
            area_tri_out = xo *np.cos(thetao) *kappa*rminor
            area_plasma = area_circle_in + area_circle_out - area_tri_in - area_tri_out
        case _:
            raise ValueError("i_plasma_geometry is not valid")
    return area_plasma
