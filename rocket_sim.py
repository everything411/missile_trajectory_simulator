# SPDX-License-Identifier: MIT

import numpy as np
from numba import njit
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 对象类型
OBJ_TYPE_ROCKET = 0  # 主火箭
OBJ_TYPE_STAGE = 1  # 废弃级
OBJ_TYPE_SAT = 2  # 卫星/载荷

# 动力模式
GUIDANCE_MODE_LAUNCH = 0.0  # 发射模式 (垂直 -> 程序转弯 -> 重力转弯)
GUIDANCE_MODE_PROGRADE = 1.0  # 顺向模式 (始终对准速度方向，用于变轨)
GUIDANCE_MODE_PASSIVE = -1.0  # 自由运动，无升力

# 一些常量
R_EARTH = 6378137.0
R_POLAR = 6356752.0
GM = 3.986004418e14
OMEGA_E = 7.292115e-5
J2 = 1.08263e-3
G0 = 9.80665

# 鼻锥类型映射
NC_TYPE_CONICAL = 0
NC_TYPE_OGIVE = 1
NC_TYPE_PARABOLIC = 2
NC_TYPE_ELLIPTICAL = 3
NC_TYPE_SEARS_HAACK = 4
NC_TYPE_V2 = 5
NC_TYPE_UNKNOWN = -1

# RKF45 系数
RK45_C = np.array([0.0, 1 / 5, 3 / 10, 3 / 5, 1.0, 7 / 8], dtype=np.float64)
RK45_A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1 / 5, 0.0, 0.0, 0.0, 0.0],
        [3 / 40, 9 / 40, 0.0, 0.0, 0.0],
        [3 / 10, -9 / 10, 6 / 5, 0.0, 0.0],
        [-11 / 54, 5 / 2, -70 / 27, 35 / 27, 0.0],
        [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
    ],
    dtype=np.float64,
)
RK45_B = np.array(
    [37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771], dtype=np.float64
)
RK45_B_STAR = np.array(
    [2825 / 27648, 0.0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4],
    dtype=np.float64,
)


# 大气层数据表
ATMOS_LAYERS = np.array(
    [
        [0.0, -0.0065],
        [11000.0, 0.0],
        [20000.0, 0.0010],
        [32000.0, 0.0028],
        [47000.0, 0.0],
        [51000.0, -0.0028],
        [71000.0, -0.0020],
        [84852.0, 0.0],
    ],
    dtype=np.float64,
)

# [Base Alt, Lapse Rate, Base Temp, Base Pressure]
HIGH_ALT_DATA = np.array(
    [
        [86.0, 186.87, 0.3734, 3.426e-6],
        [90.0, 186.87, 0.1836, 3.416e-6],
        [100.0, 195.08, 3.201e-2, 5.604e-7],
        [110.0, 240.0, 7.104e-3, 9.661e-8],
        [120.0, 360.0, 2.538e-3, 2.222e-8],
        [130.0, 525.0, 1.250e-3, 8.152e-9],
        [140.0, 680.0, 7.203e-4, 3.831e-9],
        [150.0, 893.0, 4.542e-4, 2.076e-9],
        [160.0, 1022.0, 3.033e-4, 1.233e-9],
        [180.0, 1169.0, 1.487e-4, 5.194e-10],
        [200.0, 1236.0, 8.474e-5, 2.541e-10],
        [250.0, 1311.0, 2.477e-5, 6.073e-11],
        [300.0, 1339.0, 8.770e-6, 1.916e-11],
        [350.0, 1354.0, 3.483e-6, 6.784e-12],
        [400.0, 1365.0, 1.498e-6, 2.621e-12],
        [450.0, 1374.0, 6.887e-7, 1.085e-12],
        [500.0, 1381.0, 3.325e-7, 4.764e-13],
        [600.0, 1391.0, 8.653e-8, 1.036e-13],
        [700.0, 1396.0, 2.508e-8, 2.520e-14],
        [800.0, 1399.0, 7.859e-9, 6.735e-15],
        [900.0, 1400.0, 2.583e-9, 1.905e-15],
        [1000.0, 1400.0, 8.784e-10, 5.642e-16],
    ],
    dtype=np.float64,
)


@dataclass
class SubSatelliteConfig:
    name_pattern: str = "Sat_{i}"  # 命名模板
    count: int = 0  # 数量
    mass: float = 10.0  # 单颗质量 (kg)
    release_start_time: float = 0.0  # 相对于入轨/KickStage生成的延时 (s)
    release_interval: float = 30.0  # 释放间隔 (s)
    separation_velocity: float = 0.5  # 弹簧分离速度 (m/s)


@dataclass
class KickStageConfig:
    enabled: bool = False
    dry_mass: float = 100.0  # 上面级结构干重 (kg)，不含子卫星
    fuel_mass: float = 200.0  # 总燃料质量 (kg)，包含入轨+离轨所需
    thrust: float = 5000.0  # 发动机推力 (N)
    isp: float = 300.0  # 比冲 (s)
    ignition_delay: float = 0.0  # 点火延迟时间 (s)，0表示自动计算

    payloads: List[SubSatelliteConfig] = field(default_factory=list)  # 子卫星载荷配置

    deorbit_enabled: bool = False  # 是否启用反推
    deorbit_fuel_mass: float = 20.0  # 预留给反推的燃料 (kg)
    deorbit_delay: float = 60.0  # 最后一颗卫星释放后多久开始反推 (s)

    @property
    def total_payload_mass(self):
        """计算所有子卫星的总质量"""
        m = 0.0
        for p in self.payloads:
            m += p.count * p.mass
        return m


@dataclass
class PayloadReleaseEvent:
    release_time: float
    mass: float
    name: str
    sep_vel: float
    direction_type: int = 0  # 0=Prograde, 3=Random


@dataclass
class TrackedObject:
    name: str
    obj_type: int
    state: np.ndarray  # [rx, ry, rz, vx, vy, vz, m]

    # 物理属性
    drag_area: float
    drag_coeff_type: int  # 0=Active(Calc), 1=Fixed
    fixed_cd: float = 0.5  # 碎片默认阻力系数

    active: bool = True

    # 动力学属性
    stage_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 5))
    )  # [thrust, dmdt, t_s, t_e]

    # 制导与载荷属性
    custom_guidance: Optional[np.ndarray] = None  # [vert, pitch, aoa, MODE]
    release_events: List[PayloadReleaseEvent] = field(default_factory=list)
    is_main: bool = False

    # 历史记录
    history: List[Dict] = field(default_factory=list)


@dataclass
class FinConfig:
    number: int = 0
    height: float = 0.0
    root_chord: float = 0.0
    tip_chord: float = 0.0
    sweep_angle: float = 0.0
    thickness: float = 0.0


@dataclass
class RocketStage:
    fuel_mass: float  # kg
    dry_mass: float  # kg
    isp: float  # s
    thrust: float  # N 真空推力

    nozzle_area: float = 0.0
    burn_time: float = 0.0
    dmdt: float = 0.0

    def __post_init__(self):
        if self.thrust > 0 and self.isp > 0:
            self.dmdt = self.thrust / (self.isp * 9.80665)
            self.burn_time = self.fuel_mass / self.dmdt
        else:
            self.dmdt = 0
            self.burn_time = 0


@dataclass
class SimConfig:
    stages: List[RocketStage]
    payload_mass: float
    rocket_diameter: float  # 直径 (m)
    nosecone_type: str = "Conical"
    nosecone_ld_ratio: float = 3.0
    reentry_diameter: float = 0.0  # 再入直径 (0表示不启用)
    fins: FinConfig = field(default_factory=FinConfig)
    fairing_mass: float = 0.0  # 整流罩总质量 (kg)
    fairing_sep_time: float = 0.0  # 分离时间 (绝对时间 s, 0表示不启用)

    # --- 3D 发射参数 ---
    launch_lat: float = 0  # 发射纬度 (Degrees), 默认0
    launch_lon: float = 0  # 发射经度
    launch_azimuth: float = 90.0  # 发射方位角 (0=North, 90=East)

    # --- 制导参数 ---
    vertical_time: float = 5.0  # 垂直爬升时间
    pitch_over_angle: float = 2.0  # 初始程序转弯偏转角度 (deg)
    guidance_aoa: float = 0.0  # 设定的飞行攻角 (用于产生升力)

    kick_stage: KickStageConfig = field(default_factory=KickStageConfig)


@njit(cache=True)
def get_nc_type_id(name: str) -> int:
    name = name.lower()
    if "conical" in name:
        return NC_TYPE_CONICAL
    if "ogive" in name:
        return NC_TYPE_OGIVE
    if "parabolic" in name:
        return NC_TYPE_PARABOLIC
    if "elliptical" in name:
        return NC_TYPE_ELLIPTICAL
    if "sears" in name:
        return NC_TYPE_SEARS_HAACK
    if "v2" in name:
        return NC_TYPE_V2
    return NC_TYPE_CONICAL


@njit(cache=True)
def lla_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    a = R_EARTH
    e2 = 1 - (R_POLAR / a) ** 2
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    return np.array([x, y, z])


@njit(cache=True)
def eci_to_lla(r_eci: np.ndarray, time: float) -> tuple:
    # 旋转 ECI 到 ECEF
    theta = OMEGA_E * time
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x_eci, y_eci, z_eci = r_eci
    x = cos_t * x_eci + sin_t * y_eci
    y = -sin_t * x_eci + cos_t * y_eci
    z = z_eci

    p = np.sqrt(x**2 + y**2)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1 - 1 / 298.257))
    h = 0.0
    a = R_EARTH
    e2 = 1 - (R_POLAR / a) ** 2

    for _ in range(5):
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        if np.abs(lat) < np.radians(85.0):
            h = p / cos_lat - N
        else:
            h = z / sin_lat - N * (1 - e2)
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))

    return np.degrees(lat), np.degrees(lon), h


@njit(cache=True)
def atmosphere(altitude_m: float) -> tuple:
    if altitude_m < 0:
        altitude_m = 0

    # Part 1: < 86km
    if altitude_m < 86000.0:
        g0 = 9.80665
        R = 287.05287
        R_earth = 6356766.0
        h = (R_earth * altitude_m) / (R_earth + altitude_m)

        p_curr = 101325.0
        t_curr = 288.15
        h_base = 0.0

        for i in range(len(ATMOS_LAYERS) - 1):
            h_bottom = ATMOS_LAYERS[i, 0]
            lapse = ATMOS_LAYERS[i, 1]
            h_top = ATMOS_LAYERS[i + 1, 0]

            if h > h_top:
                h_target = h_top
                segment_end = False
            else:
                h_target = h
                segment_end = True

            delta_h = h_target - h_base
            t_next = t_curr + lapse * delta_h

            if abs(lapse) < 1e-10:
                p_next = p_curr * np.exp(-g0 * delta_h / (R * t_curr))
            else:
                exponent = -g0 / (R * lapse)
                p_next = p_curr * (t_next / t_curr) ** exponent

            if segment_end:
                t_curr = t_next
                p_curr = p_next
                break
            else:
                t_curr = t_next
                p_curr = p_next
                h_base = h_top

        rho = p_curr / (R * t_curr)
        return rho, t_curr, p_curr

    # Part 2: High Alt
    else:
        z_km = altitude_m / 1000.0
        if z_km >= 1000.0:
            return 5.642e-16, 1400.0, 8.784e-10

        rho, temp_k, pressure_pa = 0.0, 0.0, 0.0

        for i in range(len(HIGH_ALT_DATA) - 1):
            z1 = HIGH_ALT_DATA[i, 0]
            z2 = HIGH_ALT_DATA[i + 1, 0]

            if z1 <= z_km <= z2:
                # Unpack row data
                t1, p1, rho1 = (
                    HIGH_ALT_DATA[i, 1],
                    HIGH_ALT_DATA[i, 2],
                    HIGH_ALT_DATA[i, 3],
                )
                t2, p2, rho2 = (
                    HIGH_ALT_DATA[i + 1, 1],
                    HIGH_ALT_DATA[i + 1, 2],
                    HIGH_ALT_DATA[i + 1, 3],
                )

                frac = (z_km - z1) / (z2 - z1)
                temp_k = t1 + (t2 - t1) * frac

                log_p = np.log(p1) + frac * (np.log(p2) - np.log(p1))
                pressure_pa = np.exp(log_p)

                log_rho = np.log(rho1) + frac * (np.log(rho2) - np.log(rho1))
                rho = np.exp(log_rho)
                return rho, temp_k, pressure_pa

    return 0.0, 0.0, 0.0


@njit(cache=True)
def calculate_kepler_elements(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """
    输入: ECI 位置 (m), ECI 速度 (m/s)
    输出: [a, e, i, raan, arg_p, true_anom, period]
    单位: a(km), i/raan/arg_p/nu(deg), period(min)
    """

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # 0. 基础向量
    h_vec = np.cross(r_vec, v_vec)  # 角动量向量
    h = np.linalg.norm(h_vec)

    # 节点向量 n = k x h
    # k = [0, 0, 1]
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    n = np.linalg.norm(n_vec)

    # 偏心率向量
    # e = (1/mu) * [ (v^2 - mu/r)*r - (r.v)*v ]
    tmp_val = v**2 - GM / r
    e_vec = (tmp_val * r_vec - np.dot(r_vec, v_vec) * v_vec) / GM
    e = np.linalg.norm(e_vec)

    # 1. 能量与半长轴
    # specific energy = v^2/2 - mu/r
    energy = 0.5 * v**2 - GM / r

    if abs(energy) < 1e-9:
        a = np.inf  # 抛物线
    else:
        a = -GM / (2 * energy)

    # 2. 倾角
    # cos(i) = h_z / h
    cos_i = h_vec[2] / h
    cos_i_clipped = min(max(cos_i, -1.0), 1.0)
    inc_rad = np.arccos(cos_i_clipped)

    # 3. 升交点赤经
    # cos(O) = n_x / n
    if n < 1e-9:
        raan_rad = 0.0  # 赤道轨道，未定义
    else:
        cos_raan = n_vec[0] / n
        cos_raan_clipped = min(max(cos_raan, -1.0), 1.0)
        raan_rad = np.arccos(cos_raan_clipped)
        if n_vec[1] < 0:
            raan_rad = 2 * np.pi - raan_rad

    # 4. 近地点幅角
    # cos(w) = n.e / (|n|*|e|)
    if n < 1e-9 or e < 1e-9:
        arg_p_rad = 0.0
    else:
        cos_w = np.dot(n_vec, e_vec) / (n * e)
        cos_w_clipped = min(max(cos_w, -1.0), 1.0)
        arg_p_rad = np.arccos(cos_w_clipped)
        if e_vec[2] < 0:
            arg_p_rad = 2 * np.pi - arg_p_rad

    # 5. 真近点角
    # cos(v) = e.r / (|e|*|r|)
    if e < 1e-9:
        true_anom_rad = 0.0
    else:
        cos_nu = np.dot(e_vec, r_vec) / (e * r)
        cos_nu_clipped = min(max(cos_nu, -1.0), 1.0)
        true_anom_rad = np.arccos(cos_nu_clipped)
        if np.dot(r_vec, v_vec) < 0:
            true_anom_rad = 2 * np.pi - true_anom_rad

    # 6. 轨道周期
    # T = 2*pi * sqrt(a^3 / mu)
    if a > 0 and not np.isinf(a):
        period_s = 2 * np.pi * np.sqrt(a**3 / GM)
        period_min = period_s / 60.0
    else:
        period_min = 0.0  # 双曲线或抛物线

    # 转换为角度和 km
    return np.array(
        [
            a / 1000.0,  # a (km)
            e,  # e
            np.degrees(inc_rad),  # i (deg)
            np.degrees(raan_rad),  # RAAN (deg)
            np.degrees(arg_p_rad),  # ArgP (deg)
            np.degrees(true_anom_rad),  # Nu (deg)
            period_min,  # T (min)
        ]
    )


@njit(cache=True)
def estimate_time_to_apoapsis(r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    """
    估算从当前状态滑行到远地点所需的时间（秒）
    注意：仅适用于椭圆轨道 (0 <= e < 1)
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    mu = GM

    # 1. 计算半长轴 a
    spec_energy = 0.5 * v**2 - mu / r
    if abs(spec_energy) < 1e-9:
        return 0.0  # 抛物线
    a = -mu / (2 * spec_energy)

    if a < 0:
        return 0.0  # 双曲线

    # 2. 计算偏心率 e
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    tmp = v**2 - mu / r
    e_vec = (tmp * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    if e >= 1.0 or e < 1e-6:
        return 0.0

    # 3. 计算当前偏近点角
    # cos(E) = (a - r) / (a * e)
    cos_E = (a - r) / (a * e)
    cos_E = min(max(cos_E, -1.0), 1.0)
    E_rad = np.arccos(cos_E)

    # 判断 E 是在 0~PI 还是 PI~2PI
    # 如果 r.v > 0，说明正在远离近地点 (0 < E < PI)
    # 如果 r.v < 0，说明正在接近近地点 (PI < E < 2PI)
    if np.dot(r_vec, v_vec) < 0:
        E_rad = 2 * np.pi - E_rad

    # 4. 计算当前平均近点角
    M_rad = E_rad - e * np.sin(E_rad)

    # 5. 计算平均角速度
    n = np.sqrt(mu / a**3)

    # 6. 计算到达远地点的时间
    # 远地点的 M = PI
    # 如果当前 M < PI，时间 = (PI - M) / n
    # 如果当前 M > PI，时间 = (3PI - M) / n (下一圈)

    target_M = np.pi
    if M_rad > np.pi:
        target_M = 3 * np.pi

    delta_t = (target_M - M_rad) / n

    return delta_t


@njit(cache=True)
def get_cd(
    mach: float,
    h: float,
    velocity: float,
    nc_type_id: int,
    nc_ld: float,
    fin_data: np.ndarray,
    rocket_diam: float,
) -> float:
    """
    HyperCFD 拟合
    fin_data format: [number, height, root_chord, tip_chord, sweep_angle, thickness]
    """
    if velocity < 0.1:
        velocity = 0.1
    if mach < 0.01:
        mach = 0.01

    # --- Fins ---
    cd_fins = 0.0
    fin_num = int(fin_data[0])

    if fin_num >= 1:
        fh = fin_data[1]
        cr_m = fin_data[2]
        ct_m = fin_data[3]
        tf_m = fin_data[5]

        lflm = 0.5 * (ct_m + cr_m)
        afep = 0.5 * (cr_m + ct_m) * fh
        afp = afep + 0.5 * cr_m * rocket_diam

        rho, _, _ = atmosphere(h)
        mu = 1.789e-5
        reynolds = rho * velocity * lflm / mu
        if reynolds <= 1.0:
            reynolds = 1.0

        reynolds_crit = 500000.0
        if reynolds < reynolds_crit:
            cf = 1.328 / np.sqrt(reynolds)
        else:
            fin_bt = reynolds**0.2
            fin_b_term = reynolds_crit * (
                (0.074 / (reynolds_crit**0.2)) - (1.328 / np.sqrt(reynolds_crit))
            )
            cf = (0.074 / fin_bt) - (fin_b_term / reynolds)

        thickness_correction = 1.0 + 2.0 * (tf_m / lflm)
        cd_fins = (
            2.0
            * cf
            * thickness_correction
            * (fin_num * afp * 4.0)
            / (np.pi * rocket_diam**2)
        )

    # --- Body ---
    cd_body = 0.0
    ld = nc_ld

    if nc_type_id == NC_TYPE_V2:
        if mach > 5:
            cd_body = 0.15
        elif mach > 1.8:
            cd_body = -0.03125 * mach + 0.30625
        elif mach > 1.2:
            cd_body = -0.25 * mach + 0.7
        elif mach > 0.8:
            cd_body = 0.625 * mach - 0.35
        else:
            cd_body = 0.15
        cd_body -= cd_fins

    elif nc_type_id == NC_TYPE_ELLIPTICAL:
        if mach >= 1.2:
            cal = 0.824584774 * (ld**-0.532619017)
            cbl = 1.0156845
            ccl = -0.226354 - 0.238389 * np.log(ld)
            cd_body = (cal * (cbl**mach)) * (mach**ccl)
        elif mach >= 1.05:
            m_conic = 1 / (-0.2383263138 - 0.266070229318 * ld)
            b_conic = 1 / (0.15266 + 0.160535 * ld)
            cd_body = m_conic * mach + b_conic
        else:
            cd_body = -0.05 * ld + 0.25

    elif nc_type_id == NC_TYPE_CONICAL:
        if mach > 1.5:
            aconic = 1.619038033 * np.exp(-1.31926217 * ld)
            bconic = ld / (-0.45318 - 0.89392 * ld)
            cconic = 0.886118 * np.exp(-ld / 1.121185)
            denom = 1 + bconic * np.exp(-cconic * mach)
            if denom != 0:
                cd_body = aconic / denom
        elif mach >= 1.05:
            m_conic = 1 / (-0.10823 - 0.81349 * ld)
            b_conic = 1 / (0.054882 + 0.363845 * ld)
            cd_body = m_conic * mach + b_conic
        else:
            cd_body = -0.075 * ld + 0.275

    elif nc_type_id == NC_TYPE_OGIVE:
        if mach >= 1.2:
            apara = 0.278184983 * np.exp(ld**-0.8894687916)
            bpara = 1.0129458
            cpara = -0.604615023 / (1.0 + 9.5779826 * np.exp(-2.2080809 * ld))
            cd_body = apara * (bpara**mach) * (mach**cpara)
        elif mach >= 1.05:
            mpara = 1.0 / (-0.156531249 - 0.35165656 * ld)
            bpara = 1.0 / (0.10668068 + 0.2160142549 * ld)
            cd_body = mpara * mach + bpara
        else:
            cd_body = -0.075 * ld + 0.275

    elif nc_type_id == NC_TYPE_PARABOLIC:
        if mach >= 1.2:
            apara = 0.2433566382 * np.exp(ld**-7.1807129)
            bpara = 1.009709
            cpara = -0.567521484056 / (
                1.0 + 5.59560038938568 * np.exp(-2.23635526782648 * ld)
            )
            cd_body = apara * (bpara**mach) * (mach**cpara)
        elif mach >= 1.05:
            mpara = 1.0 / (-0.1595385088 - 0.41826608398336 * ld)
            bpara = 1.0 / (0.123489761128 + 0.244747303231711 * ld)
            cd_body = mpara * mach + bpara
        else:
            cd_body = -0.025 * ld + 0.125

    elif nc_type_id == NC_TYPE_SEARS_HAACK:
        if mach >= 1.2:
            apara = 0.243884345 * np.exp(ld**-0.80690309)
            bpara = 1.0047095
            cpara = -0.60330669 / (1.0 + 14.6196741884 * np.exp(-3.27801239521 * ld))
            cd_body = apara * (bpara**mach) * (mach**cpara)
        elif mach >= 1.05:
            mpara = 1.0 / (-0.111417758 - 0.436291862 * ld)
            bpara = 1.0 / (0.090907066 + 0.26210278132 * ld)
            cd_body = mpara * mach + bpara
        else:
            cd_body = -0.05 * ld + 0.25

    elif nc_type_id == NC_TYPE_UNKNOWN:
        cd_body = 0.5

    return cd_body + cd_fins


@njit(cache=True)
def calculate_gravity(r_vec: np.ndarray) -> np.ndarray:
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.zeros(3)

    x, y, z = r_vec
    z2 = z**2
    r2 = r_mag**2
    factor = 1.5 * J2 * (R_EARTH / r_mag) ** 2

    gx = -GM * x / r_mag**3 * (1 - factor * (5 * z2 / r2 - 1))
    gy = -GM * y / r_mag**3 * (1 - factor * (5 * z2 / r2 - 1))
    gz = -GM * z / r_mag**3 * (1 - factor * (5 * z2 / r2 - 3))
    return np.array([gx, gy, gz])


@njit(cache=True)
def physics_kernel(
    t: float,
    state: np.ndarray,
    stage_matrix: np.ndarray,
    fin_data: np.ndarray,
    geo_params: np.ndarray,  # [launch_az, diam, reentry_dia, nc_ld]
    guidance_params: np.ndarray,  # [vert_time, pitch, aoa, MODE]
    nc_type_id: int,
    drag_props: np.ndarray,  # [drag_coeff_type, fixed_cd]
) -> np.ndarray:

    r = state[0:3]
    v = state[3:6]
    m = state[6]

    # Unpack geometric params
    launch_az = geo_params[0]
    rocket_dia = geo_params[1]
    reentry_dia = geo_params[2]
    nc_ld = geo_params[3]

    # 1. Environment
    _, _, altitude = eci_to_lla(r, t)
    rho, temp_k, pressure = atmosphere(altitude)
    c_sound = np.sqrt(1.4 * 287.0 * temp_k) if temp_k > 0 else 300.0

    # 2. Relative Velocity
    omega_vec = np.array([0.0, 0.0, OMEGA_E])
    v_wind = np.cross(omega_vec, r)
    v_rel = v - v_wind
    v_rel_mag = np.linalg.norm(v_rel)
    mach = v_rel_mag / c_sound

    # 3. Identify Stage
    thrust_val = 0.0
    dmdt_val = 0.0
    is_burning = False

    # stage_matrix shape: (N, 4) -> [thrust, dmdt, t_start, t_end]
    for i in range(stage_matrix.shape[0]):
        # 增加质量检查：如果质量已经极小(燃料耗尽)，不再产生推力
        if stage_matrix[i, 2] <= t < stage_matrix[i, 3]:
            # 只有当仍有足够质量时才产生推力
            if m > 1.0:
                thrust_vac = stage_matrix[i, 0]
                dmdt_val = stage_matrix[i, 1]
                stage_area = stage_matrix[i, 4] 
                
                # F_actual = F_vac - A_exit * P_ambient
                thrust_val = thrust_vac - stage_area * pressure

                if thrust_val < 0:
                    thrust_val = 0.0
                    
                is_burning = True
            break

    # 4. Orientation & Guidance
    guidance_mode = guidance_params[3]

    # 基础方向向量
    r_mag = np.linalg.norm(r)
    if r_mag > 1e-3:
        up = r / r_mag
    else:
        up = np.array([0.0, 0.0, 1.0])

    orientation = np.zeros(3)

    # === MODE 0: LAUNCH (发射程序) ===
    if guidance_mode == GUIDANCE_MODE_LAUNCH:
        ez = np.array([0.0, 0.0, 1.0])
        east = np.cross(ez, up)
        norm_east = np.linalg.norm(east)
        if norm_east > 1e-9:
            east = east / norm_east
        else:
            east = np.array([1.0, 0.0, 0.0])

        north = np.cross(up, east)

        az_rad = np.radians(launch_az)
        launch_dir_h = np.sin(az_rad) * east + np.cos(az_rad) * north

        vert_time = guidance_params[0]
        pitch_angle = guidance_params[1]
        guidance_aoa = guidance_params[2]

        if t < vert_time:
            orientation = up
        elif t < vert_time + 10.0:
            ratio = (t - vert_time) / 10.0
            tilt_angle = np.radians(pitch_angle) * ratio
            orientation = np.cos(tilt_angle) * up + np.sin(tilt_angle) * launch_dir_h
            # Normalize to be safe
            norm_o = np.linalg.norm(orientation)
            if norm_o > 0:
                orientation = orientation / norm_o
            else:
                orientation = up
        else:
            # Gravity Turn / AoA phase
            if v_rel_mag > 1.0:
                vel_dir = v_rel / v_rel_mag
                # 如果设定了 AoA 且不为 0
                if abs(guidance_aoa) > 0.001:
                    # 计算轨迹法向量 (Trajectory Normal) 用于施加 AoA
                    # 这里简化为在垂直平面内施加
                    traj_normal = np.cross(v_rel, -up)
                    tn_norm = np.linalg.norm(traj_normal)

                    if tn_norm > 1e-9:
                        traj_normal /= tn_norm
                        aoa_rad = np.radians(guidance_aoa)
                        # 在速度方向的基础上，向法向偏转 AoA 角度
                        # Cross(Normal, Vel) gives the 'Lift' direction usually
                        lift_plane_vec = np.cross(traj_normal, vel_dir)
                        orientation = vel_dir * np.cos(
                            aoa_rad
                        ) + lift_plane_vec * np.sin(aoa_rad)
                    else:
                        orientation = vel_dir
                else:
                    orientation = vel_dir
            else:
                orientation = up

    # === MODE 1: PROGRADE (顺向模式) ===
    elif guidance_mode == GUIDANCE_MODE_PROGRADE:
        v_mag = np.linalg.norm(v)
        if v_mag > 0.1:
            orientation = v / v_mag
        else:
            orientation = up

    # === MODE -1: PASSIVE (纯被动/随动模式) ===
    elif guidance_mode == -1.0:
        # 模拟气动稳定（顺风）
        # 假设物体重心设计合理，会自动对准相对风向
        if v_rel_mag > 0.1:
            orientation = v_rel / v_rel_mag
        else:
            orientation = up

    # === FALLBACK ===
    else:
        if v_rel_mag > 0.1:
            orientation = v_rel / v_rel_mag
        else:
            orientation = up

    # 5. Forces
    # Thrust
    f_thrust = thrust_val * orientation
    mdot = -dmdt_val if is_burning else 0.0

    # Aero
    curr_diam = (
        rocket_dia if is_burning else (reentry_dia if reentry_dia > 0 else rocket_dia)
    )

    # 获取 Cd
    use_fixed_cd = drag_props[0] == 1.0
    fixed_cd_val = drag_props[1]

    if use_fixed_cd:
        cd = fixed_cd_val
    else:
        cd = get_cd(
            mach.item(),
            altitude,
            v_rel_mag.item(),
            nc_type_id,
            nc_ld,
            fin_data,
            curr_diam,
        )

    area = np.pi * (curr_diam / 2.0) ** 2
    q = 0.5 * rho * v_rel_mag**2

    f_drag = np.zeros(3)
    if v_rel_mag > 0:
        f_drag = -q * cd * area * (v_rel / v_rel_mag)

    # Lift (Simplified) - 仅在 Launch Mode 且有 AoA 时启用
    f_lift = np.zeros(3)
    guidance_aoa = guidance_params[2]

    if (
        guidance_mode == GUIDANCE_MODE_LAUNCH
        and abs(guidance_aoa) > 0.001
        and v_rel_mag > 10.0
        and rho > 1e-9
    ):
        cl = 2.0 * np.radians(guidance_aoa)  # Linear approx
        f_lift_mag = q * cl * area

        cross_vec = np.cross(v_rel, orientation)
        lift_dir = np.cross(cross_vec, v_rel)
        ld_norm = np.linalg.norm(lift_dir)

        if ld_norm > 1e-9:
            lift_dir /= ld_norm
            f_lift = f_lift_mag * lift_dir

    f_gravity = calculate_gravity(r) * m

    f_total = f_thrust + f_drag + f_lift + f_gravity

    # 防止质量为0除零错误
    if m > 1e-6:
        acc = f_total / m
    else:
        acc = np.zeros(3)

    # Return [vx, vy, vz, ax, ay, az, mdot]
    res = np.empty(7, dtype=np.float64)
    res[0:3] = v
    res[3:6] = acc
    res[6] = mdot
    return res


class RocketSimulator3D:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.time = 0.0
        self.objects: List[TrackedObject] = []

        # --- 1. 预计算主火箭推力矩阵 ---
        stages_list = []
        acc_t = 0.0
        self.stage_timings = []  # [(t_end, dry_mass, index)]
        num_stages = len(self.cfg.stages)  # 获取总级数

        for i, s in enumerate(self.cfg.stages):
            t_start = acc_t
            t_end = acc_t + s.burn_time
            # 只有第一级才设置喷管面积（用于背压修正）
            if i == 0:
                nozzle_area = s.nozzle_area
            else:
                nozzle_area = 0.0  # 其他级使用真空推力，无需背压修正
            stages_list.append([s.thrust, s.dmdt, t_start, t_end, nozzle_area])
            # 仅当不是最后一级时，才添加“级间分离”事件
            # 最后一级应在“载荷分离”事件中处理
            if i < num_stages - 1:
                self.stage_timings.append((t_end, s.dry_mass, i))
            acc_t = t_end

        self.main_stage_matrix = np.array(stages_list, dtype=np.float64)

        # 卫星分离时间 (默认为末级关机后 2秒)
        self.payload_sep_time = acc_t + 2.0
        self.payload_separated = False
        # === 整流罩状态标记 ===
        self.fairing_separated = False
        # 如果配置为0，视为不需要分离
        if self.cfg.fairing_mass <= 0 or self.cfg.fairing_sep_time <= 0:
            self.fairing_separated = True

        # --- 2. 预计算气动数据 ---
        f = self.cfg.fins
        self.fin_data = np.array(
            [
                float(f.number),
                f.height,
                f.root_chord,
                f.tip_chord,
                f.sweep_angle,
                f.thickness,
            ],
            dtype=np.float64,
        )

        self.geo_params = np.array(
            [
                self.cfg.launch_azimuth,
                self.cfg.rocket_diameter,
                self.cfg.reentry_diameter,
                self.cfg.nosecone_ld_ratio,
            ],
            dtype=np.float64,
        )

        self.guidance_params = np.array(
            [
                self.cfg.vertical_time,
                self.cfg.pitch_over_angle,
                self.cfg.guidance_aoa,
                GUIDANCE_MODE_LAUNCH,
            ],
            dtype=np.float64,
        )

        self.nc_type_id = get_nc_type_id(self.cfg.nosecone_type)

        # --- 3. 初始化主火箭对象 ---
        r0, v0 = self._get_initial_state()
        m0 = (
            sum(s.fuel_mass + s.dry_mass for s in self.cfg.stages)
            + self.cfg.payload_mass
        )

        if len(self.cfg.stages) > 0:
            stage1 = self.cfg.stages[0]

            thrust_sl = stage1.thrust
            weight_force = m0 * G0  # G0 = 9.80665

            twr = thrust_sl / weight_force

            if twr <= 1.0:
                # 无法起飞，抛出异常终止仿真
                raise ValueError(
                    f"\n[ERROR] Insufficient Thrust-to-Weight Ratio ({twr:.3f} <= 1.0)!\n"
                    f"The rocket is too heavy to lift off. \n"
                    f"Action: Increase Stage 1 thrust or reduce payload/fuel mass."
                )
            elif twr < 1.2:
                # 推重比过低，重力损失会很大
                print("WARNING: Low TWR (< 1.2). Gravity losses will be high.")

        main_rocket = TrackedObject(
            name="Main Rocket",
            obj_type=OBJ_TYPE_ROCKET,
            state=np.concatenate((r0, v0, [m0])),
            drag_area=np.pi * (self.cfg.rocket_diameter / 2) ** 2,
            drag_coeff_type=0,  # 动态计算 Cd
            stage_matrix=self.main_stage_matrix,
            is_main=True,
        )
        self.objects.append(main_rocket)

    def _get_initial_state(self):
        r_ecef = lla_to_ecef(self.cfg.launch_lat, self.cfg.launch_lon, 5.0)
        omega_vec = np.array([0, 0, OMEGA_E])
        v_rot = np.cross(omega_vec, r_ecef)
        return r_ecef, v_rot

    def _resolve_object_params(self, obj: TrackedObject):
        """
        辅助方法：根据物体类型解析物理参数，解决参数继承和错配问题。
        返回: (geo_params, fin_data, guidance_params, drag_props, nc_type_id)
        """

        # 1. 阻力属性 (Drag Props)
        # [UseFixedCD(0/1), FixedCDValue]
        drag_props = np.array(
            [float(obj.drag_coeff_type), obj.fixed_cd], dtype=np.float64
        )

        # 2. 几何与尾翼参数 (Geometry & Fins)
        if obj.is_main:
            # 只有主火箭才使用 Config 中的完整几何参数
            geo_params = self.geo_params
            fin_data = self.fin_data
            nc_type = self.nc_type_id
        else:
            # Kick Stage, Satellites, Debris
            # 根据 drag_area 反推等效直径
            eff_diam = 2.0 * np.sqrt(obj.drag_area / np.pi)

            # [LaunchAz(Ignored), Diam, ReentryDiam, NcLD]
            geo_params = np.array([0.0, eff_diam, eff_diam, 1.0], dtype=np.float64)

            # 无尾翼
            fin_data = np.zeros(6, dtype=np.float64)
            nc_type = NC_TYPE_UNKNOWN

        # 3. 制导参数 (Guidance)
        # 目标格式: [VertTime, Pitch, AoA, MODE]

        if obj.custom_guidance is not None:
            # 优先使用物体自带的制导设定
            # 确保长度为 4
            g_params = obj.custom_guidance

        elif obj.is_main:
            # 主火箭继承 Config 的发射参数，并设为 MODE_LAUNCH (0)
            base = self.guidance_params
            g_params = np.array(
                [base[0], base[1], base[2], GUIDANCE_MODE_LAUNCH], dtype=np.float64
            )

        else:
            # 其他所有物体（卫星、残骸），如果没有自定义制导
            # 默认为 PASSIVE 模式，无升力
            g_params = np.array(
                [0.0, 0.0, 0.0, GUIDANCE_MODE_PASSIVE], dtype=np.float64
            )

        return geo_params, fin_data, g_params, drag_props, nc_type

    def _calculate_derivatives(
        self, t: float, current_state: np.ndarray, obj: TrackedObject
    ) -> np.ndarray:
        """
        计算状态导数 (Physics Wrapper)。
        已优化：稳健的参数构造，消除隐式继承 Bug。
        """

        # 1. 使用辅助函数解析所有物理参数
        geo_params, fin_data, g_params, drag_props, nc_type = (
            self._resolve_object_params(obj)
        )

        # 2. 调用更新后的物理内核
        return physics_kernel(
            t,
            current_state,
            obj.stage_matrix,  # 动力矩阵
            fin_data,  # 尾翼数据
            geo_params,  # 几何数据
            g_params,  # 制导数据 (含 Mode)
            nc_type,  # 鼻锥类型
            drag_props,  # 阻力属性 (含开关)
        )

    def _rk45_step_single(self, t: float, dt: float, obj: TrackedObject):
        """
        对单个物体执行一步 RK45，返回 (新状态, 误差向量)
        """
        y0 = obj.state

        # k1
        k1 = self._calculate_derivatives(t, y0, obj)

        # k2
        y_temp = y0 + dt * (RK45_A[1, 0] * k1)
        k2 = self._calculate_derivatives(t + RK45_C[1] * dt, y_temp, obj)

        # k3
        y_temp = y0 + dt * (RK45_A[2, 0] * k1 + RK45_A[2, 1] * k2)
        k3 = self._calculate_derivatives(t + RK45_C[2] * dt, y_temp, obj)

        # k4
        y_temp = y0 + dt * (RK45_A[3, 0] * k1 + RK45_A[3, 1] * k2 + RK45_A[3, 2] * k3)
        k4 = self._calculate_derivatives(t + RK45_C[3] * dt, y_temp, obj)

        # k5
        y_temp = y0 + dt * (
            RK45_A[4, 0] * k1
            + RK45_A[4, 1] * k2
            + RK45_A[4, 2] * k3
            + RK45_A[4, 3] * k4
        )
        k5 = self._calculate_derivatives(t + RK45_C[4] * dt, y_temp, obj)

        # k6
        y_temp = y0 + dt * (
            RK45_A[5, 0] * k1
            + RK45_A[5, 1] * k2
            + RK45_A[5, 2] * k3
            + RK45_A[5, 3] * k4
            + RK45_A[5, 4] * k5
        )
        k6 = self._calculate_derivatives(t + RK45_C[5] * dt, y_temp, obj)

        # 计算 5阶解 (用于更新状态)
        y_new = y0 + dt * (
            RK45_B[0] * k1 + RK45_B[2] * k3 + RK45_B[3] * k4 + RK45_B[5] * k6
        )

        # 计算 4阶解 (仅用于估算误差)
        y_star = y0 + dt * (
            RK45_B_STAR[0] * k1
            + RK45_B_STAR[2] * k3
            + RK45_B_STAR[3] * k4
            + RK45_B_STAR[4] * k5
            + RK45_B_STAR[5] * k6
        )

        # 误差向量
        error_vec = np.abs(y_new - y_star)

        return y_new, error_vec

    def run_adaptive(
        self,
        initial_dt: float = 0.1,
        min_dt: float = 1e-4,
        max_dt: float = 10.0,  # 允许在大气外大步长滑行
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):

        print(f"Starting Adaptive Simulation (RK45)...")
        self._record_all_states()

        dt = initial_dt

        running = True

        last_record_time = self.time
        while running:
            t = self.time

            # 0. 仿真结束检查
            active_objs = [o for o in self.objects if o.active]
            if not active_objs or t > 50000:  # 稍微放宽时间限制
                print("Simulation finished.")
                break

            # 1. 获取下一个关键事件时间点
            next_event_t = self._get_next_event_time(t)

            # 2. 步长调整逻辑 (Event Crossing Check)
            hit_event = False
            # 如果当前步长会跨越事件，强制缩短步长，使其刚好落在事件上
            if next_event_t != float("inf") and (t + dt) > next_event_t + 1e-7:
                dt = next_event_t - t
                hit_event = True
                if dt < min_dt:
                    dt = min_dt  # 保护

            # 3. 尝试步进 (Trial Step)
            step_accepted = False

            while not step_accepted:
                proposed_states = []
                max_error_ratio = 0.0

                # 对所有活跃物体计算 RK45
                for obj in active_objs:
                    # 使用精确的海拔计算
                    lat, lon, altitude = eci_to_lla(obj.state[0:3], t)

                    # 计算径向速度，用于判断是否在下降
                    v_rad = 0.0
                    if np.linalg.norm(obj.state[0:3]) > 0:
                        v_rad = np.dot(obj.state[3:6], obj.state[0:3]) / np.linalg.norm(
                            obj.state[0:3]
                        )

                    # 判定条件：
                    # 1. 高度低于地面 (允许 50m 的缓冲，防止数值抖动)
                    # 2. 并不是刚发射 (t > 1.0) 或者 正在向下掉 (v_rad < 0)
                    # 3. 必须处于 active 状态
                    if obj.active and altitude < -50.0 and t > 1.0 and v_rad < 0:
                        obj.active = False
                        print(f"IMPACT: {obj.name} impacted ground at t={t:.2f}s")
                        proposed_states.append((obj, obj.state))
                        continue

                    y_new, err_vec = self._rk45_step_single(t, dt, obj)

                    # 误差计算
                    scale = atol + np.abs(obj.state) * rtol
                    ratio_vec = err_vec / (scale + 1e-30)
                    curr_max_ratio = np.max(ratio_vec)

                    if curr_max_ratio > max_error_ratio:
                        max_error_ratio = curr_max_ratio

                    proposed_states.append((obj, y_new))

                # 判定
                if max_error_ratio <= 1.0 or dt <= min_dt:
                    # === ACCEPT ===
                    step_accepted = True
                    self.time += dt

                    # 更新状态
                    for obj, y_new in proposed_states:
                        obj.state = y_new

                    # 记录
                    if hit_event or (self.time - last_record_time > 1.0):
                        self._record_all_states()

                    # 计算下一步推荐步长
                    if max_error_ratio < 1e-10:
                        max_error_ratio = 1e-10

                    if not hit_event:
                        dt_next = dt * 0.9 * (max_error_ratio**-0.2)
                        dt_next = min(dt_next, dt * 5.0, max_dt)
                        dt = dt_next
                    else:
                        # 刚处理完事件，重置步长以保证物理计算稳定
                        dt = initial_dt

                else:
                    # === REJECT ===
                    dt_next = dt * 0.9 * (max_error_ratio**-0.2)
                    dt_next = max(dt_next, dt * 0.1)  # 别缩太快
                    dt = dt_next
                    if dt < min_dt:
                        dt = min_dt  # 强行推进

            # 4. 执行事件检查
            # 只有当 step_accepted 为 True，且时间推进后，才检查
            self._check_events()

    def _get_next_event_time(self, current_t: float) -> float:
        future_times = []

        # 1. 主火箭级间分离时间
        for stage_evt in self.stage_timings:
            t_sep = stage_evt[0]
            if t_sep > current_t:
                future_times.append(t_sep)

        # 2. 载荷/上面级分离时间
        if not self.payload_separated and self.payload_sep_time > current_t:
            future_times.append(self.payload_sep_time)

        # 3. 垂直爬升结束时间 (影响制导)
        if self.cfg.vertical_time > current_t:
            future_times.append(self.cfg.vertical_time)
        # 整流罩
        if not self.fairing_separated and self.cfg.fairing_sep_time > current_t:
            future_times.append(self.cfg.fairing_sep_time)

        # 4. [新增] 扫描所有物体的子卫星释放计划
        for obj in self.objects:
            if obj.active and obj.release_events:
                # 因为 release_events 是按时间排序的，只看第一个即可
                next_release = obj.release_events[0].release_time
                if next_release > current_t:
                    future_times.append(next_release)

        # 5. 扫描所有物体的推力开关机时间 (确保积分精确落在关机点)
        for obj in self.objects:
            if obj.active:
                for row in obj.stage_matrix:
                    t_start, t_end = row[2], row[3]
                    if t_start > current_t:
                        future_times.append(t_start)
                    if t_end > current_t:
                        future_times.append(t_end)

        if not future_times:
            return float("inf")  # 没有未来事件了

        return min(future_times)

    def _spawn_object(
        self,
        parent: TrackedObject,
        mass: float,
        name: str,
        obj_type: int,
        drag_area: Optional[float] = None,
        stage_matrix=None,
        guidance=None,
        velocity_offset=None,
    ):
        # 1. 继承状态
        new_state = parent.state.copy()
        new_state[6] = mass  # 重置质量

        # 2. 应用速度增量和位置防重叠
        if velocity_offset is not None:
            # 速度直接叠加
            new_state[3:6] += velocity_offset

            # 位置偏移：沿 dV 方向移动 5米，防止物理重叠
            v_norm = np.linalg.norm(velocity_offset)
            if v_norm > 1e-6:
                pos_offset = (velocity_offset / v_norm) * 5.0
                new_state[0:3] += pos_offset
            else:
                # 如果 dV 为 0 (例如单纯释放)，沿速度方向或随机方向偏移
                v_parent = np.linalg.norm(parent.state[3:6])
                if v_parent > 0.1:
                    dir_vec = parent.state[3:6] / v_parent
                    new_state[0:3] += dir_vec * 5.0
                else:
                    new_state[0] += 5.0  # 随便移一下

        # 3. 默认参数处理
        if drag_area is None:
            drag_area = parent.drag_area  # 默认继承
        if stage_matrix is None:
            stage_matrix = np.zeros((0, 5))  # 默认无动力

        # 4. 创建对象
        new_obj = TrackedObject(
            name=name,
            obj_type=obj_type,
            state=new_state,
            drag_area=drag_area,
            drag_coeff_type=1,  # 分离物体通常使用固定Cd
            stage_matrix=stage_matrix,
            custom_guidance=guidance,
        )

        self.objects.append(new_obj)
        return new_obj

    def _check_events(self):
        t = self.time

        # 1. 查找主火箭 (用于级间分离)
        main_rocket = next(
            (o for o in self.objects if o.is_main and o.active),
            None,
        )

        # 2. 处理主火箭级间分离
        if main_rocket:
            self._handle_main_rocket_staging(main_rocket, t)
            self._handle_fairing_separation(main_rocket, t)
            self._handle_main_payload_separation(main_rocket, t)

        # 3. 处理通用释放事件 (KickStage 释放子卫星)
        # 遍历所有拥有 release_events 的物体
        for obj in self.objects:
            if not obj.active or not obj.release_events:
                continue

            # 处理到期的事件
            while obj.release_events and t >= obj.release_events[0].release_time:
                evt = obj.release_events.pop(0)
                self._execute_release_event(obj, evt)

    def _handle_main_rocket_staging(self, rocket: TrackedObject, t: float):
        # 筛选出当前时间点需要发生的事件
        remaining = []
        for sep in self.stage_timings:
            sep_time, dry_mass, idx = sep
            if t >= sep_time:
                print(f"Event: Stage {idx+1} Separation at t={t:.2f}s")
                # 减重
                rocket.state[6] -= dry_mass
                # 生成废弃级 (调用统一工厂)
                self._spawn_object(
                    parent=rocket,
                    mass=dry_mass,
                    name=f"Stage {idx+1} Body",
                    obj_type=OBJ_TYPE_STAGE,
                    drag_area=rocket.drag_area * 4.0,  # 翻滚模拟
                    stage_matrix=np.zeros((0, 5)),
                )
            else:
                remaining.append(sep)
        self.stage_timings = remaining

    def _handle_main_payload_separation(self, rocket: TrackedObject, t: float):
        if not self.payload_separated and t >= self.payload_sep_time:
            self.payload_separated = True
            print(f"Event: Payload Separation at t={t:.2f}s")

            # 载荷总质量 (包含 KickStage 干重+燃料+子卫星)
            payload_total_mass = self.cfg.payload_mass

            # 主火箭扣除载荷
            rocket.state[6] -= payload_total_mass

            # 决定生成什么：Kick Stage (分配器) 还是 被动卫星
            if self.cfg.kick_stage.enabled:
                self._setup_and_spawn_kick_stage(rocket, payload_total_mass, t)
            else:
                self._spawn_object(
                    rocket, payload_total_mass, "Satellite", OBJ_TYPE_SAT, drag_area=1.5
                )

            # 主火箭变为末级残骸
            rocket.name = "Upper Stage Body"
            rocket.obj_type = OBJ_TYPE_STAGE
            rocket.drag_area *= 4.0
            rocket.stage_matrix = np.zeros((0, 5))

    # ==========================================
    # 业务逻辑 B: 上面级/分配器逻辑
    # ==========================================
    def _setup_and_spawn_kick_stage(self, parent: TrackedObject, mass: float, t: float):
        """配置并生成 Kick Stage，包含入轨点火、子卫星计划以及反推离轨"""
        ks_cfg = self.cfg.kick_stage

        # --- 1. 计算入轨点火 (Insertion Burn) ---
        # 扣除预留给反推的燃料，剩下的用于入轨
        insertion_fuel = ks_cfg.fuel_mass
        if ks_cfg.deorbit_enabled:
            insertion_fuel -= ks_cfg.deorbit_fuel_mass
            if insertion_fuel < 0:
                insertion_fuel = 0

        dmdt = ks_cfg.thrust / (ks_cfg.isp * 9.80665)
        burn_time_insertion = insertion_fuel / dmdt

        # 滑行逻辑
        r, v = parent.state[0:3], parent.state[3:6]
        time_to_apo = estimate_time_to_apoapsis(r, v)

        if time_to_apo > burn_time_insertion and np.linalg.norm(r) > (R_EARTH + 100000):
            delay = time_to_apo - burn_time_insertion / 2.0
            print(f"  -> Coasting to Apoapsis ({delay:.1f}s delay)")
        else:
            delay = ks_cfg.ignition_delay

        t_start_1 = t + delay
        t_end_1 = t_start_1 + burn_time_insertion

        # 这里的 nozzle_area 设为 0，真空推力，不受大气压影响
        # 矩阵格式: [thrust, dmdt, t_start, t_end, area]
        matrix_list = [[ks_cfg.thrust, dmdt, t_start_1, t_end_1, 0.0]]

        # --- 2. 规划子卫星释放时间 ---
        # 记录最后一个事件的时间，用于安排反推
        last_event_time = t_end_1 + 10.0

        # 临时存储事件，稍后赋值给对象
        payload_events = []

        current_release_base = t_end_1 + 10.0  # 入轨后 10s 开始释放

        for p_cfg in ks_cfg.payloads:
            curr_t = current_release_base + p_cfg.release_start_time
            for i in range(p_cfg.count):
                evt = PayloadReleaseEvent(
                    release_time=curr_t,
                    mass=p_cfg.mass,
                    name=p_cfg.name_pattern.format(i=i + 1),
                    sep_vel=p_cfg.separation_velocity,
                    direction_type=3,  # Random
                )
                payload_events.append(evt)

                # 更新最后时间
                if curr_t > last_event_time:
                    last_event_time = curr_t

                curr_t += p_cfg.release_interval

        # 排序事件
        payload_events.sort(key=lambda x: x.release_time)

        # --- 3. 计算反推离轨点火 (Deorbit Burn) ---
        if ks_cfg.deorbit_enabled and ks_cfg.deorbit_fuel_mass > 0:
            burn_time_deorbit = ks_cfg.deorbit_fuel_mass / dmdt

            t_start_2 = last_event_time + ks_cfg.deorbit_delay
            t_end_2 = t_start_2 + burn_time_deorbit

            print(
                f"  -> Scheduled Deorbit Burn at t={t_start_2:.1f}s (Duration: {burn_time_deorbit:.1f}s)"
            )

            # 推力设为负值 (-ks_cfg.thrust)
            matrix_list.append([-ks_cfg.thrust, dmdt, t_start_2, t_end_2, 0.0])

        # --- 4. 生成对象 ---
        ks_obj = self._spawn_object(
            parent=parent,
            mass=mass,
            name="Dispenser_Stage",
            obj_type=OBJ_TYPE_ROCKET,
            drag_area=2.0,
            stage_matrix=np.array(matrix_list, dtype=np.float64),
            guidance=np.array([0.0, 0.0, 0.0, 1.0]),  # 1.0 = Prograde Mode
        )

        ks_obj.release_events = payload_events

    def _execute_release_event(self, parent: TrackedObject, evt: PayloadReleaseEvent):
        # 1. 计算分离 Delta V (子卫星相对于惯性系的速度增量贡献)

        v_rel_vec = self._calculate_separation_velocity(
            parent, evt.sep_vel, evt.direction_type
        )

        # 当前总质量
        m_total = parent.state[6]
        m_sat = evt.mass
        m_parent_new = m_total - m_sat

        if m_total <= 0:
            return

        # 动量守恒分配速度
        dv_sat = v_rel_vec * (m_parent_new / m_total)
        dv_parent = -v_rel_vec * (m_sat / m_total)

        # 2. 生成子卫星 (应用 dv_sat)
        self._spawn_object(
            parent=parent,
            mass=m_sat,
            name=evt.name,
            obj_type=OBJ_TYPE_SAT,
            drag_area=0.5,
            velocity_offset=dv_sat,  # _spawn_object 内部会叠加到 parent.v 上
        )

        # 3. 更新母体状态 (应用反冲)
        parent.state[0:3] += dv_parent * 0.001  # 位置微调防重叠
        parent.state[3:6] += dv_parent  # 速度应用反冲
        parent.state[6] = m_parent_new  # 质量更新

        print(
            f"Event: {parent.name} deployed {evt.name} (Recoil dV: {np.linalg.norm(dv_parent):.4f} m/s)"
        )

    def _calculate_separation_velocity(
        self, parent: TrackedObject, speed: float, mode: int
    ) -> np.ndarray:
        if speed <= 0:
            return np.zeros(3)

        # 获取母体姿态基向量
        v = parent.state[3:6]
        r = parent.state[0:3]

        prograde = (
            v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([1, 0, 0])
        )

        if mode == 0:  # Prograde
            return prograde * speed
        elif mode == 3:  # Random
            # 简单随机
            rnd = np.random.normal(size=3)
            rnd /= np.linalg.norm(rnd)
            return rnd * speed

        return np.zeros(3)

    def _record_all_states(self):
        # 遍历所有物体并记录
        for obj in self.objects:
            if not obj.active:
                continue

            r = obj.state[0:3]
            v = obj.state[3:6]
            m = obj.state[6]

            lat, lon, alt = eci_to_lla(r, self.time)

            # --- 速度计算 ---
            omega_vec = np.array([0, 0, OMEGA_E])
            v_rel = v - np.cross(omega_vec, r)
            v_mag_rel = np.linalg.norm(v_rel)  # 相对地表速度 (用于气动)
            v_mag_inertial = np.linalg.norm(v)
            r_mag = np.linalg.norm(r)

            if r_mag > 0:
                spec_energy = 0.5 * v_mag_inertial**2 - GM / r_mag
            else:
                spec_energy = -np.inf

            total_energy = spec_energy * m

            # --- 射程计算 ---
            r0 = lla_to_ecef(self.cfg.launch_lat, self.cfg.launch_lon, 0)
            curr_ecef = lla_to_ecef(lat, lon, 0)

            n_r0 = np.linalg.norm(r0)
            n_curr = np.linalg.norm(curr_ecef)

            if n_r0 > 1 and n_curr > 1:
                cos_val = np.dot(r0, curr_ecef) / (n_r0 * n_curr)
                cos_val = np.clip(cos_val, -1.0, 1.0)
                angle = np.arccos(cos_val)
                downrange = angle * R_EARTH
            else:
                downrange = 0.0
            hist = {
                "Time": round(self.time, 3),
                "Object": obj.name,
                "Type": obj.obj_type,
                "Downrange": downrange,
                "Altitude": alt,
                "Velocity": v_mag_rel,
                "InertialVelocity": v_mag_inertial,
                "SpecificEnergy": spec_energy,
                "TotalEnergy": total_energy,
                "Mass": m,
                "Latitude": lat,
                "Longitude": lon,
            }
            if obj.obj_type == OBJ_TYPE_SAT:
                kep = calculate_kepler_elements(r, v)
                # 如果是双曲线轨道(e>1)或者处于大气层深处，a可能为负或无意义
                orbit_a_km = kep[0]
                orbit_e = kep[1]
                orbit_i = kep[2]
                orbit_period = kep[6]

                # 近地点/远地点高度估算 (椭圆轨道时有效)
                if orbit_e < 1.0 and orbit_a_km > 0:
                    apoapsis_km = orbit_a_km * (1 + orbit_e) - R_EARTH / 1000.0
                    periapsis_km = orbit_a_km * (1 - orbit_e) - R_EARTH / 1000.0
                else:
                    apoapsis_km = 0.0
                    periapsis_km = 0.0
                kep_hist = {
                    "Orbit_a_km": orbit_a_km,
                    "Orbit_e": orbit_e,
                    "Orbit_i": orbit_i,
                    "Orbit_Peri_km": periapsis_km,
                    "Orbit_Apo_km": apoapsis_km,
                    "Orbit_T_min": orbit_period,
                    "Orbit_RAAN": kep[3],
                    "Orbit_ArgP": kep[4],
                    "Orbit_Nu": kep[5],
                }
                hist.update(kep_hist)

            # 记录数据
            obj.history.append(hist)

    def _handle_fairing_separation(self, rocket: TrackedObject, t: float):
        if not self.fairing_separated and t >= self.cfg.fairing_sep_time:
            self.fairing_separated = True

            # 1. 物理属性变更
            f_mass = self.cfg.fairing_mass
            rocket.state[6] -= f_mass  # 减重

            rocket.drag_coeff_type = 1  # 切换为 Fixed Cd
            rocket.fixed_cd = 0.8  # 钝体载荷的典型值
            # 减小一点主火箭迎风面积
            rocket.drag_area *= 0.8

            print(f"Event: Fairing Separation at t={t:.2f}s (Mass -{f_mass}kg)")

            # 2. 生成碎片 (两瓣整流罩)
            # 给定一个侧向分离速度
            v_sep = 2.0  # m/s

            # 左半瓣
            self._spawn_object(
                parent=rocket,
                mass=f_mass / 2.0,
                name="Fairing Half A",
                obj_type=OBJ_TYPE_STAGE,  # 视为碎片
                drag_area=rocket.drag_area / 2.0,  # 碎片阻力面积较大
                velocity_offset=self._calculate_separation_velocity(
                    rocket, v_sep, 3
                ),  # 随机/侧向方向
            )

            # 右半瓣
            self._spawn_object(
                parent=rocket,
                mass=f_mass / 2.0,
                name="Fairing Half B",
                obj_type=OBJ_TYPE_STAGE,
                drag_area=rocket.drag_area / 2.0,
                velocity_offset=self._calculate_separation_velocity(rocket, v_sep, 3),
            )

    def export_csv(self):
        """导出所有物体的轨迹到不同的文件"""
        import os

        # 按物体名称分组导出
        for obj in self.objects:
            if not obj.history:
                continue

            # 清理文件名
            safe_name = "".join([c if c.isalnum() else "_" for c in obj.name])
            filename = f"trace_{safe_name}.csv"

            keys = obj.history[0].keys()
            try:
                with open(filename, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(obj.history)
                print(f"Exported data for '{obj.name}' to {filename}")
            except Exception as e:
                print(f"Failed to export {obj.name}: {e}")

    def plot(self):
        # 1. 极简边距设置
        plt.style.use("seaborn-v0_8-darkgrid")
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        fig.suptitle(
            f"Orbit & Trajectory Dashboard (T={self.time:.1f}s)",
            fontsize=16,
            weight="bold",
        )

        # 2. 定义 2x3 网格
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2])

        # Row 1: 数据分析
        ax_prof = fig.add_subplot(gs[0, 0])
        ax_vel = fig.add_subplot(gs[0, 1])
        ax_orb = fig.add_subplot(gs[0, 2])

        # Row 2: 空间可视化
        ax_map = fig.add_subplot(gs[1, 0])
        ax_ecef = fig.add_subplot(gs[1, 1], projection="3d")
        ax_eci = fig.add_subplot(gs[1, 2], projection="3d")

        # 颜色生成
        colors = plt.cm.jet(np.linspace(0, 1, len(self.objects)))

        # 范围记录
        max_range_ecef = R_EARTH / 1000.0
        max_range_eci = R_EARTH / 1000.0

        print("Generating 6-Panel Analysis...")

        for idx, obj in enumerate(self.objects):
            if not obj.history:
                continue

            # --- 数据准备 ---
            hist = obj.history
            times = np.array([h["Time"] for h in hist])

            # 降采样
            step = max(1, len(times) // 1200)

            # 基础数据
            t_plot = times[::step]
            alt = np.array([h["Altitude"] for h in hist])[::step] / 1000.0
            dr = np.array([h["Downrange"] for h in hist])[::step] / 1000.0
            v_inertial = np.array([h["InertialVelocity"] for h in hist])[::step]
            lats = np.array([h["Latitude"] for h in hist])[::step]
            lons = np.array([h["Longitude"] for h in hist])[::step]

            color = colors[idx]
            label = obj.name

            # [Plot 1] Flight Profile
            ax_prof.plot(dr, alt, label=label, color=color, linewidth=1.5)

            # [Plot 2] Velocity
            ax_vel.plot(t_plot, v_inertial, label=label, color=color)

            # [Plot 3] Orbit Evolution (Apogee/Perigee History)
            # 需要重新计算每一步的瞬时轨道根数 只有当高度较高且速度够快时才计算，避免发射台附近的奇异值
            valid_mask = (alt > 50) & (v_inertial > 1000)
            if np.any(valid_mask):
                t_valid = t_plot[valid_mask]
                r_mag = (alt[valid_mask] * 1000.0) + R_EARTH  # m
                v_mag = v_inertial[valid_mask]  # m/s

                # 活力公式 (Vis-Viva) 估算半长轴 a
                # E = v^2/2 - mu/r = -mu/2a  =>  1/a = 2/r - v^2/mu
                inv_a = (2.0 / r_mag) - (v_mag**2 / GM)

                # 筛选椭圆轨道 (inv_a > 0)
                ellipse_mask = inv_a > 1e-9
                if np.any(ellipse_mask):
                    t_ell = t_valid[ellipse_mask]
                    a = 1.0 / inv_a[ellipse_mask]  # m

                    # 半长轴高度代表轨道能量
                    sma_km = (a / 1000.0) - (R_EARTH / 1000.0)

                    # 绘制半长轴高度 (代表轨道总能量)
                    ax_orb.plot(
                        t_ell,
                        sma_km,
                        color=color,
                        linestyle="-",
                        linewidth=1.5,
                        alpha=0.8,
                    )

            # [Plot 4] Ground Track
            ax_map.plot(lons, lats, color=color, linewidth=1.2)
            ax_map.scatter(lons[-1], lats[-1], color=color, s=15, marker="x")

            # --- 3D 坐标转换 ---
            r_m = (alt * 1000.0) + R_EARTH
            lat_r = np.radians(lats)
            lon_r = np.radians(lons)

            x_ecef = r_m * np.cos(lat_r) * np.cos(lon_r) / 1000.0
            y_ecef = r_m * np.cos(lat_r) * np.sin(lon_r) / 1000.0
            z_ecef = r_m * np.sin(lat_r) / 1000.0

            # ECI 转换
            theta = OMEGA_E * t_plot
            c, s = np.cos(theta), np.sin(theta)
            x_eci = x_ecef * c - y_ecef * s
            y_eci = x_ecef * s + y_ecef * c
            z_eci = z_ecef

            # [Plot 5] ECEF 3D
            ax_ecef.plot(x_ecef, y_ecef, z_ecef, color=color, linewidth=1)
            if np.max(np.abs(x_ecef)) > max_range_ecef:
                max_range_ecef = np.max(np.abs(x_ecef))

            # [Plot 6] ECI 3D
            ax_eci.plot(x_eci, y_eci, z_eci, color=color, linewidth=1, label=label)
            if np.max(np.abs(x_eci)) > max_range_eci:
                max_range_eci = np.max(np.abs(x_eci))

        # --- 图表装饰 ---

        # 1. Profile
        ax_prof.set_title("1. Flight Profile", fontsize=10, weight="bold")
        ax_prof.set_ylabel("Alt (km)")
        ax_prof.set_xlabel("Downrange (km)")

        # 2. Velocity
        ax_vel.set_title("2. Inertial Velocity", fontsize=10, weight="bold")
        ax_vel.set_ylabel("V (m/s)")
        ax_vel.axhline(7800, color="grey", ls=":", alpha=0.5)

        # 3. Orbit Energy (SMA)
        ax_orb.set_title(
            "3. Orbital Energy (Semi-Major Axis Alt)", fontsize=10, weight="bold"
        )
        ax_orb.set_ylabel("SMA Altitude (km)")
        ax_orb.set_xlabel("Time (s)")
        ax_orb.set_ylim(bottom=-R_EARTH / 1000.0)  # 允许显示负值(亚轨道)
        ax_orb.grid(True, which="both", alpha=0.3)
        ax_orb.text(
            0.05,
            0.9,
            "Height > 0 means Orbit capable",
            transform=ax_orb.transAxes,
            fontsize=8,
            color="green",
        )

        # 4. Map
        ax_map.set_title("4. Ground Track", fontsize=10, weight="bold")
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.set_aspect("equal")
        ax_map.grid(True, ls=":")

        # Common 3D Setup
        def setup_3d(ax, title, limit):
            ax.set_title(title, fontsize=10, weight="bold")
            # Wireframe Earth
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            Re = R_EARTH / 1000.0
            x = Re * np.cos(u) * np.sin(v)
            y = Re * np.sin(u) * np.sin(v)
            z = Re * np.cos(v)
            ax.plot_wireframe(x, y, z, color="gray", alpha=0.1, lw=0.5)
            # Aspect Ratio
            lim = max(limit, Re * 1.2)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            # Remove pane color for cleaner look
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(False)  # Clean grid

        # 5. ECEF
        setup_3d(ax_ecef, "5. ECEF (Rotating Frame)", max_range_ecef)

        # 6. ECI
        setup_3d(ax_eci, "6. ECI (Inertial Frame)", max_range_eci)

        # Legend
        ax_eci.legend(loc="upper right", fontsize="xx-small")

        plt.show()


if __name__ == "__main__":
    # ==========================================
    # 仿真配置：基于 "Electron-ish" 轻型火箭参数
    # ==========================================

    # --- 1. 定义火箭各级 (Rocket Stages) ---
    # 第一级: Rutherford x9
    # Electron 真实数据参考：真空推力 ~224kN，海平面 ~190kN，真空比冲 ~311s
    stage_1 = RocketStage(
        fuel_mass=11350.0,
        dry_mass=950.0,
        isp=311.0,         # 真空比冲
        thrust=224000.0,   # 输入真空推力
        # 计算喷管面积：(F_vac - F_sl) / P_sl = (224000 - 190000) / 101325 ≈ 0.335
        nozzle_area=0.335, # 调整面积以获得正确的海平面推力
    )

    # 第二级: Rutherford Vacuum
    stage_2 = RocketStage(
        fuel_mass=2050.0,
        dry_mass=250.0,
        isp=343.0,
        thrust=25800.0,    # 真空推力 (Electron 二级约为 25.8kN)
        nozzle_area=0.8,   # 真空发动机喷管其实很大，但在高空 pressure≈0，这项影响很小
    )

    # --- 2. 定义载荷与 Kick Stage ---
    # 定义星座释放计划：释放 5 颗 10kg 的卫星
    constellation_plan = SubSatelliteConfig(
        name_pattern="Sat_Mini_{i}",
        count=5,
        mass=40.0,  # 单颗质量
        release_start_time=20.0,  # 入轨/KickStage燃尽后20秒开始
        release_interval=15.0,  # 每15秒释放一颗
        separation_velocity=1.2,  # 弹簧分离速度 m/s
    )

    # 定义 Kick Stage (上面级/分配器)
    # 类似于 Curie Engine
    kick_stage_conf = KickStageConfig(
        enabled=True,
        dry_mass=40.0,
        fuel_mass=50.0,  # 增加一点总燃料 (入轨需要约40-50kg)
        thrust=120.0,
        isp=320.0,
        # === 开启自毁模式 ===
        deorbit_enabled=True,
        deorbit_fuel_mass=20.0,  # 留 15kg 用于反推
        deorbit_delay=300.0,  # 最后一颗卫星走后 300秒 再自毁
        # ===================
        payloads=[constellation_plan],
    )

    # --- 3. 组装 SimConfig ---
    config = SimConfig(
        stages=[stage_1, stage_2],
        payload_mass=290.0,  # 有效载荷总重 (卫星+分配器+KickStage燃料)
        rocket_diameter=1.2,  # 直径 (m)
        nosecone_type="Ogive",  # 鼻锥类型
        nosecone_ld_ratio=4.0,  # 长细比
        reentry_diameter=1.2,
        fairing_mass=50.0,
        fairing_sep_time=185.0,
        # 气动尾翼 (电子号没有，设为0)
        fins=FinConfig(0, 0, 0, 0, 0, 0),
        # 发射场: 新西兰 Mahia (近似)
        launch_lat=-39.26,
        launch_lon=177.86,
        launch_azimuth=90.0,  # 向正东发射，利用地球自转
        # 制导律参数
        vertical_time=12.0,  # 垂直爬升12秒避开回转塔
        pitch_over_angle=5.0,  # 初始转弯角度 3 度
        guidance_aoa=0.0,  # 零攻角重力转弯
        kick_stage=kick_stage_conf,
    )

    # --- 4. 运行仿真 ---
    print("Initialize Simulation...")
    sim = RocketSimulator3D(config)

    print("\n--- Mission Start ---")
    # 运行足够长的时间以覆盖入轨和卫星释放
    # 步长自适应，不用担心时间过长
    sim.run_adaptive(max_dt=5.0)

    print("\n--- Mission Complete ---")

    # --- 5. 简单的结果分析 ---
    # 查找主载荷或 KickStage
    kicker = next((o for o in sim.objects if "Dispenser" in o.name), None)

    if kicker and kicker.history:
        last_state = kicker.history[-1]
        print(f"\n[Orbit Status - {kicker.name}]")
        print(f"Altitude: {last_state['Altitude']/1000:.2f} km")
        print(f"Velocity: {last_state['InertialVelocity']:.2f} m/s")

        # 如果有轨道根数记录
        if "Orbit_a_km" in last_state:
            print(
                f"Orbit: {last_state['Orbit_Peri_km']:.1f} km x {last_state['Orbit_Apo_km']:.1f} km"
            )
            print(f"Inclination: {last_state['Orbit_i']:.2f} deg")

    # 统计释放的卫星
    sats = [o for o in sim.objects if o.obj_type == OBJ_TYPE_SAT]
    print(f"\n[Deployment] Deployed {len(sats)} satellites.")
    for s in sats:
        if s.history:
            h = s.history[-1]
            print(
                f" - {s.name}: H={h['Altitude']/1000:.1f}km, V={h['InertialVelocity']:.1f}m/s"
            )
            if "Orbit_a_km" in h:
                print(
                    f"\tperiapsis {h['Orbit_Peri_km']:.1f} km apoapsis {h['Orbit_Apo_km']:.1f} km inclination: {h['Orbit_i']:.2f} deg"
                )
    # --- 6. 绘图 ---
    sim.plot()
    # sim.export_csv()
