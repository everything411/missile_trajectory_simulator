import math
import numpy as np
from numba import njit
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


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
    thrust: float  # N (Sea Level / Nominal)
    burn_time: float = 0.0
    dmdt: float = 0.0

    def __post_init__(self):
        # 自动计算派生属性
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
    rocket_diameter: float
    nozzle_area: float
    nosecone_type: str = "Conical"
    nosecone_ld_ratio: float = 3.0
    reentry_diameter: float = 0.0
    fins: FinConfig = field(default_factory=FinConfig)

    # --- 3D 发射参数 ---
    launch_lat: float = 0  # 发射纬度 (Degrees), 默认0
    launch_lon: float = 0  # 发射经度
    launch_azimuth: float = 90.0  # 发射方位角 (0=North, 90=East)

    # --- 制导参数 ---
    vertical_time: float = 5.0  # 垂直爬升时间
    pitch_over_angle: float = 2.0  # 初始程序转弯角度 (deg)
    guidance_aoa: float = 0.0  # 设定的飞行攻角 (用于产生升力, 阶段3特性)


# ==========================================
# 0. 预处理常量与全局数组
# ==========================================
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


# 大气层数据表 (转为numpy数组以提升性能)
# Format: [Base Alt, Lapse Rate, Base Temp, Base Pressure]
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


@njit(cache=True)
def lla_to_ecef_numba(lat: float, lon: float, alt: float) -> np.ndarray:
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
def eci_to_lla_numba(r_eci: np.ndarray, time: float) -> tuple:
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
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))

    return np.degrees(lat), np.degrees(lon), h


@njit(cache=True)
def atmosphere_numba(altitude_m: float) -> tuple:
    if altitude_m < 0:
        altitude_m = 0

    # Part 1: < 86km
    if altitude_m < 86000:
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
def get_cd_numba(
    mach: float,
    h: float,
    velocity: float,
    nc_type_id: int,
    nc_ld: float,
    fin_data: np.ndarray,
    rocket_diam: float,
) -> float:
    """
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

        rho, _, _ = atmosphere_numba(h)
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
            cd_body = 0.075 * ld + 0.275

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
def calculate_gravity_numba(r_vec: np.ndarray) -> np.ndarray:
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
    stage_matrix: np.ndarray,  # [thrust, dmdt, t_start, t_end]
    fin_data: np.ndarray,
    geo_params: np.ndarray,  # [launch_az, rocket_dia, nozzle_area, reentry_dia, nc_ld]
    guidance_params: np.ndarray,  # [vert_time, pitch_angle, aoa]
    nc_type_id: int,
) -> np.ndarray:

    r = state[0:3]
    v = state[3:6]
    m = state[6]

    # Unpack geometric params
    launch_az = geo_params[0]
    rocket_dia = geo_params[1]
    nozzle_area = geo_params[2]
    reentry_dia = geo_params[3]
    nc_ld = geo_params[4]

    # 1. Environment
    _, _, altitude = eci_to_lla_numba(r, t)
    rho, temp_k, pressure = atmosphere_numba(altitude)
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
        if stage_matrix[i, 2] <= t < stage_matrix[i, 3]:
            thrust_nominal = stage_matrix[i, 0]
            dmdt_val = stage_matrix[i, 1]
            # Pressure correction for first stage (index 0 check via time usually)
            if i == 0:
                thrust_val = thrust_nominal + nozzle_area * (101325.0 - pressure)
            else:
                thrust_val = thrust_nominal
            is_burning = True
            break

    # 4. Orientation & Guidance
    r_mag = np.linalg.norm(r)
    up = r / r_mag
    ez = np.array([0.0, 0.0, 1.0])
    east = np.cross(ez, up)
    norm_east = np.linalg.norm(east)
    if norm_east > 1e-9:
        east = east / norm_east
    else:
        east = np.array([1.0, 0.0, 0.0])  # Pole singularity handling

    north = np.cross(up, east)

    az_rad = np.radians(launch_az)
    launch_dir_h = np.sin(az_rad) * east + np.cos(az_rad) * north

    vert_time = guidance_params[0]
    pitch_angle = guidance_params[1]
    guidance_aoa = guidance_params[2]

    orientation = np.zeros(3)

    if t < vert_time:
        orientation = up
    elif t < vert_time + 10.0:
        ratio = (t - vert_time) / 10.0
        tilt_angle = np.radians(pitch_angle) * ratio
        orientation = np.cos(tilt_angle) * up + np.sin(tilt_angle) * launch_dir_h
        orientation = orientation / np.linalg.norm(orientation)
    else:
        # Gravity Turn / AoA phase
        if v_rel_mag > 1.0:
            vel_dir = v_rel / v_rel_mag
            if guidance_aoa != 0:
                traj_normal = np.cross(v_rel, -up)
                tn_norm = np.linalg.norm(traj_normal)
                if tn_norm > 1e-9:
                    traj_normal /= tn_norm
                    aoa_rad = np.radians(guidance_aoa)
                    orientation = vel_dir * np.cos(aoa_rad) + np.cross(
                        traj_normal, vel_dir
                    ) * np.sin(aoa_rad)
                else:
                    orientation = vel_dir
            else:
                orientation = vel_dir
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
    cd = get_cd_numba(
        mach.item(), altitude, v_rel_mag.item(), nc_type_id, nc_ld, fin_data, curr_diam
    )

    area = np.pi * (curr_diam / 2.0) ** 2
    q = 0.5 * rho * v_rel_mag**2

    f_drag = np.zeros(3)
    if v_rel_mag > 0:
        f_drag = -q * cd * area * (v_rel / v_rel_mag)

    # Lift (Simplified)
    f_lift = np.zeros(3)
    if guidance_aoa != 0 and v_rel_mag > 10.0:
        cl = 2.0 * np.radians(guidance_aoa)  # Linear approx
        f_lift_mag = q * cl * area
        cross_vec = np.cross(v_rel, orientation)
        lift_dir = np.cross(cross_vec, v_rel)
        ld_norm = np.linalg.norm(lift_dir)
        if ld_norm > 1e-9:
            lift_dir /= ld_norm
            f_lift = f_lift_mag * lift_dir

    f_gravity = calculate_gravity_numba(r) * m

    f_total = f_thrust + f_drag + f_lift + f_gravity
    acc = f_total / m

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
        self.state_history = []

        # --- Precompute Arrays for Numba ---
        # 1. Stages: [thrust, dmdt, t_start, t_end]
        stages_list = []
        acc_t = 0.0
        for s in self.cfg.stages:
            t_start = acc_t
            t_end = acc_t + s.burn_time
            stages_list.append([s.thrust, s.dmdt, t_start, t_end])
            acc_t = t_end

        self.stage_matrix = np.array(stages_list, dtype=np.float64)
        self.stage_timings = [
            (row[2], row[3]) for row in stages_list
        ]  # For separation logic

        # 2. Fins: [number, height, root, tip, sweep, thick]
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

        # 3. Geometry & Guidance
        self.geo_params = np.array(
            [
                self.cfg.launch_azimuth,
                self.cfg.rocket_diameter,
                self.cfg.nozzle_area,
                self.cfg.reentry_diameter,
                self.cfg.nosecone_ld_ratio,
            ],
            dtype=np.float64,
        )

        self.guidance_params = np.array(
            [self.cfg.vertical_time, self.cfg.pitch_over_angle, self.cfg.guidance_aoa],
            dtype=np.float64,
        )

        self.nc_type_id = get_nc_type_id(self.cfg.nosecone_type)

        # Init State
        r0, v0 = self._get_initial_state()
        m0 = (
            sum(s.fuel_mass + s.dry_mass for s in self.cfg.stages)
            + self.cfg.payload_mass
        )
        self.current_state = np.concatenate((r0, v0, [m0]))

    def _get_initial_state(self):
        # 使用 Numba 版本的工具函数
        r_ecef = lla_to_ecef_numba(self.cfg.launch_lat, self.cfg.launch_lon, 5.0)
        r_eci = r_ecef
        omega_vec = np.array([0, 0, OMEGA_E])
        v_rot = np.cross(omega_vec, r_eci)
        return r_eci, v_rot

    def _calculate_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        # 只是一个 Wrapper，调用编译好的 Kernel
        return physics_kernel(
            t,
            state,
            self.stage_matrix,
            self.fin_data,
            self.geo_params,
            self.guidance_params,
            self.nc_type_id,
        )

    def run(self, dt: float = 0.05):
        """主循环"""
        print(f"Starting 3D Simulation with dt={dt}s...")

        self._record_state()

        while True:
            t = self.time
            state_vec = self.current_state

            # --- 1. 撞地检测 ---
            rx, ry, rz = state_vec[0:3]
            # 简单的距离判断作为第一道防线
            r_mag = math.sqrt(rx**2 + ry**2 + rz**2)

            # 精确的椭球判断
            is_underground = (
                (rx**2 + ry**2) / R_EARTH**2 + (rz**2) / R_POLAR**2
            ) < 0.999999

            if t > 1.0 and is_underground:
                print(
                    f"Impact detected at t={t:.2f}s, Velocity={np.linalg.norm(state_vec[3:6]):.1f} m/s"
                )
                break

            if t > 10000:
                print("Timeout reached.")
                break

            # --- 3. RK4 积分 ---
            try:
                k1 = self._calculate_derivatives(t, state_vec)
                k2 = self._calculate_derivatives(t + dt / 2, state_vec + dt * k1 / 2)
                k3 = self._calculate_derivatives(t + dt / 2, state_vec + dt * k2 / 2)
                k4 = self._calculate_derivatives(t + dt, state_vec + dt * k3)

                y_next = state_vec + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            except Exception as e:
                print(f"Integration error at t={t:.2f}: {e}")
                break

            # --- 4. 级间分离逻辑 ---
            for i, (t_s, t_e) in enumerate(self.stage_timings):
                # 检测是否刚刚跨越了分离时间点
                if t < t_e and (t + dt) >= t_e:
                    dry_mass = self.cfg.stages[i].dry_mass
                    print(
                        f"Event: Stage {i+1} Separation at t={t:.2f}s. Dropping Dry Mass: {dry_mass}kg"
                    )
                    y_next[6] -= dry_mass

            self.time += dt
            self.current_state = y_next

            # if int(self.time * 10) % 5 == 0:
            self._record_state()

        print(f"Simulation ended at t={self.time:.2f}s")

    def _record_state(self):
        r = self.current_state[0:3]
        v = self.current_state[3:6]
        m = self.current_state[6]

        lat, lon, alt = eci_to_lla_numba(r, self.time)

        # 计算相对速度用于 Mach
        omega_vec = np.array([0, 0, OMEGA_E])
        v_rel = v - np.cross(omega_vec, r)
        v_mag = np.linalg.norm(v_rel)

        # 简单的射程计算 (大圆距离)
        r0 = lla_to_ecef_numba(self.cfg.launch_lat, self.cfg.launch_lon, 0)
        curr_ecef = lla_to_ecef_numba(lat, lon, 0)
        # 弧长近似
        cos_val = np.dot(r0, curr_ecef) / (np.linalg.norm(r0) * np.linalg.norm(curr_ecef))
        cos_val = np.clip(cos_val, -1, 1)
        angle = np.arccos(cos_val)
        downrange = angle * R_EARTH

        self.state_history.append(
            {
                "Time": self.time,
                "Altitude": alt,
                "Downrange": downrange,
                "Velocity": v_mag,
                "Mass": m,
                "Latitude": lat,
                "Longitude": lon,
            }
        )

    def export_csv(self, filename: str):
        if not self.state_history:
            return
        keys = self.state_history[0].keys()
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.state_history)
        print(f"Data exported to {filename}")
