# =============================================================
# 2D Ideal MHD Solver Engine
# =============================================================
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
import time

# -------------------------------------------------------------
# section 0: physical constants and solver config
# -------------------------------------------------------------

@dataclass
class SolverConfig:
    """
    mhd solver config parameters.
    """
    # domain
    nx: int = 256
    nz: int= 512
    x_min: float = 0.0
    x_max: float= 1.0
    z_min: float = 0.0
    z_max: float = 4.0

    gamma: float= 5.0 / 3.0
    
    # GLM cleaning stuff
    ch: float= 0.0
    cr: float = 0.18
    
    # time stepping 
    cfl: float = 0.4
    t_end: float= 2.0
    
    # ghost cells
    nghost: int= 2
    nvar: int = 9
    
    def __post_init__(self):
        
        self.dx= (self.x_max - self.x_min) / self.nx
        self.dz = (self.z_max - self.z_min) / self.nz
        # arrays
        self.nx_tot = self.nx + 2 * self.nghost
        self.nz_tot= self.nz + 2 * self.nghost


# -------------------------------------------------------------
# MHD state setup
# -------------------------------------------------------------

class MHDState:
    """
    thermo + electromagnetic state of the MHD system on the grid.
    
    conservative variables Q (nvar, nx_tot, nz_tot):
        Q[0] = rho
        Q[1] = rho * vx
        Q[2] = rho * vz
        Q[3] = rho * vy
        Q[4] = Bx
        Q[5] = Bz
        q[6] = By
        Q[7] = E
        Q[8] = psi
    
    primitive variables W (nvar, nx_tot, nz_tot):
        W[0] = rho
        W[1] = vx
        W[2] = vz
        W[3] = vy
        W[4] = Bx
        W[5] = Bz
        W[6] = By
        W[7] = p  (thermal pressure)
        W[8] = psi
    """
    
    RHO= 0
    MX = 1;  VX = 1
    MZ = 2;  VZ= 2
    MY= 3;  VY = 3
    BX = 4
    BZ = 5
    BY= 6
    EN = 7;  PR = 7
    PSI= 8
    
    def __init__(self, config: SolverConfig):
        self.cfg = config
        self.Q= np.zeros((config.nvar, config.nx_tot, config.nz_tot), dtype=np.float64)
        
        # coord arrays
        self.x = np.linspace(
            config.x_min - (config.nghost - 0.5) * config.dx,
            config.x_max + (config.nghost - 0.5) * config.dx,
            config.nx_tot
        )
        self.z = np.linspace(
            config.z_min - (config.nghost - 0.5) * config.dz,
            config.z_max + (config.nghost - 0.5) * config.dz,
            config.nz_tot
        )
        
        # 2d grid setup
        self.X, self.Z= np.meshgrid(self.x, self.z, indexing='ij')
  
    def cons_to_prim(self, Q: Optional[np.ndarray] = None) -> np.ndarray:
        """
            E = p/(γ-1) + 0.5*ρ*(vx²+vy²+vz²) + 0.5*(Bx²+By²+Bz²)
            p = (γ-1) * [E - 0.5*ρ*v² - 0.5*B²]
        """
        if Q is None:
            Q = self.Q
            
        W = np.empty_like(Q)
        
        rho = Q[self.RHO]
        # floor density to avoid division by zero
        rho = np.maximum(rho, 1e-12)
        
        inv_rho = 1.0 / rho
        
        W[self.RHO] = rho
        W[self.VX] = Q[self.MX] * inv_rho
        W[self.VZ]= Q[self.MZ] * inv_rho
        W[self.VY] = Q[self.MY] * inv_rho
        W[self.BX] = Q[self.BX]
        W[self.BZ]= Q[self.BZ]
        W[self.BY] = Q[self.BY]
        W[self.PSI] = Q[self.PSI]
        
        vx= W[self.VX]
        vz = W[self.VZ]
        vy= W[self.VY]
        Bx = W[self.BX]
        Bz= W[self.BZ]
        By = W[self.BY]
        
        v_sq= vx**2 + vy**2 + vz**2
        B_sq= Bx**2 + By**2 + Bz**2
        
        kinetic = 0.5 * rho * v_sq
        magnetic = 0.5 * B_sq
        
        p = (self.cfg.gamma - 1.0) * (Q[self.EN] - kinetic - magnetic)
        
        # pressure floor
        W[self.PR] = np.maximum(p, 1e-12)
        
        return W
    
    def prim_to_cons(self, W: np.ndarray) -> np.ndarray:
        """
        convert primitive vars W -> conservative vars Q.
        
        E = p/(γ-1) + 0.5*ρ*v² + 0.5*B²
        """
        Q= np.empty_like(W)
        
        rho= W[self.RHO]
        vx = W[self.VX]
        vz = W[self.VZ]
        vy = W[self.VY]
        Bx= W[self.BX]
        Bz = W[self.BZ]
        By = W[self.BY]
        p = W[self.PR]
        psi = W[self.PSI]
        
        Q[self.RHO]= rho
        Q[self.MX] = rho * vx
        Q[self.MZ] = rho * vz
        Q[self.MY]= rho * vy
        Q[self.BX] = Bx
        Q[self.BZ] = Bz
        Q[self.BY] = By
        Q[self.PSI]= psi
        
        v_sq = vx**2 + vy**2 + vz**2
        B_sq = Bx**2 + By**2 + Bz**2
        
        Q[self.EN]= p / (self.cfg.gamma - 1.0) + 0.5 * rho * v_sq + 0.5 * B_sq
        
        return Q
    
    def compute_fast_speed(self, W: np.ndarray) -> np.ndarray:
        """
        magnetosonic speed calculation
        
        c_f² = 0.5 * [ (a² + b²) + sqrt( (a² + b²)² - 4*a²*bx² ) ]
        
        where:
            a² = γp/ρ (sound speed squared)
            b² = B²/ρ  (total alfven speed squared)
            bx = Bx/√ρ (normal alfven speed)
        """
        rho= np.maximum(W[self.RHO], 1e-12)
        
        p= np.maximum(W[self.PR], 1e-12)
        Bx= W[self.BX]
        By = W[self.BY]
        Bz= W[self.BZ]
        
        a_sq = self.cfg.gamma * p / rho
        B_sq= Bx**2 + By**2 + Bz**2
        
        # total alfven squared
        b_sq = B_sq / rho 
        
        bx_sq= Bx**2 / rho
        
        sum_ab = a_sq + b_sq
        discriminant= sum_ab**2 - 4.0 * a_sq * bx_sq
        discriminant = np.maximum(discriminant, 0.0)
        
        cf_sq = 0.5 * (sum_ab + np.sqrt(discriminant))
        
        return np.sqrt(np.maximum(cf_sq, 0.0))


# -------------------------------------------------------------
# Ideal MHD fluxes
# -------------------------------------------------------------

def compute_flux_x(W: np.ndarray, Q: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    """
    F[rho*vx] = rho*vx² + p_tot - Bx²
    F[rho*vz] = rho*vx*vz - Bx*Bz
    F[rho*vy] = rho*vx*vy - Bx*By
    F[Bx]     = psi (for glm)
    F[Bz]     = Bz*vx - Bx*vz
    F[By]     = By*vx - Bx*vy
    F[E]      = (E + p_tot)*vx - Bx*(v·B)
    F[psi]    = ch² * Bx (also for glm)
    
    p_tot = p + 0.5*B²
    """
    S= MHDState
    
    rho= W[S.RHO]
    vx= W[S.VX]
    vz = W[S.VZ]
    vy = W[S.VY]
    Bx = W[S.BX]
    Bz= W[S.BZ]
    By = W[S.BY]
    p = W[S.PR]
    psi = W[S.PSI]
    E= Q[S.EN]
    
    B_sq = Bx**2 + By**2 + Bz**2
    p_tot= p + 0.5 * B_sq
    v_dot_B= vx * Bx + vy * By + vz * Bz
    
    F = np.empty_like(Q)
    
    F[S.RHO] = rho * vx
    F[S.MX]= rho * vx * vx + p_tot - Bx * Bx
    F[S.MZ] = rho * vx * vz - Bx * Bz
    F[S.MY] = rho * vx * vy - Bx * By
    
    # glm
    F[S.BX]= psi 
    F[S.BZ] = Bz * vx - Bx * vz
    F[S.BY]= By * vx - Bx * vy
    F[S.EN] = (E + p_tot) * vx - Bx * v_dot_B
    F[S.PSI]= cfg.ch**2 * Bx   # Glm
    
    return F


def compute_flux_z(W: np.ndarray, Q: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    """
    ideal mhd flux in the z direction
    
    G[rho]    = rho * vz
    G[rho*vx] = rho*vz*vx - Bz*Bx
    G[rho*vz] = rho*vz² + p_tot - Bz²
    G[rho*vy] = rho*vz*vy - Bz*By
    G[Bx] = Bx*vz - Bz*vx
    G[Bz] = psi  (GLM)
    G[By]= By*vz - Bz*vy
    G[E] = (E + p_tot)*vz - Bz*(v·B)
    G[psi] = ch² * Bz  (GLM)
    """
    S = MHDState
    
    rho = W[S.RHO]
    vx = W[S.VX]
    vz= W[S.VZ]
    vy= W[S.VY]
    Bx = W[S.BX]
    Bz= W[S.BZ]
    By= W[S.BY]
    p = W[S.PR]
    psi= W[S.PSI]
    E = Q[S.EN]
    
    B_sq = Bx**2 + By**2 + Bz**2
    p_tot= p + 0.5 * B_sq
    v_dot_B = vx * Bx + vy * By + vz * Bz
    
    G= np.empty_like(Q)
    
    G[S.RHO] = rho * vz
    G[S.MX] = rho * vz * vx - Bz * Bx
    G[S.MZ]= rho * vz * vz + p_tot - Bz * Bz
    G[S.MY]= rho * vz * vy - Bz * By
    G[S.BX] = Bx * vz - Bz * vx
    
    # glm stuff
    G[S.BZ] = psi                    
    G[S.BY] = By * vz - Bz * vy
    G[S.EN]= (E + p_tot) * vz - Bz * v_dot_B
    G[S.PSI]= cfg.ch**2 * Bz
    
    return G

# -------------------------------------------------------------
# section 3: HLLD Riemann solver
# -------------------------------------------------------------
# Reference: Miyoshi & Kusano, J. Comput. Phys. 208 (2005) 315-344
# resolves all 7 MHD characteristic waves:
# S_L, S_L*, S_M, S_R*, S_R plus the two slow waves

def hlld_riemann_solver_x(
    WL: np.ndarray, WR: np.ndarray,
    QL: np.ndarray, QR: np.ndarray,
    cfg: SolverConfig
) -> np.ndarray:
    """
    hlld approximate Riemann solver for the x-direction.
    """
    S = MHDState
    gamma= cfg.gamma
    
    # step 0: extract primitive variables
    rhoL = np.maximum(WL[S.RHO], 1e-12)
    vxL= WL[S.VX]
    vzL = WL[S.VZ]
    vyL= WL[S.VY]
    BxL = WL[S.BX]
    BzL= WL[S.BZ]
    ByL= WL[S.BY]
    pL = np.maximum(WL[S.PR], 1e-12)
    psiL = WL[S.PSI]
    
    rhoR= np.maximum(WR[S.RHO], 1e-12)
    vxR = WR[S.VX]
    vzR = WR[S.VZ]
    vyR = WR[S.VY]
    BxR= WR[S.BX]
    BzR = WR[S.BZ]
    ByR= WR[S.BY]
    pR = np.maximum(WR[S.PR], 1e-12)
    psiR= WR[S.PSI]
    
    EL = QL[S.EN]
    ER= QR[S.EN]
    
    # normal magnetic field, use arithmetic average at interface
    Bn= 0.5 * (BxL + BxR)
    
    # step 1: compute wave speeds S_L, S_R
    BsqL = BxL**2 + ByL**2 + BzL**2
    BsqR= BxR**2 + ByR**2 + BzR**2
    
    a2L = gamma * pL / rhoL    # sound speed squared
    a2R= gamma * pR / rhoR
    
    bnL_sq = Bn**2 / rhoL
    bnR_sq= Bn**2 / rhoR
    
    bsqL = BsqL / rhoL
    bsqR= BsqR / rhoR
    
    # fast speed
    sumL = a2L + bsqL
    sumR= a2R + bsqR
    
    discL = np.maximum(sumL**2 - 4.0 * a2L * bnL_sq, 0.0)
    discR= np.maximum(sumR**2 - 4.0 * a2R * bnR_sq, 0.0)
    
    cfL = np.sqrt(0.5 * (sumL + np.sqrt(discL)))
    cfR= np.sqrt(0.5 * (sumR + np.sqrt(discR)))
    
    # Davis wave speed estimates
    SL = np.minimum(vxL - cfL, vxR - cfR)
    SR= np.maximum(vxL + cfL, vxR + cfR)
    
    # step 2: compute contact wave speed S_M and total pressure in star region
    ptL = pL + 0.5 * BsqL   
    ptR= pR + 0.5 * BsqR
    
    num_SM= (rhoR * vxR * (SR - vxR) - rhoL * vxL * (SL - vxL) - ptR + ptL)
    den_SM= (rhoR * (SR - vxR) - rhoL * (SL - vxL))
    den_SM = np.where(np.abs(den_SM) < 1e-14, 1e-14 * np.sign(den_SM + 1e-30), den_SM)
    SM= num_SM / den_SM
    
    # total pressure in the star region
    pT_star = ptL + rhoL * (SL - vxL) * (SM - vxL)
    pT_star= np.maximum(pT_star, 1e-14)
    
    # step 3: compute the star states U*_L and U*_R
    def compute_star_state(rho, vx, vz, vy, Bx_in, Bz, By, E, p_tot, SX, SM_val, pT_s, Bn_val):
        
        denom = SX - SM_val
        denom= np.where(np.abs(denom) < 1e-14, 1e-14 * np.sign(denom + 1e-30), denom)
        inv_denom= 1.0 / denom
        
        # density in star region
        rho_s = rho * (SX - vx) * inv_denom
        rho_s= np.maximum(rho_s, 1e-12)
        
        vx_s= SM_val
        
        # tangential velocities
        Bn_sq = Bn_val**2
        
        vz_s = vz - Bz * Bn_val * (SM_val - vx) * inv_denom / \
               np.where(np.abs(rho * (SX - vx) * (SX - SM_val) - Bn_sq) < 1e-14,
                        1e-14, rho * (SX - vx) * (SX - SM_val) - Bn_sq)
        
        vy_s = vy - By * Bn_val * (SM_val - vx) * inv_denom / \
               np.where(np.abs(rho * (SX - vx) * (SX - SM_val) - Bn_sq) < 1e-14,
                        1e-14, rho * (SX - vx) * (SX - SM_val) - Bn_sq)
        
        # tangential B-fields
        coeff= (rho * (SX - vx)**2 - Bn_sq) / \
                np.where(np.abs(rho * (SX - vx) * (SX - SM_val) - Bn_sq) < 1e-14,
                         1e-14, rho * (SX - vx) * (SX - SM_val) - Bn_sq)
        
        Bz_s= Bz * coeff
        By_s = By * coeff
        
        # energy in star region
        vB = vx * Bn_val + vz * Bz + vy * By
        vB_s= SM_val * Bn_val + vz_s * Bz_s + vy_s * By_s
        
        E_s = ((SX - vx) * E - p_tot * vx + pT_s * SM_val + Bn_val * (vB - vB_s)) * inv_denom
        
        U_star = np.empty_like(Q_template)
        U_star[S.RHO]= rho_s
        U_star[S.MX] = rho_s * vx_s
        U_star[S.MZ]= rho_s * vz_s
        U_star[S.MY]= rho_s * vy_s
        U_star[S.BX]= Bn_val
        U_star[S.BZ]= Bz_s
        U_star[S.BY]= By_s
        U_star[S.EN] = E_s
        U_star[S.PSI] = 0.0  # glm psi handled separately
        
        return U_star, rho_s, vx_s, vz_s, vy_s, Bz_s, By_s
    
    Q_template= QL
    
    UstarL, rhoSL, vxSL, vzSL, vySL, BzSL, BySL = compute_star_state(
        rhoL, vxL, vzL, vyL, BxL, BzL, ByL, EL, ptL, SL, SM, pT_star, Bn)
    
    UstarR, rhoSR, vxSR, vzSR, vySR, BzSR, BySR= compute_star_state(
        rhoR, vxR, vzR, vyR, BxR, BzR, ByR, ER, ptR, SR, SM, pT_star, Bn)
    
    # step 4: compute alfven wave speeds S*_L, S*_R and double-star states
    sqrt_rhoSL= np.sqrt(np.maximum(rhoSL, 1e-12))
    sqrt_rhoSR = np.sqrt(np.maximum(rhoSR, 1e-12))
    
    abs_Bn= np.abs(Bn)
    
    SstarL= SM - abs_Bn / (sqrt_rhoSL + 1e-30)
    SstarR = SM + abs_Bn / (sqrt_rhoSR + 1e-30)
    
    inv_sum_sqrt = 1.0 / (sqrt_rhoSL + sqrt_rhoSR + 1e-30)
    sign_Bn= np.sign(Bn)
    
    vzSS = (sqrt_rhoSL * vzSL + sqrt_rhoSR * vzSR + (BzSR - BzSL) * sign_Bn) * inv_sum_sqrt
    vySS= (sqrt_rhoSL * vySL + sqrt_rhoSR * vySR + (BySR - BySL) * sign_Bn) * inv_sum_sqrt
    
    BzSS= (sqrt_rhoSL * BzSR + sqrt_rhoSR * BzSL + 
            sqrt_rhoSL * sqrt_rhoSR * (vzSR - vzSL) * sign_Bn) * inv_sum_sqrt
    
    BySS = (sqrt_rhoSL * BySR + sqrt_rhoSR * BySL + 
            sqrt_rhoSL * sqrt_rhoSR * (vySR - vySL) * sign_Bn) * inv_sum_sqrt
    
    vB_starL= SM * Bn + vzSL * BzSL + vySL * BySL
    vB_SS= SM * Bn + vzSS * BzSS + vySS * BySS
    
    E_ssL = UstarL[S.EN] - sqrt_rhoSL * sign_Bn * (vB_starL - vB_SS)
    
    vB_starR= SM * Bn + vzSR * BzSR + vySR * BySR
    E_ssR = UstarR[S.EN] + sqrt_rhoSR * sign_Bn * (vB_starR - vB_SS)
    
    UssL= np.empty_like(QL)
    UssL[S.RHO] = rhoSL
    UssL[S.MX] = rhoSL * SM
    UssL[S.MZ]= rhoSL * vzSS
    UssL[S.MY]= rhoSL * vySS
    UssL[S.BX]= Bn
    UssL[S.BZ]= BzSS
    UssL[S.BY]= BySS
    UssL[S.EN] = E_ssL
    UssL[S.PSI] = 0.0
    
    UssR = np.empty_like(QR)
    UssR[S.RHO]= rhoSR
    UssR[S.MX] = rhoSR * SM
    UssR[S.MZ] = rhoSR * vzSS
    UssR[S.MY]= rhoSR * vySS
    UssR[S.BX]= Bn
    UssR[S.BZ]= BzSS
    UssR[S.BY]= BySS
    UssR[S.EN] = E_ssR
    UssR[S.PSI] = 0.0
    
    # step 5: compute the physical flux
    FL = compute_flux_x(WL, QL, cfg)
    FR = compute_flux_x(WR, QR, cfg)
    
    FstarL= FL + SL[np.newaxis, ...] * (UstarL - QL)
    FstarR = FR + SR[np.newaxis, ...] * (UstarR - QR)
    
    FssL= FstarL + SstarL[np.newaxis, ...] * (UssL - UstarL)
    FssR= FstarR + SstarR[np.newaxis, ...] * (UssR - UstarR)
    
    # step 6: select correct flux
    F_hlld= np.copy(FL)
    
    for n in range(cfg.nvar):
        F_hlld[n] = np.where(
            SL >= 0.0,
            FL[n],
            np.where(
                SstarL >= 0.0,
                FstarL[n],
                np.where(
                    SM >= 0.0,
                    FssL[n],
                    np.where(
                        SstarR >= 0.0,
                        FssR[n],
                        np.where(
                            SR >= 0.0,
                            FstarR[n],
                            FR[n]
                        )
                    )
                )
            )
        )
    
    # step 7: glm flux correction
    ch = cfg.ch
    if ch > 0:
        Bx_avg= 0.5 * (BxL + BxR)
        psi_avg= 0.5 * (psiL + psiR)
        Bx_jump= BxR - BxL
        psi_jump= psiR - psiL
        
        F_hlld[S.BX] = psi_avg - 0.5 * ch * Bx_jump
        F_hlld[S.PSI]= ch**2 * Bx_avg - 0.5 * ch * psi_jump
    
    return F_hlld


def hlld_riemann_solver_z(
    WL: np.ndarray, WR: np.ndarray,
    QL: np.ndarray, QR: np.ndarray,
    cfg: SolverConfig
) -> np.ndarray:
    """
    hlld approximate Riemann solver for the z-direction.
    """
    S = MHDState
    
    # rotated copies
    WL_rot= np.copy(WL)
    WR_rot = np.copy(WR)
    QL_rot= np.copy(QL)
    QR_rot= np.copy(QR)
    
    WL_rot[S.VX], WL_rot[S.VZ]= WL[S.VZ].copy(), WL[S.VX].copy()
    WR_rot[S.VX], WR_rot[S.VZ]= WR[S.VZ].copy(), WR[S.VX].copy()
    
    QL_rot[S.MX], QL_rot[S.MZ] = QL[S.MZ].copy(), QL[S.MX].copy()
    QR_rot[S.MX], QR_rot[S.MZ] = QR[S.MZ].copy(), QR[S.MX].copy()
    
    WL_rot[S.BX], WL_rot[S.BZ]= WL[S.BZ].copy(), WL[S.BX].copy()
    WR_rot[S.BX], WR_rot[S.BZ]= WR[S.BZ].copy(), WR[S.BX].copy()
    QL_rot[S.BX], QL_rot[S.BZ]= QL[S.BZ].copy(), QL[S.BX].copy()
    QR_rot[S.BX], QR_rot[S.BZ]= QR[S.BZ].copy(), QR[S.BX].copy()
    
    # solve in rotated frame
    F_rot = hlld_riemann_solver_x(WL_rot, WR_rot, QL_rot, QR_rot, cfg)
    
    # rotate flux back
    F_out= np.copy(F_rot)
    F_out[S.MX], F_out[S.MZ]= F_rot[S.MZ].copy(), F_rot[S.MX].copy()
    F_out[S.BX], F_out[S.BZ]= F_rot[S.BZ].copy(), F_rot[S.BX].copy()
    
    return F_out


# -------------------------------------------------------------
# section 4: muscl reconstruction with tvd limiters
# -------------------------------------------------------------

def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    MinMod limiter
    """
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def vanleer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    van Leer limiter
    """
    abs_a= np.abs(a)
    abs_b= np.abs(b)
    return (abs_a * b + abs_b * a) / (abs_a + abs_b + 1e-30)


def muscl_reconstruct_x(W: np.ndarray, cfg: SolverConfig, 
                         limiter: str = 'vanleer') -> Tuple[np.ndarray, np.ndarray]:
    """
    2nd-order muscl reconstruction in x-direction.
    """
    ng= cfg.nghost
    lim_func = vanleer if limiter == 'vanleer' else minmod
    
    # slopes
    dW = W[:, 1:, :] - W[:, :-1, :]
    
    # limited slopes
    slopes = lim_func(dW[:, :-1, :], dW[:, 1:, :])
    
    n_int= slopes.shape[1] - 1
    
    WL_rec= W[:, 1:1+n_int, :] + 0.5 * slopes[:, :n_int, :]
    WR_rec = W[:, 2:2+n_int, :] - 0.5 * slopes[:, 1:1+n_int, :]
    
    # enforce positivity
    WL_rec[MHDState.RHO] = np.maximum(WL_rec[MHDState.RHO], 1e-12)
    WR_rec[MHDState.RHO]= np.maximum(WR_rec[MHDState.RHO], 1e-12)
    WL_rec[MHDState.PR]= np.maximum(WL_rec[MHDState.PR], 1e-12)
    WR_rec[MHDState.PR] = np.maximum(WR_rec[MHDState.PR], 1e-12)
    
    return WL_rec, WR_rec


def muscl_reconstruct_z(W: np.ndarray, cfg: SolverConfig,
                         limiter: str = 'vanleer') -> Tuple[np.ndarray, np.ndarray]:
    """
    2nd-order muscl reconstruction in z-direction.
    """
    ng = cfg.nghost
    lim_func= vanleer if limiter == 'vanleer' else minmod
    
    dW= W[:, :, 1:] - W[:, :, :-1]
    slopes = lim_func(dW[:, :, :-1], dW[:, :, 1:])
    
    n_int = slopes.shape[2] - 1
    
    WL_rec = W[:, :, 1:1+n_int] + 0.5 * slopes[:, :, :n_int]
    WR_rec= W[:, :, 2:2+n_int] - 0.5 * slopes[:, :, 1:1+n_int]
    
    WL_rec[MHDState.RHO]= np.maximum(WL_rec[MHDState.RHO], 1e-12)
    WR_rec[MHDState.RHO]= np.maximum(WR_rec[MHDState.RHO], 1e-12)
    WL_rec[MHDState.PR]= np.maximum(WL_rec[MHDState.PR], 1e-12)
    WR_rec[MHDState.PR]= np.maximum(WR_rec[MHDState.PR], 1e-12)
    
    return WL_rec, WR_rec

# -------------------------------------------------------------
# section 5: glm source terms
# -------------------------------------------------------------

def glm_source_update(Q: np.ndarray, cfg: SolverConfig, dt: float) -> np.ndarray:
    """
    apply the glm divergence cleaning source term.
    """
    S = MHDState
    
    if cfg.ch > 0:
        # parabolic damping coefficient
        dx_min = min(cfg.dx, cfg.dz)
        cp_sq= cfg.ch * dx_min / cfg.cr
        
        # exponential decay factor
        decay = np.exp(-cfg.ch**2 * dt / (cp_sq + 1e-30))
        
        Q[S.PSI] *= decay
    
    return Q


def update_glm_speed(W: np.ndarray, state: MHDState, cfg: SolverConfig):
    """
    update the glm cleaning speed c_h to be the max fast 
    magnetosonic speed on the domain.
    """
    cf= state.compute_fast_speed(W)
    
    vx = np.abs(W[MHDState.VX])
    vz= np.abs(W[MHDState.VZ])
    
    max_speed_x = np.max(vx + cf)
    max_speed_z = np.max(vz + cf)
    
    cfg.ch= max(max_speed_x, max_speed_z)
    
    return max(max_speed_x, max_speed_z)


# -------------------------------------------------------------
# section 6: spatial operator (rhs) — finite volume flux differencing
# -------------------------------------------------------------

def compute_rhs(Q: np.ndarray, state: MHDState, cfg: SolverConfig, limiter: str = 'vanleer') -> np.ndarray:
    """
    compute the right-hand side of the semi-discrete finite volume scheme
    """
    S= MHDState
    ng = cfg.nghost
    
    # convert to primitives for reconstruction
    W = state.cons_to_prim(Q)
    
    # x-direction fluxes
    WL_x, WR_x = muscl_reconstruct_x(W, cfg, limiter)
    
    QL_x= state.prim_to_cons(WL_x)
    QR_x = state.prim_to_cons(WR_x)
    
    Fx = hlld_riemann_solver_x(WL_x, WR_x, QL_x, QR_x, cfg)
    
    # z-direction fluxes
    WL_z, WR_z= muscl_reconstruct_z(W, cfg, limiter)
    
    QL_z= state.prim_to_cons(WL_z)
    QR_z = state.prim_to_cons(WR_z)
    
    Gz = hlld_riemann_solver_z(WL_z, WR_z, QL_z, QR_z, cfg)
    
    # flux differencing: L(Q) = -dF/dx - dG/dz
    nfx = Fx.shape[1]  
    nfz= Gz.shape[2]  
    
    L= np.zeros_like(Q)
    
    nx_cells = cfg.nx
    nz_cells= cfg.nz
    
    # x-flux differencing for interior cells
    i_start = ng          
    i_end= ng + nx_cells  
    
    j_start = ng
    j_end = ng + nz_cells
    
    dFx= (Fx[:, i_start-1:i_end-1, :] - Fx[:, i_start-2:i_end-2, :]) / cfg.dx
    
    # z-flux differencing
    dGz= (Gz[:, :, j_start-1:j_end-1] - Gz[:, :, j_start-2:j_end-2]) / cfg.dz
    
    L[:, i_start:i_end, j_start:j_end] = (
        -dFx[:, :, j_start:j_end] 
        -dGz[:, i_start:i_end, :]
    )
    
    return L


# -------------------------------------------------------------
# section 7: ssp-rk3 time integrator
# -------------------------------------------------------------

def compute_dt(state: MHDState, cfg: SolverConfig, W: np.ndarray) -> float:
    """
    compute the stable time step from cfl condition.
    """
    ng= cfg.nghost
    
    # only consider interior cells
    W_int = W[:, ng:-ng, ng:-ng]
    
    cf = state.compute_fast_speed(W_int)
    
    max_speed_x = np.max(np.abs(W_int[MHDState.VX]) + cf)
    max_speed_z= np.max(np.abs(W_int[MHDState.VZ]) + cf)
    
    inv_dt = max_speed_x / cfg.dx + max_speed_z / cfg.dz
    
    if inv_dt < 1e-30:
        return 1e-6  # fallback for quiescent initial conditions
    
    dt = cfg.cfl / inv_dt
    
    return dt


def ssp_rk3_step(state: MHDState, cfg: SolverConfig, dt: float, bc_func, limiter: str = 'vanleer') -> np.ndarray:
    """
    3rd-order strong stability preserving runge-kutta (ssp-rk3).
    """
    Qn= np.copy(state.Q)
    
    # stage 1
    L1 = compute_rhs(Qn, state, cfg, limiter)
    Q1= Qn + dt * L1
    Q1 = glm_source_update(Q1, cfg, dt)
    bc_func(Q1, cfg, state.current_time + dt)
    
    # stage 2
    L2 = compute_rhs(Q1, state, cfg, limiter)
    Q2= 0.75 * Qn + 0.25 * (Q1 + dt * L2)
    Q2 = glm_source_update(Q2, cfg, 0.25 * dt)
    bc_func(Q2, cfg, state.current_time + 0.5 * dt)
    
    # stage 3
    L3= compute_rhs(Q2, state, cfg, limiter)
    Q_new= (1.0/3.0) * Qn + (2.0/3.0) * (Q2 + dt * L3)
    Q_new = glm_source_update(Q_new, cfg, (2.0/3.0) * dt)
    bc_func(Q_new, cfg, state.current_time + dt)
    
    return Q_new


# -------------------------------------------------------------
# section 8: gold-hoyle plectoneme initialiser
# -------------------------------------------------------------

def gold_hoyle_initialiser(state: MHDState, cfg: SolverConfig, rho0: float = 1.0, p0: float = 1.0, B0: float = 1.0, a: float = 0.3, twist: float = 2.0) -> None:
    """
    initialise the magnetic field as a gold-hoyle twisted flux tube.
    """
    S = MHDState
    
    x_centre = 0.5 * (cfg.x_max + cfg.x_min)
    
    X= state.X   
    Z= state.Z
    
    # radial dist from tube axis
    r = np.abs(X - x_centre)
    
    alpha= twist / a   
    
    alpha_r_sq = (alpha * r)**2
    denom = 1.0 + alpha_r_sq
    
    # gold-hoyle field components
    Bz_field= B0 / denom                    
    By_field = B0 * alpha * r / denom          
    Bx_field= np.zeros_like(r)                
    
    B_sq = Bx_field**2 + By_field**2 + Bz_field**2
    
    # force-free equilibrium
    p= p0 * np.ones_like(r)
    rho = rho0 * np.ones_like(r)
    
    vx = np.zeros_like(r)
    vz = np.zeros_like(r)
    vy= np.zeros_like(r)
    
    W = np.zeros_like(state.Q)
    W[S.RHO]= rho
    W[S.VX] = vx
    W[S.VZ] = vz
    W[S.VY] = vy
    W[S.BX]= Bx_field
    W[S.BZ] = Bz_field
    W[S.BY]= By_field
    W[S.PR]= p
    W[S.PSI] = np.zeros_like(r)
    
    state.Q= state.prim_to_cons(W)
    
    print(f"Plectoneme ready: alpha={alpha:.3f}, center={x_centre:.2f}")


# -------------------------------------------------------------
# section 9: boundary conditions 
# -------------------------------------------------------------

def apply_boundary_conditions(Q: np.ndarray, cfg: SolverConfig, t: float, peristaltic: bool = True, P0_wall: float = 0.5, A_wall: float = 0.2, k_wall: float = 2.0 * np.pi, v_wall: float = 1.0) -> None:
    """
    apply boundary conditions including peristaltic magnetic compression.
    """
    S = MHDState
    ng= cfg.nghost
    nx = cfg.nx
    nz = cfg.nz
    
    # z-boundaries: periodic
    Q[:, :, 0:ng] = Q[:, :, nz:nz+ng]
    Q[:, :, nz+ng:nz+2*ng]= Q[:, :, ng:2*ng]
    
    # x-boundaries: peristaltic magnetic wall
    if peristaltic:
        z_ghost = np.linspace(cfg.z_min - (ng - 0.5) * cfg.dz, cfg.z_max + (ng - 0.5) * cfg.dz, cfg.nz_tot)
        
        P_mag_target= P0_wall + A_wall * np.sin(k_wall * (z_ghost - v_wall * t))
        P_mag_target= np.maximum(P_mag_target, 1e-6) 
        
        # left wall
        for ig in range(ng):
            idx_ghost = ng - 1 - ig     
            idx_interior = ng + ig      
            
            Q[:, idx_ghost, :] = Q[:, idx_interior, :]
            Q[S.MX, idx_ghost, :]= -Q[S.MX, idx_interior, :]
            Q[S.BX, idx_ghost, :] = -Q[S.BX, idx_interior, :]
            
            Bx_g = Q[S.BX, idx_ghost, :]
            Bz_g= Q[S.BZ, idx_ghost, :]
        
            B_existing_sq = Bx_g**2 + Bz_g**2
            By_sq_needed= 2.0 * P_mag_target - B_existing_sq
            By_sq_needed = np.maximum(By_sq_needed, 0.0)
            
            By_sign = np.sign(Q[S.BY, idx_interior, :] + 1e-30)
            Q[S.BY, idx_ghost, :]= By_sign * np.sqrt(By_sq_needed)
            
            rho_g = np.maximum(Q[S.RHO, idx_ghost, :], 1e-12)
            v_sq = (Q[S.MX, idx_ghost, :]**2 + Q[S.MZ, idx_ghost, :]**2 + Q[S.MY, idx_ghost, :]**2) / rho_g**2
            B_sq_new = Q[S.BX, idx_ghost, :]**2 + Q[S.BZ, idx_ghost, :]**2 + Q[S.BY, idx_ghost, :]**2
            
            rho_int = np.maximum(Q[S.RHO, idx_interior, :], 1e-12)
            v_sq_int = (Q[S.MX, idx_interior, :]**2 + Q[S.MZ, idx_interior, :]**2 + Q[S.MY, idx_interior, :]**2) / rho_int**2
            B_sq_int = (Q[S.BX, idx_interior, :]**2 + Q[S.BZ, idx_interior, :]**2 + Q[S.BY, idx_interior, :]**2)
            
            p_int = (cfg.gamma - 1.0) * (Q[S.EN, idx_interior, :] - 0.5 * rho_int * v_sq_int - 0.5 * B_sq_int)
            p_int= np.maximum(p_int, 1e-12)
            
            Q[S.EN, idx_ghost, :] = (p_int / (cfg.gamma - 1.0) + 0.5 * rho_g * v_sq + 0.5 * B_sq_new)
        
        # right wall
        for ig in range(ng):
            idx_ghost= ng + nx + ig          
            idx_interior = ng + nx - 1 - ig 
            
            Q[:, idx_ghost, :]= Q[:, idx_interior, :]
            Q[S.MX, idx_ghost, :] = -Q[S.MX, idx_interior, :]
            Q[S.BX, idx_ghost, :]= -Q[S.BX, idx_interior, :]
            
            Bx_g = Q[S.BX, idx_ghost, :]
            Bz_g = Q[S.BZ, idx_ghost, :]
            
            B_existing_sq = Bx_g**2 + Bz_g**2
            By_sq_needed= 2.0 * P_mag_target - B_existing_sq
            By_sq_needed = np.maximum(By_sq_needed, 0.0)
            By_sign= np.sign(Q[S.BY, idx_interior, :] + 1e-30)
            
            Q[S.BY, idx_ghost, :] = By_sign * np.sqrt(By_sq_needed)
            
            rho_g= np.maximum(Q[S.RHO, idx_ghost, :], 1e-12)
            v_sq = (Q[S.MX, idx_ghost, :]**2 + Q[S.MZ, idx_ghost, :]**2 + Q[S.MY, idx_ghost, :]**2) / rho_g**2
            B_sq_new= (Q[S.BX, idx_ghost, :]**2 + Q[S.BZ, idx_ghost, :]**2 + Q[S.BY, idx_ghost, :]**2)
            
            rho_int = np.maximum(Q[S.RHO, idx_interior, :], 1e-12)
            v_sq_int= (Q[S.MX, idx_interior, :]**2 + Q[S.MZ, idx_interior, :]**2 + Q[S.MY, idx_interior, :]**2) / rho_int**2
            B_sq_int = (Q[S.BX, idx_interior, :]**2 + Q[S.BZ, idx_interior, :]**2 + Q[S.BY, idx_interior, :]**2)
            
            p_int = (cfg.gamma - 1.0) * (Q[S.EN, idx_interior, :] - 0.5 * rho_int * v_sq_int - 0.5 * B_sq_int)
            p_int = np.maximum(p_int, 1e-12)
            
            Q[S.EN, idx_ghost, :]= (p_int / (cfg.gamma - 1.0) + 0.5 * rho_g * v_sq + 0.5 * B_sq_new)
    else:
        # simple reflective BCs 
        for ig in range(ng):
            # left
            Q[:, ng-1-ig, :] = Q[:, ng+ig, :]
            Q[S.MX, ng-1-ig, :] = -Q[S.MX, ng+ig, :]
            Q[S.BX, ng-1-ig, :]= -Q[S.BX, ng+ig, :]
            
            # right
            Q[:, ng+nx+ig, :] = Q[:, ng+nx-1-ig, :]
            Q[S.MX, ng+nx+ig, :]= -Q[S.MX, ng+nx-1-ig, :]
            Q[S.BX, ng+nx+ig, :] = -Q[S.BX, ng+nx-1-ig, :]


# -------------------------------------------------------------
# section 10: canonical helicity tracker
# -------------------------------------------------------------

def compute_magnetic_helicity(state: MHDState, cfg: SolverConfig) -> float:
    """
    compute the magnetic helicity
    """
    S= MHDState
    ng = cfg.nghost
    
    W = state.cons_to_prim()
    Bz= W[S.BZ, ng:-ng, ng:-ng]
    By = W[S.BY, ng:-ng, ng:-ng]
    Bx = W[S.BX, ng:-ng, ng:-ng]
    
    # compute Ay by integrating -Bz along x
    Ay = -np.cumsum(Bz, axis=0) * cfg.dx
    
    h_m= Ay * By
    H_m = np.sum(h_m) * cfg.dx * cfg.dz
    
    return H_m


def compute_canonical_helicity(state: MHDState, cfg: SolverConfig, m_ion: float = 1.0, q_ion: float = 1.0) -> tuple:
    """
    compute canonical helicity: K = ∫ P · (∇ × P) dV
    """
    S = MHDState
    ng= cfg.nghost
    
    W = state.cons_to_prim()
    
    vx = W[S.VX, ng:-ng, ng:-ng]
    vz= W[S.VZ, ng:-ng, ng:-ng]
    vy= W[S.VY, ng:-ng, ng:-ng]
    Bx = W[S.BX, ng:-ng, ng:-ng]
    Bz= W[S.BZ, ng:-ng, ng:-ng]
    By = W[S.BY, ng:-ng, ng:-ng]
    
    # term 3: magnetic helicity
    H_m= compute_magnetic_helicity(state, cfg)
    
    # term 2: cross helicity
    v_dot_B= vx * Bx + vz * Bz + vy * By
    H_cross = np.sum(v_dot_B) * cfg.dx * cfg.dz
    
    # term 1: kinetic helicity
    dvydz= np.zeros_like(vy)
    dvydx = np.zeros_like(vy)
    dvxdz= np.zeros_like(vx)
    dvzdx = np.zeros_like(vz)
    
    dvydz[:, 1:-1]= (vy[:, 2:] - vy[:, :-2]) / (2.0 * cfg.dz)
    dvydx[1:-1, :] = (vy[2:, :] - vy[:-2, :]) / (2.0 * cfg.dx)
    dvxdz[:, 1:-1]= (vx[:, 2:] - vx[:, :-2]) / (2.0 * cfg.dz)
    dvzdx[1:-1, :] = (vz[2:, :] - vz[:-2, :]) / (2.0 * cfg.dx)
    
    omega_x = dvydz
    omega_z= -dvydx
    omega_y = dvxdz - dvzdx
    
    v_dot_omega = vx * omega_x + vz * omega_z + vy * omega_y
    H_kin= np.sum(v_dot_omega) * cfg.dx * cfg.dz
    
    # total
    K = m_ion**2 * H_kin + 2.0 * m_ion * q_ion * H_cross + q_ion**2 * H_m
    
    return K, H_m, H_cross, H_kin


# -------------------------------------------------------------
# section 11: diagnostics and visualization
# -------------------------------------------------------------

def compute_divergence_B(state: MHDState, cfg: SolverConfig) -> np.ndarray:
    """
    compute ∇·B using central differences
    """
    S= MHDState
    ng = cfg.nghost
    
    Bx= state.Q[S.BX]
    Bz = state.Q[S.BZ]
    
    dBxdx = (Bx[2:, 1:-1] - Bx[:-2, 1:-1]) / (2.0 * cfg.dx)
    dBzdz= (Bz[1:-1, 2:] - Bz[1:-1, :-2]) / (2.0 * cfg.dz)
    
    min_nx = min(dBxdx.shape[0], dBzdz.shape[0])
    min_nz= min(dBxdx.shape[1], dBzdz.shape[1])
    
    divB = dBxdx[:min_nx, :min_nz] + dBzdz[:min_nx, :min_nz]
    
    return divB


class DiagnosticLogger:
    """
    tracks time-series of key physical quantities.
    """
    def __init__(self):
        self.time = []
        self.total_energy= []
        self.kinetic_energy= []
        self.magnetic_energy = []
        self.thermal_energy = []
        self.max_divB= []
        self.l2_divB = []
        self.canonical_helicity = []
        self.magnetic_helicity= []
        self.cross_helicity = []
        self.kinetic_helicity = []
        self.max_mach= []
        self.dt_history = []
    
    def log(self, t: float, state: MHDState, cfg: SolverConfig, dt: float):
        S= MHDState
        ng = cfg.nghost
        
        W = state.cons_to_prim()
        Q= state.Q
        
        rho= W[S.RHO, ng:-ng, ng:-ng]
        vx = W[S.VX,  ng:-ng, ng:-ng]
        vz = W[S.VZ,  ng:-ng, ng:-ng]
        vy = W[S.VY,  ng:-ng, ng:-ng]
        Bx= W[S.BX,  ng:-ng, ng:-ng]
        Bz= W[S.BZ,  ng:-ng, ng:-ng]
        By = W[S.BY,  ng:-ng, ng:-ng]
        p = W[S.PR,  ng:-ng, ng:-ng]
        
        dV= cfg.dx * cfg.dz
        
        v_sq = vx**2 + vy**2 + vz**2
        B_sq= Bx**2 + By**2 + Bz**2
        
        E_kin= 0.5 * np.sum(rho * v_sq) * dV
        E_mag = 0.5 * np.sum(B_sq) * dV
        E_therm = np.sum(p / (cfg.gamma - 1.0)) * dV
        E_tot = E_kin + E_mag + E_therm
        
        divB= compute_divergence_B(state, cfg)
        max_dB= np.max(np.abs(divB))
        l2_dB = np.sqrt(np.mean(divB**2))
        
        K, H_m, H_c, H_k = compute_canonical_helicity(state, cfg)
        
        cs = np.sqrt(cfg.gamma * p / (rho + 1e-30))
        mach= np.sqrt(v_sq) / (cs + 1e-30)
        
        self.time.append(t)
        self.total_energy.append(E_tot)
        self.kinetic_energy.append(E_kin)
        self.magnetic_energy.append(E_mag)
        self.thermal_energy.append(E_therm)
        self.max_divB.append(max_dB)
        self.l2_divB.append(l2_dB)
        self.canonical_helicity.append(K)
        self.magnetic_helicity.append(H_m)
        self.cross_helicity.append(H_c)
        self.kinetic_helicity.append(H_k)
        self.max_mach.append(np.max(mach))
        self.dt_history.append(dt)


def plot_state(state: MHDState, cfg: SolverConfig, t: float, logger: DiagnosticLogger, step: int):
    """
    generate a comprehensive diagnostic plot.
    """
    S = MHDState
    ng = cfg.nghost
    
    W= state.cons_to_prim()
    
    x_int = state.x[ng:-ng]
    z_int = state.z[ng:-ng]
    
    rho= W[S.RHO, ng:-ng, ng:-ng]
    p= W[S.PR,  ng:-ng, ng:-ng]
    Bz= W[S.BZ,  ng:-ng, ng:-ng]
    By = W[S.BY,  ng:-ng, ng:-ng]
    Bx = W[S.BX,  ng:-ng, ng:-ng]
    vx= W[S.VX,  ng:-ng, ng:-ng]
    vz = W[S.VZ,  ng:-ng, ng:-ng]
    
    B_mag= np.sqrt(Bx**2 + By**2 + Bz**2)
    P_mag = 0.5 * B_mag**2
    beta= 2.0 * p / (B_mag**2 + 1e-30)
    
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig= plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # row 1: field quantities
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(z_int, x_int, rho, shading='auto', cmap='inferno')
    ax1.set_title(f'Density ρ (t={t:.4f})', fontsize=10)
    ax1.set_xlabel('z'); ax1.set_ylabel('x')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    ax2= fig.add_subplot(gs[0, 1])
    im2= ax2.pcolormesh(z_int, x_int, B_mag, shading='auto', cmap='plasma')
    ax2.set_title('|B|', fontsize=10)
    ax2.set_xlabel('z'); ax2.set_ylabel('x')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.pcolormesh(z_int, x_int, By, shading='auto', cmap='RdBu_r')
    ax3.set_title('B_y (twist)', fontsize=10)
    ax3.set_xlabel('z'); ax3.set_ylabel('x')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    ax4= fig.add_subplot(gs[0, 3])
    im4= ax4.pcolormesh(z_int, x_int, np.log10(beta + 1e-30), shading='auto', cmap='coolwarm', vmin=-2, vmax=2)
    ax4.set_title('log10(β)', fontsize=10)
    ax4.set_xlabel('z'); ax4.set_ylabel('x')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # row 2: pressure and velocity
    ax5= fig.add_subplot(gs[1, 0])
    im5 = ax5.pcolormesh(z_int, x_int, p, shading='auto', cmap='viridis')
    ax5.set_title('Thermal Pressure p', fontsize=10)
    ax5.set_xlabel('z'); ax5.set_ylabel('x')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    ax6= fig.add_subplot(gs[1, 1])
    im6 = ax6.pcolormesh(z_int, x_int, P_mag, shading='auto', cmap='magma')
    ax6.set_title('Magnetic Pressure', fontsize=10)
    ax6.set_xlabel('z'); ax6.set_ylabel('x')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    ax7= fig.add_subplot(gs[1, 2])
    speed = np.sqrt(vx**2 + vz**2)
    im7= ax7.pcolormesh(z_int, x_int, speed, shading='auto', cmap='hot')
    ax7.set_title('|v|', fontsize=10)
    ax7.set_xlabel('z'); ax7.set_ylabel('x')
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    ax8= fig.add_subplot(gs[1, 3])
    skip = max(1, cfg.nx // 32)
    ax8.streamplot(z_int, x_int, Bz, Bx, color=B_mag, cmap='plasma', density=1.5, linewidth=0.8)
    ax8.set_title('B-field streamlines', fontsize=10)
    ax8.set_xlabel('z'); ax8.set_ylabel('x')
    ax8.set_xlim(z_int[0], z_int[-1])
    ax8.set_ylim(x_int[0], x_int[-1])
    
    # row 3: time histories
    if len(logger.time) > 1:
        ax9= fig.add_subplot(gs[2, 0])
        ax9.semilogy(logger.time, logger.total_energy, 'k-', label='Total', linewidth=1.5)
        ax9.semilogy(logger.time, logger.magnetic_energy, 'b-', label='Magnetic')
        ax9.semilogy(logger.time, logger.kinetic_energy, 'r-', label='Kinetic')
        ax9.semilogy(logger.time, logger.thermal_energy, 'g-', label='Thermal')
        ax9.set_title('Energy Evolution', fontsize=10)
        ax9.set_xlabel('t'); ax9.legend(fontsize=7)
        ax9.grid(True, alpha=0.3)
        
        ax10= fig.add_subplot(gs[2, 1])
        ax10.semilogy(logger.time, logger.max_divB, 'r-', label='max|∇·B|')
        ax10.semilogy(logger.time, logger.l2_divB, 'b-', label='L2(∇·B)')
        ax10.set_title('∇·B Constraint', fontsize=10)
        ax10.set_xlabel('t'); ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)
        
        ax11= fig.add_subplot(gs[2, 2])
        ax11.plot(logger.time, logger.canonical_helicity, 'k-', linewidth=1.5)
        ax11.set_title('Canonical Helicity K', fontsize=10)
        ax11.set_xlabel('t')
        ax11.grid(True, alpha=0.3)
        
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.plot(logger.time, logger.magnetic_helicity, 'b-', label='H_m', linewidth=1.5)
        ax12.plot(logger.time, logger.cross_helicity, 'r-', label='H_cross')
        ax12.set_title('Helicity Components', fontsize=10)
        ax12.set_xlabel('t'); ax12.legend(fontsize=8)
        ax12.grid(True, alpha=0.3)
    
    fig.suptitle(f'2D Ideal MHD - Gold-Hoyle Plectoneme under Peristaltic Compression\nStep {step}, t = {t:.5f}, HLLD + MUSCL-VanLeer + SSP-RK3 + GLM', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

