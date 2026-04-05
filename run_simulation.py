from mhd_solver import *

if __name__ == '__main__':
    # Default production settings
    cfg = SolverConfig(nx=128, nz=256, t_end=2.0)
    state = MHDState(cfg)
    gold_hoyle_initialiser(state, cfg, twist=3.0)
    print('🚀 Solver initialized and ready for execution.')
