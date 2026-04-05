from mhd_solver import *

def main():
    print("Initializing 2D MHD Gold-Hoyle Plectoneme Simulation...")
    cfg = SolverConfig(nx=64, nz=128, t_end=0.25)
    state = MHDState(cfg)
    
    # Initialize
    gold_hoyle_initialiser(state, cfg, twist=16.67)
    
    # Run loop logic here...
    print("Simulation framework ready.")

if __name__ == "__main__":
    main()
