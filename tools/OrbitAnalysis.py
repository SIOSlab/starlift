from Solution import Solution

# Execute
path_str = "orbitFiles/DRO_11.241_days.p"
DRO = Solution(path_str)
# DRO.nondimensionalize()
DRO.plot_orbit()
