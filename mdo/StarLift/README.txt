This folder is a top-level folder for Starlift's MDO code, adapted from Hofheins et al. and their active debris removal mission. Some files are deprecated; the relevant folder and file structure is as such:

Libraries: contains spreadsheet libraries for use as code input, designed to provide human accessibility for mission parameters that can vary (e.g. dV, trajectories, payload mass, etc.)

MDOtoolbox: contains the main MDOrun.m code, as well as helper .m functions to run that code. These helper .m functions are the main functional blocks in the overall MDO diagram (e.g. propulsion, power, etc.)

Misc: contains various coding experiments, not necessarily attached to MDO. For example, tabletesting.m serves solely to test certain Matlab functions regarding reading .xlsx files.

OPT: most relevant to Starlift is OPTga.m, which is a genetic algorithm that currently optimizes for cost of a spacecraft program. This folder also contains helper functions for the genetic algorithm, contained in optimizationAlgorithms. As of current update, other files are deprecated from old project.

Last updated: 4/24/24