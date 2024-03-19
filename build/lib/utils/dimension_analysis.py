import sys
import os
import pdb

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from unit import Unit
from formula import Formula


class Dimensional_Analysis:

    @classmethod
    def func(cls, formula: Formula):
        
        dim_matrix = []
        for var in formula.variables:
            dim_matrix.append([
                formula.units[var].length,
                formula.units[var].mass,
                formula.units[var].time,
                formula.units[var].electric_current,
                formula.units[var].temperature,
                formula.units[var].luminous,
                formula.units[var].mole,
            ])

        for row in dim_matrix:
            print(row)
            
    #     dim_matrix = []
    #     for dim in dataset.units:
    #         dim_matrix.append([dataset.units[dim].distance,
    #                            dataset.units[dim].time,
    #                            dataset.units[dim].mass,
    #                            dataset.units[dim].electric_charge,
    #                            dataset.units[dim].temperature
    #                            ])

    # @classmethod
    # def analysis(cls, dataset):
    #     dim_matrix = []
    #     for dim in dataset.units:
    #         dim_matrix.append([dataset.units[dim].distance,
    #                            dataset.units[dim].time,
    #                            dataset.units[dim].mass,
    #                            dataset.units[dim].electric_charge,
    #                            dataset.units[dim].temperature
    #                            ])
        
    #     dim_matrix = sp.Matrix(dim_matrix).T
    #     rref_matrix, pivot_columns = dim_matrix.rref()
    #     null_space = np.array(dim_matrix.nullspace())
    #     shape = null_space.shape
    #     null_space = null_space.reshape((shape[0], shape[1]))
        
    #     return null_space

