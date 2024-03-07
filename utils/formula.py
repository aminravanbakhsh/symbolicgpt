import sys
import os
import pdb

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import numpy as np
from unit import Unit

class Formula:

    DEFAULT_VARIABLE_BOUND_LIST = [1, 5, 10, 100, 1000]
    DEFAULT_CONSTANT_BOUND_LIST = [1, 5, 10, 100, 1000]

    def __init__(self, 
                    equation: str, 
                    variables = [] , 
                    constants = [], 
                    variable_bounds = None, 
                    constant_bounds = None, 
                    units = None
                    ):
        """
        inputs:
            equation: 
                type: string, 
                The equation explaining the relation between variables
            
            variables: 
                type: list
                The list of variables' name in the equation

            constants:
                type: list
                The list of constants' name in the equation

            variable_bounds:
                type: dict
                The bounds of each varaible

            constant_bounds:
                type: dict
                bounds of each constant

            units:
                type: dict
                The Dimensionality of variables. This units usually are used in physics data sets. 
        """

        self.equation = str(equation)
        self.variables = variables
        self.constants = constants

        if variable_bounds == None:
            self.variable_bounds = {}
            for var in self.variables:
                self.variable_bounds[var] = [ -Formula.DEFAULT_VARIABLE_BOUND_LIST[2], Formula.DEFAULT_VARIABLE_BOUND_LIST[2]]

        else:
            assert isinstance(variable_bounds, dict), "Variable is not a dictionary"
            assert len(variable_bounds) == len(variables), "inconsistent variable_bounds and variables"


            for key, value in variable_bounds.items():
                assert isinstance(key, str), "Key is not a string"
                # assert isinstance(value, list), "Value is not a list"

            self.variable_bounds = variable_bounds

        if constant_bounds == None:
            self.constant_bounds = {}
            for c in self.constants:
                self.constant_bounds[c] = [ -Formula.DEFAULT_CONSTANT_BOUND_LIST[2], Formula.DEFAULT_CONSTANT_BOUND_LIST[2]]
        else:
            assert isinstance(constant_bounds, dict), "Variable is not a dictionary"
            assert len(constant_bounds) == len(constants), "inconsistent constant_bounds and constants"

            for key, value in constant_bounds.items():
                assert isinstance(key, str), "Key is not a string"
                assert isinstance(value, list), "Value is not a list"

            self.constant_bounds = constant_bounds

        self.units = None
        if units == None:
            units_dict = {}
            for var in self.variables:
                units_dict[var] = Unit()

            self.units = units_dict
        else:
            assert isinstance(units, dict), "units is not a dict"
            assert len(units) == len(variables)
            assert set(units.keys()) == set(variables), "inconsistent units and variables."
            for key, value in units.items():
                assert isinstance(key, str), "Key is not a string"
                assert isinstance(value, Unit), "Value is not a list"

            self.units = units
