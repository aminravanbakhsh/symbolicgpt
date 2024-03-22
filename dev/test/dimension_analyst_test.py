import sys
import os
import pdb

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dimension_analyst import Dimensional_Analyst
from formula import Formula
from unit import Unit

def test_001():
    print("\n-----------------------------------------------")
    print("test_001:")
    print("-----------------------------------------------")

    data = {
        "equation": "c_0 + c_1 * x + c_2 * x**2",
        "variables_dict": {
            "x": [
                -10,
                10
            ]
        },
        "constants_dict": {
            "c_0": [
                -10,
                10
            ],
            "c_1": [
                -10,
                10
            ],
            "c_2": [
                -10,
                10
            ]
        },

        'units': {
            "x": Unit(params = {
                "mass": 1,
                "acceleration": 1,
            })
        }
    }

    f = Formula(equation        = data['equation'].replace('^', '**'), 
                variables       = list(data["variables_dict"].keys()),
                constants       = list(data["constants_dict"].keys()),
                variable_bounds =  data["variables_dict"],
                constant_bounds = data["constants_dict"],
                units           = data['units'])
                
    Dimensional_Analyst.func(f)




def test_002():
    print("\n-----------------------------------------------")
    print("test_002:")
    print("-----------------------------------------------")


    data = {
            # Kepler's third law
        
            "target": "T",
            "equation": "2*pi * sqrt(d^3/(G*(m1+m2)))",
            "variables": {
                "m1": [0, 10],
                "m2": [0, 10],
                " d": [0.0001,10],
                " G": 1,
                " T": None
            },
            "units": {
                "m1": Unit(mass=1),
                "m2": Unit(mass=1),
                " d": Unit(length=1),
                " G": Unit(mass=-1,length=3,time=-2),
                " T": Unit(time=1),
                }        
        }
    

    f = Formula(equation        = data['equation'].replace('^', '**'), 
                variables       = list(data["variables"].keys()),
                constants       = [],
                variable_bounds =  data["variables"],
                constant_bounds = {},
                units           = data['units'])
                
    Dimensional_Analyst.func(f)
