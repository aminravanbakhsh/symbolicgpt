import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unit import Unit

def test_001():
    print("\n-----------------------------------------------")
    print("test_001:")
    print("-----------------------------------------------")
    
    u = Unit(params={
                'length'   :  3,
                'velocity' :  2,
            })

    """
    VALID_UNITS = {
                    'length' :              [  1,  0,  0,  0,  0,  0,  0],
                    'mass' :                [  0,  1,  0,  0,  0,  0,  0],
                    'time':                 [  0,  0,  1,  0,  0,  0,  0],
                    'electric_current':     [  0,  0,  0,  1,  0,  0,  0],
                    'temperature':          [  0,  0,  0,  0,  1,  0,  0],
                    'luminous':             [  0,  0,  0,  0,  0,  1,  0],
                    'mole':                 [  0,  0,  0,  0,  0,  0,  1],
                    
                    # customizsed units:
                    'velocity':             [  1,  0, -1,  0,  0,  0,  0],
                    'acceleration':         [  1,  0, -2,  0,  0,  0,  0],
                    'force':                [  1,  1, -2,  0,  0,  0,  0],
                    'electric_charge':      [  0,  0,  1,  1,  0,  0,  0],
                    'frequency':            [  0,  0, -1,  0,  0,  0,  0],
                   }
    """