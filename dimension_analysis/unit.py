
class Units:

    """
        All of the units in the physics world:

            1) Length - meter (m)
            2) Mass - kilogram (kg)
            3) Time - second (s)
            4) Electric current - ampere (A)
            5) Temperature - kelvin (K)
            6) Luminous intensity - candela (cd)
            7) Amount of substance - mole (mole)

            
        Each variable or constant consists of some units. For example, velocity could be written length over time,
        and therefore its dimension is meter/sec is SI system. 

        The International System of Units, internationally known by the abbreviation SI 
        (from French Système international d'unités), is the modern form of the metric system and the world's most widely used system of measurement.
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


    def __init__(self, length=0, mass=0, time=0, electric_current=0, temperature=0, luminous=0, mole=0, params=None):

        assert type(params) == dict, ValueError("params is not a dictionary.")

        self.length = length
        self.mass = mass
        self.time = time
        self.electric_current = electric_current
        self.temperature = temperature
        self.luminous = luminous
        self.mole = mole

        for key in params:
            if key in Units.VALID_UNITS:
                self.length += Units.VALID_UNITS[key][0]
                self.mass += Units.VALID_UNITS[key][1]
                self.time += Units.VALID_UNITS[key][2]
                self.electric_current += Units.VALID_UNITS[key][3]
                self.temperature += Units.VALID_UNITS[key][4]
                self.luminous += Units.VALID_UNITS[key][5]
                self.mole += Units.VALID_UNITS[key][6]
            
            else:
                raise ValueError(f"{key} is not a valid unit")


    def __str__(self):
        return "length: {:3} | mass: {:3} | time: {:3} | electric_current: {:3} | temperature: {:3} | luminous: {:3} | mole: {:3}".format(self.length, self.mass, self.time, self.electric_current, self.temperature, self.luminous, self.mole) 

    def serialize(self):
        return {
            'length': self.length,
            'mass': self.mass,
            'time': self.time,
            'electric_charge': self.electric_current,
            'temperature': self.temperature,
            'luminous': self.luminous,
            'mole': self.mole,
        }