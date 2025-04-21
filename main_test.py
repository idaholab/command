# Copyright 2024, Battelle Energy Alliance, LLC All rights reserved

import sys
sys.path.append('..')
from command import base, ctrl, dataio, dt, mlo

# if True, will print out which system is running and how long different operations are taking
base.debug_flag = False

influxdb_server = 'http://localhost:8086'
dash_server = 'http://127.0.0.1:8050'

if __name__ == '__main__':

    num = [1]
    den = [1, 2, 3]

    # initializes the simulator object
    s = base.Simulator(time_step=0.1, simulation_speed=1)

    # adds the parts
    s.add([
        base.Variables('r', 'y1', 'u1', 'e1', 'y2', 'u2', 'e2'),
        base.SquareWave(inputs='sim_time', outputs='r', period=100, phase=0),

        ctrl.PIDController(inputs='e1', outputs='u1', kp=5, ki=10, kd=0, tau=0.1),
        ctrl.TransferFunction(inputs='u1', outputs='y1', tf_matrices=[num, den]),
        base.Subtraction(inputs='r', inputs2='y1', outputs='e1'),

        ctrl.PIDController(inputs='e2', outputs='u2', kp=5, ki=2, kd=0, tau=0.1),
        ctrl.TransferFunction(inputs='u2', outputs='y2', tf_matrices=[num, den]),
        base.Subtraction(inputs='r', inputs2='y2', outputs='e2'),

        dataio.Visualization(update_time=2, n_time_steps=1000, debug=True, plot_dicts=[
            {'variables': ['y1', 'y2', 'r']},
            {'variables': ['u1', 'u2']},
            {'variables': ['e1', 'e2']},
        ]),
    ])

    # compiles and starts the simulation
    s.compile()
    s.start()
