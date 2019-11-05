import network as nw
import numpy as np

import matplotlib.pyplot as plt


# Dots
np.random.seed( 10 )

###############################
# Parameters
T = 20
tau = .001 # execution timestep for the rate model

tc_primary = 0.01 # thalamo-cortical weights
tc_secondary = 0.01 # thalamo-cortical weights
cc_ei = .03 # intra-cortical weights
cc_ie = -0.9 # intra-cortical weights


###############################
# Building and instructing

# update_func, tau, rand_mean=0.0, rand_std=0.00001, isSensory=False, isReadout=False ):
L1 = nw.Node( nw._load, tau )
L2 = nw.Node( nw._load, tau )
L3 = nw.Node( nw._load, tau )

W1e = nw.Node( nw._sigmoid, tau )
W1i = nw.Node( nw._sigmoid, tau )
W2e = nw.Node( nw._sigmoid, tau )
W2i = nw.Node( nw._sigmoid, tau )
W3e = nw.Node( nw._sigmoid, tau )
W3i = nw.Node( nw._sigmoid, tau )

# Arcs
tc11 = nw.Arc( target=W1e, source=L1, weight=tc_primary )
tc12 = nw.Arc( target=W2e, source=L1, weight=tc_secondary )
tc13 = nw.Arc( target=W3e, source=L1, weight=tc_secondary )
tc11 = nw.Arc( target=W1i, source=L1, weight=tc_primary )
tc12 = nw.Arc( target=W2i, source=L1, weight=tc_secondary )
tc13 = nw.Arc( target=W3i, source=L1, weight=tc_secondary )

tc21 = nw.Arc( target=W1e, source=L2, weight=tc_secondary )
tc22 = nw.Arc( target=W2e, source=L2, weight=tc_primary )
tc23 = nw.Arc( target=W3e, source=L2, weight=tc_secondary )
tc21 = nw.Arc( target=W1i, source=L2, weight=tc_secondary )
tc22 = nw.Arc( target=W2i, source=L2, weight=tc_primary )
tc23 = nw.Arc( target=W3i, source=L2, weight=tc_secondary )

tc31 = nw.Arc( target=W1e, source=L3, weight=tc_secondary )
tc32 = nw.Arc( target=W2e, source=L3, weight=tc_secondary )
tc33 = nw.Arc( target=W3e, source=L3, weight=tc_primary )
tc31 = nw.Arc( target=W1i, source=L3, weight=tc_secondary )
tc32 = nw.Arc( target=W2i, source=L3, weight=tc_secondary )
tc33 = nw.Arc( target=W3i, source=L3, weight=tc_primary )


# cc11 = nw.Arc( target=W1, source=L1, weight=tc_primary )
# cc12 = nw.Arc( target=W2, source=L1, weight=tc_secondary )
# cc13 = nw.Arc( target=W3, source=L1, weight=tc_secondary )

# cc21 = nw.Arc( target=W1, source=L2, weight=tc_secondary )
# cc22 = nw.Arc( target=W2, source=L2, weight=tc_primary )
# cc23 = nw.Arc( target=W3, source=L2, weight=tc_secondary )

# cc31 = nw.Arc( target=W1, source=L3, weight=tc_secondary )
# cc32 = nw.Arc( target=W2, source=L3, weight=tc_secondary )
# cc33 = nw.Arc( target=W3, source=L3, weight=tc_primary )


###################
# Main loop

# storage
W1e_states = []

for t in range(T):
    print "\ntime:", t

    # INPUT
    # input for this timestep
    W1e.update( 5. ) 
    print "W1e.state:",W1e.state
    # storage
    W1e_states.append( W1e.state )

print W1e_states
