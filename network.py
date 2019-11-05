import numpy as np
import types
import matplotlib.pyplot as plt


"""
NETWORK
 
Each Node is a population of continuous variables with a generic update function.

A Strategy design is adopted to flexibly assign different update functions to different nodes.

See:
http://stackoverflow.com/questions/963965/how-is-this-strategy-pattern-written-in-python-the-sample-in-wikipedia
http://codereview.stackexchange.com/questions/20718/strategy-design-pattern-with-various-duck-type-classes
"""

# update functions
def _linear( self, inputs ):
    self.state = (1 - self.tau)*self.state + self.tau*inputs.clip(0) + np.random.normal( self.rand_mean, self.rand_std, 1 )

def _sigmoid( self, inputs ):
    self.state = (1 - self.tau)*self.state + self.tau*np.tanh( inputs ) + np.random.normal( self.rand_mean, self.rand_std, 1 )

def _cubic( self, inputs, a=.038, b= -.36, c=1., d=-.02 ):
    # d + cr + br^2 + ar^3
    self.state = (1 - self.tau)*( d + c*self.state + b*self.state**2 + a*self.state**3 ) + self.tau*inputs.clip(0) + np.random.normal( self.rand_mean, self.rand_std, 1 )

def _load( self, inputs ):
    self.state = inputs + np.random.normal( self.rand_mean, self.rand_std, 1 )




###############################
# Node
class Node( object ):
    """
    Node

    Parameters:
        update_func : function reference
            function used to update the node state 
        tau : float
            time constant for the dynamic: 0<tau<1 (the smaller tau the more the system remembers)
        rand_std : float (default=0.00001)
            amount of noise injected during state update
        isSensory : bool (default=False)
            whether the node is relaying external input (the update function is simply assigning the input that has to have the same shape of the node)
        isReadout : bool (default=False)
            whether the node is used to copy the results outside for analysis
    """

    def __init__( self, update_func, tau=0.1, rand_mean=0.0, rand_std=0.00001, isSensory=False, isReadout=False ):
        self.isSensory = isSensory
        self.isReadout = isReadout
        self.tau = tau  # timestep execution
        self.rand_mean = rand_mean  # mean of injected noise
        self.rand_std = rand_std  # stdev of injected noise
        self.update = types.MethodType( update_func, self, Node ) # strategy design
        self.state = np.random.normal( rand_mean, rand_std ) # init units' states
        self.inputs = 0.0

    def update( self ):
        pass # defined at init time





class Arc( object ):
    """
    Directed Arc

    The arc represents the connections between two nodes.
    Connections are represented by target node, to ease reading during execution.

    The resulting connection array will be organized in such a way that:
    - the read() will take source state
    - and return the input to the target

    Parameters:
        target : Node
            target node
        source : Node
            source node
        weight : float
            base weight
    """

    def __init__( self, target, source, weight ):
        self.source = source
        self.target = target
        self.weight = weight
        self.connections = np.array([])
        self.output = None
        # self.read = types.MethodType( read_func, self, Arc ) # strategy design

    def read( self, transpose=False ):
        """
        Reads the source state and multiply (dot) it by the connections defined to that source.
        """
        conns = self.connections 
        state = self.source.state
        if transpose:
            self.output = np.dot( conns.T, state )
        else:
            self.output = np.dot( conns, state )
        return self.output




###############################

class Network( object ):
    """
    Network

    It contains nodes and arcs to form a connected network.

    Parameters:
        structure : dictionary
            All nodes and arcs with their parameters
        parameters : dictionary
            External parameters to instruct and simulate the network
    """

    def __init__( self, structure=None, parameters={} ):
        self.structure = structure
        self.parameters = parameters
        self.nodes = {}
        self.afferents = {}

        if structure:
            # node construction
            for name,params in structure['nodes'].items():
                self.nodes[name] = Node( *params )
                # init
                self.afferents[name] = []

            # arc construction
            for name,params in structure['arcs'].items():
                target = self.nodes[ params[0] ]
                source = self.nodes[ params[1] ]
                #          target  source  weight
                arc = Arc( target, source, params[2] )
                # stacked by target node 
                self.afferents[ params[0] ].append( arc )


    def simulate( self, external_inputs ):
        """
        For each timestep, all inputs are computed (according to afference to each node), then all nodes are updated.
        """
        # time iteration
        for t in range(self.parameters['Blank']+self.parameters['Time']+self.parameters['Blank']):
            # different areas at different times
            for area in self.structure['sequence']:
                # Inputs
                for name in area:
                    # list of source nodes for the current target one
                    # Arc read
                    if self.nodes[name].isSensory:
                        self.nodes[name].inputs = external_inputs[int(name[-1])-1][t] # current input id is last in the name of the node (-1)
                    else:
                        self.nodes[name].inputs = np.zeros( self.nodes[name].shape )
                        for arc in self.afferents[ name ]: 
                            self.nodes[name].inputs += arc.read()
                # Dynamic
                for name in area:
                    # if 'PFC' in name:
                    #     print self.nodes[name].inputs
                    self.nodes[name].update( self.nodes[name].inputs )
                    # Storage
                    self.nodes[name].states.append( self.nodes[name].state )

                    if False: # Save images for visual inspection
                        im = plt.imshow( self.nodes[name].state, cmap='gray', interpolation='nearest' )
                        plt.savefig( self.parameters['rootdir']+'/'+name+"_"+str(t)+".png" )
                        plt.clf()
                        plt.close()


    # to be used in the trials loop
    def store_states( self, trials ):
        for name,node in self.nodes.items():
            # print "    states for", name, "=",len(node.states)
            trials[name].append( node.states )
        return trials


    # this func expects a np.array with all trials for a population (node)
    # to be used outside trials loop, 'trials' is expected as parameter
    def firing_rates( self, trials, selected, sample_name, sample_time ):
        results = {}
        for name,node in self.nodes.items():
            # print "    ------"
            # print "    trials per node",name,"=",len(trials[name])
            nptrials = np.array( trials[name] )
            # Population firing rates
            # axis = 0:trials, 1:time, 2:pop_x, 3:pop_y
            npfr = nptrials.sum(axis=(2,3)) / (node.shape[0] * node.shape[1]) 
            # Trial averaged population firing rates
            trial_avg_npfr = ( nptrials.sum(axis=(0,2,3)) / self.parameters['Trials'] ) / (node.shape[0] * node.shape[1]) 
            # save
            results[name] = npfr
            results['trial_avg_'+name] = trial_avg_npfr
            # samples for scatterplot
            if sample_name in name:
                results[name+'_sample'] = [ fr[sample_time] for fr in npfr ]
        return results


#############


def plot2D( filename, data, title, xlabel, ylabel, xlim=None, ylim=None, withLegend=False ):
    plt.close('all')
    plt.figure()

    for label,a,color in data:
        plt.plot( a, color, label=label )

    if xlim:
        plt.xlim( xlim )

    if ylim:
        plt.ylim( ylim )

    if( withLegend ):
        plt.legend()

    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    plt.title( title )
    plt.savefig( filename )
    plt.clf()
    plt.close()
