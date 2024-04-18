AvailableEnvironments = ['local', 'cluster']

PatternsToEnvironments = {
    'local*' : 'local',
    'cluster*' : 'cluster',
    'node*' : 'cluster',
}

PathsToEnvironments = {
    # in the frame of the template network, all paths beginning 
    # by /scratch/ are accessible only from the 'cluster' environment
    '/scratch/*' : 'cluster',  
}