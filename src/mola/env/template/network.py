PathsToEnvironments = {
    # in the frame of the template network, all paths beginning 
    # by /scratch/ are accessible only from the 'cluster' environment
    '/scratch/*' : 'cluster',  
}

def guess_localhost():
    # use hostname or an environment variable to detect your local machine name
    host = 'localhost' # be default for now
    return cluster

if __name__ == '__main__':
    cluster = guess_localhost()
    print(cluster)