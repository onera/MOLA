########## CAUTION IMPORTANT NOTE !!!!! 
## COMMENTS INSIDE THE DICTIONNARIES ARE FORBIDEN IN THIS FILE

PathsToEnvironments = {
    '/scratch*' : 'spiro',
    '/tmp_user/sator/*' : 'sator',
    '/tmp_user/juno/*' : 'juno',
}

def guess_localhost():
    import os
    cluster = os.getenv('ONERA_CLUSTERNAME', 'ld')
    if cluster == 'visung':
        cluster = 'ld'  # same environment on ld and visung
    return cluster

if __name__ == '__main__':
    cluster = guess_localhost()
    print(cluster)
