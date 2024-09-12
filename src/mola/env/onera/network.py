########## CAUTION IMPORTANT NOTE !!!!! 
## COMMENTS INSIDE THE DICTIONNARIES ARE FORBIDEN IN THIS FILE

AvailableEnvironments = ['ld', 'spiro', 'sator', 'juno']

# NOTE THE COMMENTS ARE OUTSIDE OF DICT (OUTSIDE OF BRACKETS {})
#   'n*' : 'sator',
#   computation nodes on juno
#   '^n0([0-2][0-9]|30|31|32)$' : 'juno',  # n[001-032]
PatternsToEnvironments = {
    'ld*' : 'ld',
    'visung*' : 'ld',
    'spiro*' : 'spiro',
    'sator*' : 'sator',
    'n03[3-9]' : 'sator',
    'n0[4-9][0-9]' : 'sator',
    'n[1-9]??' : 'sator',
    'n????' : 'sator',
    'f0[1-4]' : 'juno',
    'v00[1-6]' : 'juno',  
    'n00[1-9]' : 'juno',  
    'n01[0-9]' : 'juno',  
    'n02[0-9]' : 'juno',  
    'n03[0-2]' : 'juno',  
    'b00[1-2]$' : 'juno',  
    'a00[1-4]$' : 'juno',  
    'g00[1-2]$' : 'juno',  
}


PathsToEnvironments = {
    '/scratch*' : 'spiro',
    '/tmp_user/sator/*' : 'sator',
    '/tmp_user/juno/*' : 'juno',
}
