def square(x):
    """My first function!!!!......"""
    z = x*x 
    return z
    
def mysum(*args):
    return sum(args)

def freq(s):
    return { e:s.count(e)   for e in s}

def get_max_line_number(file_name):
    from operator import itemgetter
    with open(file_name,"rt") as f:
        lines = f.readlines()
    dt = [ (index+1,len(line)) for index,line in enumerate(lines)]   
    sd = sorted(dt, key=itemgetter(1))
    return sd[-1][0]

    
    
