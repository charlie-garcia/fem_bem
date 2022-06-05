from matplotlib import pyplot as plt

def clean_env():
    try:
        from IPython import get_ipython
        get_ipython().magic('clear')
        get_ipython().magic('reset -f')
        plt.close('all')
        
    except:
        pass

def GetMemory():
    import sys, os, psutil
    process = psutil.Process(os.getpid())
    print('Memory Usage : %.2f MB\n' %(process.memory_info()[0]/(1024**2)))

def VarSize(variable):
    import sys
    print('Variable size : %.2f MB\n' %(sys.getsizeof(variable)/(1024**2)))

def mat_to_bytes(dim, nfreq, dtype=64, out="GB"):
    """Calculate the size of a numpy array in bytes.
    :param nrows: the number of rows of the matrix.
    :param ncols: the number of columns of the matrix.
    :param dtype: the size of each element in the matrix. Defaults to 32bits.
    :param out: the output unit. Defaults to gigabytes (GB)
    :returns: the size of the matrix in the given unit
    :rtype: a float
    """
    sizes = {v: i for i, v in enumerate("BYTES KB MB GB TB".split())}
    return dim**2 * nfreq * dtype / 8 / 1024. ** sizes[out]
