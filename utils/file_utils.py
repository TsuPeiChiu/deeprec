import h5py as h5

def open_hdf(filename, acc='r', cache_size=None):
    if cache_size:
        propfaid = h5.h5p.create(h5.h5p.FILE_ACCESS)
        settings = list(propfaid.get_cache())
        settings[2] = cache_size
        propfaid.set_cache(*settings)
        fid = h5.h5f.open(filename.encode(), fapl=propfaid)
        _file = h5.File(fid, acc)
    else:
        _file = h5.File(filename, acc)
    return _file

def read_hdf(path, cache_size):
    f = open_hdf(path, cache_size=cache_size)
    data = dict()
    for k, v in f['data'].items():
        data[k] = v
    #for k, v in f['pos'].items():
    #    data[k] = v
    return (f, data)