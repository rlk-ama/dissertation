from netCDF4 import Dataset
import argparse

def toCDF4(source, dest, length):
    dataset = Dataset(dest, "w", format="NETCDF3_64BIT")
    nr = dataset.createDimension("nr", length)
    #np = dataset.createDimension("np", 1)
    y = dataset.createVariable("y", "f8", ("nr",))# "np"))
    #r = dataset.createVariable("r", "f8", ("np",))
    #phi = dataset.createVariable("phi", "f8", ("np",))
    #sigma = dataset.createVariable("sigma", "f8", ("np",))
    time = dataset.createVariable("time", "f8", ("nr",))#"np",))
    time[:] = [i for i in range(1,length+1)]
    #r[:] = 44.7
    #phi[:] = 10
    #sigma[:] = 0.5
    line = list(map(float, source.readline().split()))
    y[:] = line
    dataset.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Options for the NetCDF4 converter")
    parser.add_argument("--source", type=argparse.FileType('r'), help="Observations, in a file, space separated")
    parser.add_argument("--dest", type=str, help="Destination file")
    parser.add_argument("--length", type=int, help="Number of observations")

    args = parser.parse_args()
    toCDF4(**args.__dict__)
