from netCDF4 import Dataset
import argparse

def fromCDF4(source, dest):
    dataset = Dataset(source, "r", format="NETCDF3_64BIT")
    rs = dataset.variables['r']
    phis = dataset.variables['phi']
    sigmas = dataset.variables['sigma']
    output = zip(rs, phis, sigmas)
    for elem in output:
       dest.write(" ".join(map(str, elem)))
       dest.write("\n")
    dataset.close()

def fromCDF4_generalized(source, dest):
    dataset = Dataset(source, "r", format="NETCDF3_64BIT")
    rs = dataset.variables['r']
    phis = dataset.variables['phi']
    sigmas = dataset.variables['sigma']
    thetas = dataset.variables['theta']
    output = zip(rs, phis, thetas, sigmas)
    for elem in output:
       dest.write(" ".join(map(str, elem)))
       dest.write("\n")
    dataset.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Options for the NetCDF4 converter")
    parser.add_argument("--dest", type=argparse.FileType('w'), help="Observations, in a file, space separated")
    parser.add_argument("--source", type=str, help="Destination file")
    parser.add_argument("--generalized", type=bool, help="Are you using the Generalized Ricker Map ?")

    args = parser.parse_args()
    arguments = {k:v for k,v in args.__dict__.items() if k != "generalized"}

    if args.generalized:
        fromCDF4_generalized(**arguments)
    else:
        fromCDF4(**arguments)

