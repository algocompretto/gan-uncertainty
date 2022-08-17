import os
import numpy as np


def read_conditional_samples(filename: object = 'eas.dat', nanval: object = -997799) -> object:
    debug_level = 1
    if not (os.path.isfile(filename)):
        print("Filename:'%s', does not exist" % filename)

    file = open(filename, "r")
    if debug_level > 0:
        print("eas: file ->%20s" % filename)

    eas = {'title': (file.readline()).strip('\n')}

    if debug_level > 0:
        print("eas: title->%20s" % eas['title'])

    dim_arr = eas['title'].split()
    if len(dim_arr) == 3:
        eas['dim'] = {}
        eas['dim']['nx'] = int(dim_arr[0])
        eas['dim']['ny'] = int(dim_arr[1])
        eas['dim']['nz'] = int(dim_arr[2])

    eas['n_cols'] = int(file.readline())

    eas['header'] = []
    for i in range(0, eas['n_cols']):
        # print (i)
        h_val = (file.readline()).strip('\n')
        eas['header'].append(h_val)

        if debug_level > 1:
            print("eas: header(%2d)-> %s" % (i, eas['header'][i]))

    file.close()

    try:
        eas['D'] = np.genfromtxt(filename, skip_header=2 + eas['n_cols'])
        if debug_level > 1:
            print("eas: Read data from %s" % filename)
    except:
        print("eas: COULD NOT READ DATA FROM %s" % filename)

    # add NaN values
    try:
        eas['D'][eas['D'] == nanval] = np.nan
    except:
        print("eas: FAILED TO HANDLE NaN VALUES (%d(" % nanval)

    # If dimensions are given in title, then convert to 2D/3D array
    if "dim" in eas:
        if eas['dim']['nz'] == 1:
            eas['Dmat'] = eas['D'].reshape((eas['dim']['ny'], eas['dim']['nx']))
        elif eas['dim']['nx'] == 1:
            eas['Dmat'] = np.transpose(eas['D'].reshape((eas['dim']['nz'], eas['dim']['ny'])))
        elif eas['dim']['ny'] == 1:
            eas['Dmat'] = eas['D'].reshape((eas['dim']['nz'], eas['dim']['nx']))
        else:
            eas['Dmat'] = eas['D'].reshape((eas['dim']['nz'], eas['dim']['ny'], eas['dim']['nx']))

        eas['Dmat'] = eas['D'].reshape((eas['dim']['nx'], eas['dim']['ny'], eas['dim']['nz']))
        eas['Dmat'] = eas['D'].reshape((eas['dim']['nz'], eas['dim']['ny'], eas['dim']['nx']))

        eas['Dmat'] = np.transpose(eas['Dmat'], (2, 1, 0))

        if debug_level > 0:
            print("eas: converted data in matrixes (Dmat)")

    eas['filename'] = filename

    return eas

