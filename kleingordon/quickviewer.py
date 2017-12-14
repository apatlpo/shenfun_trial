import h5py
import matplotlib.pyplot as plt
import sys, time


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print('    %s: %s'.format(key, val))


if len(sys.argv) > 1:

    # open file and print variables
    fname = sys.argv[1]
    f = h5py.File(fname, 'r')

    if len(sys.argv) == 2:

        f.visititems(print_attrs)

    else:

        # go within data tree
        g=f
        for i in range(2,len(sys.argv)):
            g=g[sys.argv[i]]
            print(g)
            if i==len(sys.argv)-1:
                flag=True
                for key, val in g.items():
                    if isinstance(val, h5py.Dataset):
                        flag=False
                        if len(val.shape)==2:
                            gname=g.name[1:].replace('/','_')+'_'
                            plt.figure()
                            plt.imshow(val)
                            #print(val[:,:])
                            plt.title(gname+' '+key)
                            plt.savefig(gname+key+'.png')
                            plt.close()
                        elif len(val.shape)==3:
                            print('Data is 3D')
                        elif len(val.shape) == 1:
                            print('Data is 1D')
                if flag:
                    g.visititems(print_attrs)

f.close()
