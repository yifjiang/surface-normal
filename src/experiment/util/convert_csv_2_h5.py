import h5py
import argparse
import numpy as np


def store_hdf5(input_csv):
    output_hdf = input_csv.replace(".csv",".h5")
    with h5py.File(output_hdf, 'w') as hf:
        # first pass: find the max n_point and the number of images
        max_num = 0
        n_images = 0
        with open(input_csv, 'r') as f:
            f.readline()
            while True:
                dummy_info = f.readline()
                if not dummy_info:
                    break
                infos = dummy_info.split(',')
                _, n_point = infos[0], int(infos[2])

                max_num = max(max_num, n_point)
                n_images += 1

                for i in range(n_point):
                    f.readline()


        print('Max n_point: %d' % max_num)


        # second pass: save to hdf5

        hf.create_dataset('data', [5 * n_images, max_num])
        idx = 0
        with open(input_csv, 'r') as f:
            f.readline()
            while True:
                dummy_info = f.readline()
                if not dummy_info:
                    break
                infos = dummy_info.split(',')
                png, n_point = infos[0], int(infos[2])
                print(png)
                data = np.zeros((5, n_point))
                for i in range(n_point):
                    coords = f.readline()
                    a1, a2, a3, a4, rel = coords[:-1].split(',')
                    data[0, i] = a1
                    data[1, i] = a2
                    data[2, i] = a3
                    data[3, i] = a4
                    data[4, i] = {'=' : 0, '<' : -1, '>' : 1}[rel]

                hf['data'][idx * 5 : (idx + 1) * 5, 0:n_point] = data
                idx += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input csv', required=True)
    # parser.add_argument('-o', '--output', help='Output hdf5', default='output.h5')
    args = parser.parse_args()

    store_hdf5(args.input)
