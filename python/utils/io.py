import struct

import numpy as np


def read_ivecs(filename):
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename):
    return read_ivecs(filename).view("float32")


def write_ivecs(filename, m):
    print(f"Writing File - {filename}")
    n, d = m.shape
    myimt = "i" * d
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", d))
            bin = struct.pack(myimt, *m[i])
            f.write(bin)
    print(f"\t{filename} wrote")


def write_fvecs(filename, m):
    m = m.astype("float32")
    write_ivecs(filename, m.view("int32"))


def read_ibin(filename):
    n, d = np.fromfile(filename, count=2, dtype="int32")
    a = np.fromfile(filename, dtype="int32")
    print(f"\t{filename} readed")
    return a[2:].reshape(n, d)


def read_fbin(filename):
    return read_ibin(filename).view("float32")


def write_sampled_data(filename, m, num_sampled, seed=42):
    num_points = m.shape[0]
    d = m.shape[1]
    m = m.view("int32")

    np.random.seed(seed)
    sequence = np.random.permutation(num_points)

    with open(filename, "wb") as f:
        for i in range(num_sampled):
            f.write(struct.pack("i", d))
            bin = struct.pack(f"{d}i", *m[sequence[i]])
            f.write(bin)
