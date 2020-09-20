import sys
import numpy as np

byteorder = "little"

with open(sys.argv[-1], 'rb') as f:
    nRootNodes = int.from_bytes(f.read(4), byteorder)
    nNodes = int.from_bytes(f.read(4), byteorder)
    nLeaves = int.from_bytes(f.read(4), byteorder)

    print("nRootNodes:", nRootNodes)
    print("nNodes:", nNodes)
    print("nLeaves:", nLeaves)

    print("")
    print("rootIndices:")

    print(np.frombuffer(f.read(nRootNodes * 4), dtype=np.int32))

    print("")
    print("cutIndices:")

    print(np.frombuffer(f.read(nNodes * 4), dtype=np.uint32))

    print("")
    print("cutValues:")

    print(np.frombuffer(f.read(nNodes * 4), dtype=np.float32))

    print("")
    print("leftIndices:")

    print(np.frombuffer(f.read(nNodes * 4), dtype=np.int32))

    print("")
    print("rightIndices:")

    print(np.frombuffer(f.read(nNodes * 4), dtype=np.int32))

    print("")
    print("responses:")

    print(np.frombuffer(f.read(nLeaves * 4), dtype=np.float32))
