import treelite


def export_treelite_model(filename):

    # From https://treelite.readthedocs.io/en/latest/

    model = treelite.Model.load("model.bin", model_format="xgboost")

    # Produce a zipped source directory, containing all model information
    # Run `make` on the target machine
    model.export_srcpkg(
        platform="unix", toolchain="gcc", pkgpath=filename + ".zip", libname=filename + ".so", verbose=True
    )

    # Like export_srcpkg, but generates a shared library immediately
    # Use this only when the host and target machines are compatible
    params = {
        "parallel_comp": 16,  # set to number of threads you want
    }
    model.export_lib(toolchain="gcc", libpath=filename + ".so", params=params)


export_treelite_model("model")

import treelite_runtime
import numpy as np
import time

# We use treelite with one thread for a fair comparison with fastforest.
predictor = treelite_runtime.Predictor("./model.so", nthread=1, verbose=True)

X = np.random.uniform(-5, 5, size=(100000, 5))

start_time = time.time()

dmat = treelite_runtime.DMatrix(X)
out_pred = predictor.predict(dmat)

print(np.mean(out_pred))

elapsed_secs = time.time() - start_time

print("Wall time for inference: {0:.2f} s".format(elapsed_secs))
