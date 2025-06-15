
import ngff_zarr as nz
import numpy as np
from numcodecs import Blosc

img = np.random.randint(0, 256, (128, 128, 128), dtype=np.uint8)

ngff_img = nz.to_ngff_image(
    data=img,
    dims=["z", "y", "x"],
    scale={"z": 1.0, "y": 1.0, "x": 1.0},
    name="test_image",
)
ms = nz.to_multiscales(
    data=ngff_img, scale_factors=[2,4], method=nz.Methods.DASK_IMAGE_GAUSSIAN
)


compressor = Blosc(cname="zstd", clevel=5, shuffle=-1)
nz.to_ngff_zarr(
    store="test-compression-04.zarr",
    multiscales=ms,
    version="0.4",
    compressors=compressor,
)
