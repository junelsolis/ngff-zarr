import zarr.codecs
import ngff_zarr as nz
import numpy as np
import zarr

img = np.random.randint(0, 256, (128, 128, 128), dtype=np.uint8)

ngff_img = nz.to_ngff_image(
    data=img,
    dims=["z", "y", "x"],
    name="test_image",
)
ms = nz.to_multiscales(
    data=ngff_img, scale_factors=[2, 4], method=nz.Methods.DASK_IMAGE_GAUSSIAN
)

compressors = zarr.codecs.BloscCodec(
    cname="zstd",
    clevel=5,
    shuffle="shuffle",
)
nz.to_ngff_zarr(
    store="test-compression-05.zarr",
    multiscales=ms,
    version="0.5",
    compressor=compressors,
)
