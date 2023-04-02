import os
from PIL import Image
import numpy as np
import skimage
import tifffile
import zarr
from .utils import (
    downsample_slice,
    open_store,
    get_min_datatype,
    get_slice_size,
    PapyrusDataException,
    Timer,
)

CHUNK_SIZE = 1024

class PapyrusImage:
    def __init__(self, zarrpath, imagepath=None, multiscale=True, xlims=None, ylims=None):
        if imagepath is not None:
            self.build_from_image(imagepath, zarrpath, multiscale=multiscale, xlims=xlims, ylims=ylims)  
        if not os.path.exists(zarrpath):
            raise PapyrusDataException(f"Provided zarr path {zarrpath} does not exist.  If generating a new zarr, the path to the image must be provided.")
        self.store, self.zarr = self.load_from_zarr(zarrpath)
        self.multiscale = self.zarr.attrs["multiscale"]
        self.shape = self.zarr.image.shape
        if self.multiscale:
            self.scale_levels = self.zarr.attrs["scale_levels"]
        else:
            self.scale_levels = 0
    
    def __getitem__(self, key):
        if len(key) == 2:
            # If we just provide the positions, we want to intelligently select which image
            # to grab from so that we don't get more data than we'll use.
            x, y = key
            if isinstance(x, int) or isinstance(y, int):
                return self.zarr.image.__getitem__(key)
            scale_size = max(
                get_slice_size(x, self.shape[0]), 
                get_slice_size(y, self.shape[1]),
            )
            scale = 0
            while (scale < self.scale_levels) and scale_size > (CHUNK_SIZE // 4):
                scale += 1
                scale_size = scale_size // 2
            print(scale)
        elif len(key) == 3:
            x, y, scale = key
        else:
            raise PapyrusDataException("Too many indices for array")
        # With three entries, the last is the downsample index and the others are the
        # slices into the sub-images.
        if not isinstance(scale, int):
            raise PapyrusDataException("The last index must be an integer to index into the downsampled image data.")
        if scale < 0:
            raise PapyrusDataException("Cannot index negative sampled images")
        if scale > self.scale_levels:
            raise PapyrusDataException(f"Only {self.scale_levels} downsampled images in this library")
        subimage = "image" if scale == 0 else f"image_downsampled{scale}"
        x = downsample_slice(x, scale)
        y = downsample_slice(y, scale)
        return self.zarr.__getattr__(subimage).__getitem__((x, y))
    
    def close(self):
        self.store.close()

    @staticmethod
    def load_from_zarr(filepath):
        with Timer("Loading from existing zarr"):
            store = open_store(filepath)
            root = zarr.group(store=store, overwrite=False)
            return store, root
    
    @staticmethod
    def build_from_image(imagepath, zarrpath, chunk_size=None, multiscale=True, xlims=None, ylims=None):
        if chunk_size is None:
            chunk_size = (CHUNK_SIZE, CHUNK_SIZE)
        if os.path.exists(zarrpath):
            raise PapyrusDataException(f"Provided zarr path {zarrpath} already exists.  Please remove before initiation so that we do not overwrite.")
        print(f"Building zarr from {imagepath}")
        # tifffile is much faster, so we use it when possible
        with Timer("Loading image data"):
            if imagepath.endswith(".tiff") or imagepath.endswith(".tif"):
                img_data = tifffile.imread(imagepath)
            else:
                img_data = np.array(Image.open(imagepath))
        with Timer("Storing full image data"):
            img_data = img_data.astype(get_min_datatype(img_data))
            if xlims is not None:
                xmin, xmax = xlims
                xslice = slice(xmin, xmax, None)
            else:
                xslice = slice(None, None, None)
            if ylims is not None:
                ymin, ymax = ylims
                yslice = slice(ymin, ymax, None)
            else:
                yslice = slice(None, None, None)
            img_data = img_data[xslice,yslice]
            store = open_store(zarrpath, "w")
            root = zarr.group(store=store, overwrite=True)
            root.array(
                name="image", 
                data=img_data, 
                chunks=chunk_size, 
                compressor="default", 
                write_empty_chunks=False, 
                fill_value=0,
            )
        # Serially add 2x downsampled copies of the image into the group until the size of the 
        # downsampled image is smaller than a quarter of a chunk.
        root.attrs["multiscale"] = multiscale
        if multiscale:
            scale_levels = int(max(
                np.ceil(np.log2(img_data.shape[0] / (chunk_size[0] // 1))),
                np.ceil(np.log2(img_data.shape[1] / (chunk_size[1] // 1))),
            ))
            scale_levels = min(10, scale_levels)
            root.attrs["scale_levels"] = scale_levels
            with Timer("Generating downsampled images"):
                for scale in range(1, scale_levels + 1):
                    anti_aliasing = True
                    if img_data.dtype in [np.bool8, np.bool_]:
                        anti_aliasing = False
                    resized_img = skimage.transform.rescale(
                        img_data, 1 / 2 ** scale, preserve_range=True, anti_aliasing=anti_aliasing,
                    ).astype(img_data.dtype)
                    root.array(
                        name=f"image_downsampled{scale}",
                        data=resized_img,
                        chunks=chunk_size,
                        compressor="default",
                        write_empty_chunks=False,
                        fill_value=0,
                    )
        store.close()