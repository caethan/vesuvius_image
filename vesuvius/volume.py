import os
import numpy as np
import skimage
import tifffile
import zarr
from utils import (
    downsample_slice,
    open_store,
    get_slice_size,
    PapyrusDataException,
    Timer,
)

CHUNK_SIZE = 1024

class PapyrusVolume:
    def __init__(self, zarrpath, tiffdirpath=None, multiscale=True, xlims=None, ylims=None, zlims=None, zstride=None):
        if tiffdirpath is not None:
            self.build_from_tiffdir(tiffdirpath, zarrpath, multiscale=multiscale, xlims=xlims, ylims=ylims, zlims=zlims)  
        if not os.path.exists(zarrpath):
            raise PapyrusDataException(f"Provided zarr path {zarrpath} does not exist.  If generating a new zarr, the path to the image must be provided.")
        self.store, self.zarr = self.load_from_zarr(zarrpath)
        self.multiscale = self.zarr.attrs["multiscale"]
        self.shape = self.zarr.volume.shape
        if self.multiscale:
            self.scale_levels = self.zarr.attrs["scale_levels"]
        else:
            self.scale_levels = 0
    
    def __getitem__(self, key):
        if len(key) == 3:
            # If we just provide the positions, we want to intelligently select which image
            # to grab from so that we don't get more data than we'll use.
            x, y, z = key
            if isinstance(x, int) or isinstance(y, int) or isinstance(z, int):
                return self.zarr.volume.__getitem__(key)
            scale_size = max(
                get_slice_size(x, self.shape[0]), 
                get_slice_size(y, self.shape[1]),
                get_slice_size(z, self.shape[2]),
            )
            scale = 0
            while (scale < self.scale_levels) and scale_size > (CHUNK_SIZE // 4):
                scale += 1
                scale_size = scale_size // 2
        elif len(key) == 4:
            x, y, z, scale = key
        else:
            raise PapyrusDataException("Too many indices for array")
        # With four entries, the last is the downsample index and the others are the
        # slices into the sub-images.
        if not isinstance(scale, int):
            raise PapyrusDataException("The last index must be an integer to index into the downsampled image data.")
        if scale < 0:
            raise PapyrusDataException("Cannot index negative sampled images")
        if scale > self.scale_levels:
            raise PapyrusDataException(f"Only {self.scale_levels} downsampled images in this library")
        subimage = "volume" if scale == 0 else f"volume_downsampled{scale}"
        x = downsample_slice(x, scale)
        y = downsample_slice(y, scale)
        z = downsample_slice(z, scale)
        return self.zarr.__getattr__(subimage).__getitem__((x, y, z))
    
    def close(self):
        self.store.close()
    
    @staticmethod
    def load_from_zarr(filepath):
        with Timer("Loading from existing zarr"):
            store = open_store(filepath)
            root = zarr.group(store=store, overwrite=False)
            return store, root
    
    @staticmethod
    def build_from_tiffdir(
        tiffdirpath,
        zarrpath,
        chunk_size=None,
        multiscale=True,
        xlims=None,
        ylims=None,
        zlims=None,
        zstride=None,
    ):
        if chunk_size is None:
            chunk_size = (CHUNK_SIZE, CHUNK_SIZE, 1)
        if os.path.exists(zarrpath):
            raise PapyrusDataException(f"Provided zarr path {zarrpath} already exists.  Please remove before initiation so that we do not overwrite.")
        print(f"Building zarr from {tiffdirpath}")
        init = True
        imgfiles = {
            int(imgfile.split(".")[0]): os.path.join(tiffdirpath, imgfile)
            for imgfile in os.listdir(tiffdirpath)
            if imgfile.endswith(".tif")
            and imgfile.split(".")[0].isdigit()
        }
        # Check that we have all the images in the zstack requested
        if zlims is None:
            zrange = np.arange(min(imgfiles.keys()), max(imgfiles.keys()) + 1, step=zstride)
        else:
            zstart, zstop = zlims
            zstart = 0 if zstart is None else zstart
            zstop = max(imgfiles.keys()) + 1 if zstop is None else zstop
            zrange = range(zstart, zstop, step=zstride)
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
        for z in zrange:
            if z not in imgfiles:
                raise PapyrusDataException(f"Missing image file {z} from {tiffdirpath}")
        with Timer("Loading tiff files"):
            store = open_store(zarrpath, mode="w")
            root = zarr.group(store=store, overwrite=True)
            for z_index, img_index in enumerate(zrange):
                imgfile = imgfiles[img_index]
                print(f"Loading file {imgfile}", end="\r")
                img_data = tifffile.imread(imgfile)[xslice, yslice]
                if init:
                    volume = root.zeros(
                        name="volume",
                        shape=(img_data.shape[0], img_data.shape[1], len(zrange)),
                        chunks=chunk_size,
                        dtype=img_data.dtype,
                        write_empty_chunks=False,
                    )
                    init = False
                volume[:,:,z_index] = img_data
            # We need this to clear out the \r from the last print statement
            print()
            
        # TODO: let's add the maximum image-projection along the z-axis here as well
            
        # For 3D multisampling, we'll do a nice 2D downsample in xy but just stride along the
        # z-axis.  At some point we should revisit this to get better behavior, but this
        # should be sufficient for now.
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
                    ds_zrange = np.arange(len(zrange))[::(2 ** scale)]
                    init = True
                    for new_zindex, old_zindex in enumerate(ds_zrange):
                        img_data = volume[:,:,old_zindex]
                        resized_img = skimage.transform.rescale(
                            img_data, 1 / 2 ** scale, preserve_range=True, anti_aliasing=True
                        ).astype(img_data.dtype)
                        if init:
                            volume_downsampled = root.zeros(
                                name=f"volume_downsampled{scale}",
                                shape=(resized_img.shape[0], resized_img.shape[1], len(ds_zrange)),
                                chunks=chunk_size,
                                compressor="default",
                                write_empty_chunks=False,
                            )
                            init = False
                        volume_downsampled[:,:,new_zindex] = resized_img

        store.close()