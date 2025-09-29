"""
Utilities for handling ECOGAL "pbcor.fits" files
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u
from astropy.utils.data import download_file

from grizli import utils

RGB_URL = "https://grizli-cutout.herokuapp.com/thumb?all_filters=False&size={cutout_size}&scl=1.0&asinh=True&filters=f115w-clear,f277w-clear,f444w-clear&rgb_scl=1.5,0.74,1.3&pl=2&ra={ra}&dec={dec}"

FILE_URL = "https://s3.amazonaws.com/alma-ecogal/dr1/pbcor/"


def query_footprints(ra, dec):
    """
    Get the API query for footprints that touch defined coordinates
    """
    query_url = f"https://grizli-cutout.herokuapp.com/ecogal?ra={ra}&dec={dec}&output=csv"
    alma = utils.read_catalog(query_url, format="csv")
    return alma


def get_ecogal_file(file_alma, cache=True, verbose=False, **kwargs):
    """
    Generate a path to an ECOGAL file, either from a local path or by downloading a remote file
    """
    from astropy.utils.data import download_file

    if os.path.exists(file_alma):
        if verbose:
            msg = f"get_ecogal_file: local file {file_alma}"
            print(msg)

        return file_alma

    remote_url = os.path.join(FILE_URL, file_alma).replace("+", "%2B")

    cache_file = download_file(remote_url, cache=cache)
    if verbose:
        msg = f"get_ecogal_file: remote {remote_url},  cache {cache_file}"
        print(msg)

    return cache_file


def ecogal_cutout(file_alma, ra, dec, cutout_size, *args, **kwargs):
    """
    Memory-efficient cutout
    """

    cache_file = get_ecogal_file(file_alma, cache=True)

    with pyfits.open(cache_file) as im:
        h = im[0].header
        wcs = pywcs.WCS(h)

        h["FILEALMA"] = file_alma

        xyz = np.squeeze(wcs.all_world2pix([ra], [dec], [h["CRVAL3"]], 0))
        xyzi = np.round(xyz).astype(int)

        pixel_scale = np.abs(utils.get_wcs_pscale(wcs))

        N = int(np.round(cutout_size / pixel_scale))

        # print(ra, dec, xyzi, self.shape)

        slx = slice(xyzi[0] - N, xyzi[0] + N + 1)
        sly = slice(xyzi[1] - N, xyzi[1] + N + 1)
        slh = wcs.slice((slice(0, 1), sly, slx)).to_header(relax=True)
        cut = im[0].data[:, sly, slx]

        for k in h:
            if k not in slh:
                try:
                    slh[k] = h[k]
                except ValueError:
                    continue

    hdu = pyfits.HDUList([pyfits.PrimaryHDU(data=cut, header=slh)])

    return hdu


def show_all_cutouts(
    ra,
    dec,
    sx=3,
    nx=5,
    cutout_size=None,
    thumb_url=RGB_URL,
    pre_cutout=True,
    **kwargs,
):
    """
    Make a plot showing all cutouts
    """
    import PIL
    import urllib

    query_url = f"https://grizli-cutout.herokuapp.com/ecogal?ra={ra}&dec={dec}&output=csv"
    alma = utils.read_catalog(query_url, format="csv")

    print(f"N={len(alma)}")

    alma["bmaj"] *= 3600
    alma["bmaj"].format = ".2f"
    alma["dr"] = (
        np.sqrt(
            (alma["crval1"] - ra) ** 2 * np.cos(dec / 180 * np.pi) ** 2
            + (alma["crval2"] - dec) ** 2
        )
        * 3600.0
    )
    alma["dr"].format = ".2f"

    # print(alma['file_alma','bmaj','dr'].to_pandas())

    ny = (len(alma) + 1) // 5 + 1
    fig, axes = plt.subplots(ny, nx, figsize=(sx * nx, sx * ny), squeeze=False)

    # cutout_size = 2.6
    if cutout_size is None:
        cutout_size = np.clip(2 * alma["bmaj"].max() * 3600, 0.6, 2.6)

    url = thumb_url.format(ra=ra, dec=dec, cutout_size=cutout_size)

    if 0:
        url += "&nirspec=True"

    try:
        jw = np.array(PIL.Image.open(urllib.request.urlopen(url)))
    except:
        jw = np.zeros((10, 10))

    ax = axes[0][0]
    ax.imshow(jw, origin="upper")
    sh = jw.shape
    ax.set_xticks([0, sh[1] - 1])
    ax.set_yticks([0, sh[0] - 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.text(
        0.5,
        0.02,
        f"({ra:.6f}, {dec:.6f})",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        color="w",
        fontsize=7,
    )
    all_pos = []

    for k, file_alma in enumerate(alma["file_alma"]):
        if 1:
            # print(file_alma)
            if pre_cutout:
                cutout_args = (ra, dec, cutout_size * 1.2)
            else:
                cutout_args = None

            eco = EcogalFile(file_alma=file_alma, cutout_args=cutout_args)

            j = (k + 1) % nx
            i = (k + 1) // nx

            pos, fig_ = eco.cutout_figure(
                ra, dec, ax=axes[i][j], cutout_size=cutout_size
            )

            print(
                f"{file_alma.split('_cont')[0]:<48} b{eco.meta['band']}  {eco.shape[1]:>4}x{eco.shape[2]:>4}  {eco.pixel_scale:.2f}  dx={pos['pbcor_dx']:.2f}\""
            )

            # pos['meta'] = {}
            for mkey in eco.meta:
                pos[mkey] = eco.meta[mkey]

            all_pos.append(pos)

        else:
            pass

        # if file_alma.startswith("2013.1.01271.S__all_UDF6462_b6_cont"):
        #     # break
        #     pass

    for ki in range(k + 2, nx * ny):
        j = ki % nx
        i = ki // nx
        axes[i][j].axis("off")

    fig.tight_layout(pad=0.3)
    fig.tight_layout(pad=1)

    resp = {"query": alma, "fig": fig, "photometry": all_pos}

    return resp


def get_pbcor_metadata(
    file_alma="2022.1.01644.S__all_MOSDEF_3324_b3_cont_noninter2sig.image.pbcor.fits",
):
    """
    Read metadata from the API
    """
    import urllib.request, json

    url = f"https://grizli-cutout.herokuapp.com/ecogal_metadata?file_alma={file_alma}"
    url = url.replace("+", "%2B")

    with urllib.request.urlopen(url) as fp:
        meta = json.loads(fp.read().decode())

    return meta


class EcogalFile:

    def __init__(
        self,
        file_alma="2022.1.01644.S__all_MOSDEF_3324_b3_cont_noninter2sig.image.pbcor.fits",
        cutout_args=None,
        cache=True,
    ):
        """
        Helper functions for ecogal pbcof products
        """
        self.file_alma = file_alma

        self.meta = get_pbcor_metadata(file_alma=self.file_alma)

        if cutout_args is None:
            cache_file = get_ecogal_file(file_alma, cache=cache)

            with pyfits.open(cache_file) as im:
                self.data = np.squeeze(im[0].data * 1)
                self.header = im[0].header.copy()
        else:
            im = ecogal_cutout(file_alma, *cutout_args)
            self.data = np.squeeze(im[0].data * 1)
            self.header = im[0].header.copy()

        self.wcs = pywcs.WCS(self.header)
        self.pixel_scale = np.abs(utils.get_wcs_pscale(self.wcs))

        self.mask = np.ones_like(self.data)
        self.mask[~np.isfinite(self.data)] = np.nan

        self.primary_beam = self._primary_beam()

        self.data_sn = self.data * self.primary_beam / self.meta["noise_fit"]

    @property
    def frequency(self):
        return self.header["CRVAL3"]

    @property
    def wavelength(self):
        return 2.99e8 / self.frequency

    @property
    def footprint(self):
        return utils.SRegion(self.meta["footprint"])

    @property
    def xyz_center(self):
        """
        Pixel coordinates of ra_center, dec_center
        """
        xyz = np.squeeze(
            self.wcs.all_world2pix(
                [self.meta["ra_center"]],
                [self.meta["dec_center"]],
                [self.frequency],
                0,
            )
        )
        return xyz

    @property
    def shape(self):
        """
        3D array shape even with collapsed frequency axis
        """
        sh = self.data.shape
        if len(sh) == 2:
            NZ = 1
            NY, NX = sh
        else:
            NZ, NY, NX = sh

        return NZ, NY, NX

    def _primary_beam(self):
        """
        Generate a primary beam map using FoV_sigma and the pixel scale
        """
        NZ, NY, NX = self.shape
        xyz = self.wcs.all_world2pix(
            [self.meta["ra_center"]],
            [self.meta["dec_center"]],
            [self.frequency],
            0,
        )
        yp, xp = np.indices((NY, NX))
        Rp = (
            np.sqrt((xp - xyz[0]) ** 2 + (yp - xyz[1]) ** 2) * self.pixel_scale
        )
        return np.exp(-(Rp**2) / 2 / self.meta["FoV_sigma"] ** 2) * self.mask

    @property
    def beam(self):
        """
        Beam parameters in pixel units
        """
        b = dict(
            bmaj_arcsec=self.header["BMAJ"] * 3600.0,
            bmin_arcsec=self.header["BMIN"] * 3600.0,
            bmaj_pix=self.header["BMAJ"] * 3600.0 / self.pixel_scale,
            bmin_pix=self.header["BMIN"] * 3600.0 / self.pixel_scale,
            bpa_pix=self.header["BPA"] / 180 * np.pi + np.pi / 2,
            pixel_scale=self.pixel_scale,
        )
        b["beam_area"] = np.pi * b["bmaj_pix"] * b["bmin_pix"]

        return b

    def evaluate_position(self, ra, dec):
        """
        Evaluate pixel value and noise at a position in the image
        """
        xyz = np.squeeze(
            self.wcs.all_world2pix([ra], [dec], [self.frequency], 0)
        )
        xyzi = np.round(xyz).astype(int)

        dra = (ra - self.meta["ra_center"]) * np.cos(dec / 180 * np.pi) * 3600
        dde = (dec - self.meta["dec_center"]) * 3600
        dx = np.sqrt(dra**2 + dde**2)

        pbcor = np.exp(-(dx**2) / 2 / self.meta["FoV_sigma"] ** 2)

        resp = {
            "data": self.data[xyzi[1], xyzi[0]],
            "err": self.meta["noise_fit"] / pbcor,
            "pbcor": pbcor,
            "pbcor_dx": dx,
            "xyz": xyz,
            "xyzi": xyzi,
            "ra": ra,
            "dec": dec,
            "wavelength": self.wavelength,
            "frequency": self.frequency,
        }

        return resp

    def cutout_figure(
        self,
        ra,
        dec,
        ax=None,
        vmin=-3,
        vmax=20,
        contour_levels=[3, 5, 10, 20],
        cutout_size=None,
    ):
        """
        Make a cutout figure
        """
        import matplotlib.patches as mpatches

        b = self.beam

        xyz = np.squeeze(
            self.wcs.all_world2pix([ra], [dec], [self.frequency], 0)
        )
        xyzi = np.round(xyz).astype(int)

        bradius = np.sqrt(b["beam_area"]) / np.pi * 1.2

        if cutout_size is None:
            cutout_size = 5 * bradius * self.pixel_scale
        elif cutout_size < 0:
            cutout_size = np.maximum(
                np.abs(cutout_size), 3 * bradius * self.pixel_scale
            )

        N = int(np.round(cutout_size / self.pixel_scale))

        # print(ra, dec, xyzi, self.shape)

        slx = slice(xyzi[0] - N, xyzi[0] + N + 1)
        sly = slice(xyzi[1] - N, xyzi[1] + N + 1)

        pos = self.evaluate_position(ra, dec)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        else:
            fig = None

        ax.imshow(
            self.data_sn[sly, slx],
            vmin=vmin,
            vmax=vmax,
            cmap="bone_r",
            origin="lower",
        )

        nlev = len(contour_levels)

        ax.contour(
            self.data_sn[sly, slx],
            levels=contour_levels,
            # colors=["pink"] + ['magenta']*4,
            colors=plt.cm.cool(np.arange(nlev) / (nlev - 1.0) * 0.5 + 0.5),
            alpha=0.5,
        )

        ax.scatter(
            *(xyz - xyzi + N)[:2],
            marker="+",
            color="bisque",
            s=100,
            zorder=100,
            alpha=0.8,
        )

        ax.scatter(
            *(self.xyz_center - xyzi + N)[:2],
            marker="o",
            s=180,
            fc="None",
            ec="coral",
            lw=2,
            alpha=0.5,
            zorder=98,
        )

        ax.set_xlim(-0.5, 2 * N + 0.5)
        ax.set_ylim(-0.5, 2 * N + 0.5)

        npad = 1 + (N > 10 * b["bmaj_pix"]) * 1

        be = mpatches.Ellipse(
            (bradius * npad, bradius * npad),
            b["bmaj_pix"],
            b["bmin_pix"],
            angle=self.header["BPA"] + 90,
            lw=1,
            facecolor="0.8",
            edgecolor="0.5",
            hatch="/////",
        )

        # beam_nmad = base_nmad * 1000 * 2 # mJy / beam

        ax.add_patch(be)

        xt = np.arange(-N, N + 1) * self.pixel_scale
        if cutout_size <= 1:
            dxt = 0.2
        elif cutout_size < 5:
            dxt = 0.5
        elif cutout_size < 10:
            dxt = 1.0
        else:
            dxt = 2.0

        imax = np.floor(np.abs(xt / dxt).max()) * dxt
        yt = np.arange(-imax, imax + 0.1 * dxt, dxt)
        xti = np.interp(yt, xt, np.arange(2 * N + 1))

        ax.set_xticks(xti)
        ax.set_yticks(xti)

        ax.grid()

        label = [
            self.file_alma.split("_cont")[0] + "\n",
            f'Band {self.meta["band"]}  {self.frequency / 1.e9:.1f} GHz / {self.wavelength*1.e3:.3f} mm',
        ]

        ax.text(
            0.98,
            0.98,
            "\n".join(label),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=7,
            # bbox={'fc':'w', 'alpha':0.8, "ec": "None",}
        )

        ax.text(
            0.98,
            0.02,
            f'{pos["data"] * 1000:.4f} ± {pos["err"] * 1000:.4f} mJy / beam\n\n{self.header["BMAJ"] * 3600:.2f} x {self.header["BMIN"] * 3600:.2f}"',
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=7,
            # bbox={'fc':'w', 'alpha':0.8, "ec": "None",}
        )

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both", length=0)

        # fig.tight_layout(pad=1)

        return pos, fig

    def cutout_with_thumb(
        self, ra, dec, sx=3.0, cutout_size=None, thumb_url=RGB_URL
    ):
        """
        Cutout figure with a DJA JWST thumbnail
        """
        import PIL
        import urllib

        if cutout_size is None:
            cutout_size = np.maximum(3 * self.meta["bmaj"] * 3600.0, 0.8)

        fig, axes = plt.subplots(1, 2, figsize=(2 * sx, sx))
        url = thumb_url.format(ra=ra, dec=dec, cutout_size=cutout_size)

        if 0:
            url += "&nirspec=True"

        try:
            jw = np.array(PIL.Image.open(urllib.request.urlopen(url)))
        except:
            jw = np.ones([10, 10])

        # if jw.min() == jw.max():
        #     url = f"https://grizli-cutout.herokuapp.com/thumb?all_filters=False&size={cutout_size}&scl=2.0&asinh=True&filters=f160w&rgb_scl=1.5,0.74,1.3&pl=2&ra={props['ra'][k]}&dec={props['dec'][k]}"
        #     jw = np.array(PIL.Image.open(urllib.request.urlopen(url)))

        ax = axes[0]
        ax.imshow(jw, origin="upper")
        sh = jw.shape
        ax.set_xticks([0, sh[1] - 1])
        ax.set_yticks([0, sh[0] - 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.text(
            0.5,
            0.02,
            f"({ra:.6f}, {dec:.6f})",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            color="w",
            fontsize=7,
        )

        self.cutout_figure(ra, dec, ax=axes[1], cutout_size=cutout_size)
        fig.tight_layout(pad=1)

        return fig

    def threshold_catalog(self, threshold=4, sign=1):
        """
        Simple source catalog from segments above a S/N threshold
        """
        from skimage import measure
        import astropy.units as u

        mask = self.data_sn * sign > threshold

        labels = measure.label(mask)
        props = utils.GTable(
            measure.regionprops_table(
                labels,
                intensity_image=self.data * 1000 * sign,
                properties=["label", "bbox", "centroid", "intensity_max"],
            )
        )
        props.rename_column("centroid-0", "y")
        props.rename_column("centroid-1", "x")
        props.rename_column("intensity_max", "smax")
        props.rename_column("label", "id")
        props["smax"] *= 1000
        props["ra"], props["dec"], nu = self.wcs.all_pix2world(
            props["x"],
            props["y"],
            self.header["CRVAL3"] * np.ones(len(props)) - 1,
            0,
        )

        xi = np.round(props["x"]).astype(int)
        yi = np.round(props["y"]).astype(int)
        props["scen"] = self.data[yi, xi] * 1000
        props["pbcor"] = self.primary_beam[yi, xi]
        props["scen_err"] = self.meta["noise_fit"] / props["pbcor"] * 1000

        props["file_alma"] = self.file_alma
        props["smax"].unit = u.mJy / u.beam
        props["scen"].unit = u.mJy / u.beam
        props["scen_err"].unit = u.mJy / u.beam

        return props
