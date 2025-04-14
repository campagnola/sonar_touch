import os, pickle
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
import numpy as np

import pyqtgraph as pg
pg.dbg()


def load_bigsky():
    import pandas
    # starplot downloaded this for me.. no idea from where
    star_df = pandas.read_csv('bigsky.0.4.0.stars.csv')
    star_df = star_df[star_df['magnitude'] < 5.5]
    return star_df



def load_gaia():
    cache_file = 'gaia_cache.pkl'

    # Define the ADQL query to fetch the brightest 10,000 stars with necessary parameters
    query = """
    SELECT TOP 1000
        source_id, ra, dec, parallax, pmra, pmdec, radial_velocity, phot_g_mean_mag, designation
    FROM gaiadr3.gaia_source
    WHERE parallax IS NOT NULL
    AND pmra IS NOT NULL
    AND pmdec IS NOT NULL
    AND radial_velocity IS NOT NULL
    AND phot_g_mean_mag IS NOT NULL
    ORDER BY phot_g_mean_mag ASC
    """

    # load cache if it exists
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_results = pickle.load(f)
    else:
        cached_results = {}

    if query in cached_results:
        print("Using cached results")
        results = cached_results[query]
    else:
        # Launch the query
        print("Querying Gaia..")
        job = Gaia.launch_job_async(query)
        results = job.get_results()

        named = Simbad.query_objects(results["designation"])
        named.keep_columns("main_id")
        results.update(named)

        # Save the results to cache
        cached_results[query] = results
        cache_str = pickle.dumps(cached_results)
        with open(cache_file, 'wb') as f:
            f.write(cache_str)




    from astropy.units import Quantity

    # Clip parallax to prevent zero or negative values
    def parallax_to_distance(parallax_column):
        # Convert MaskedColumn to float array, preserving units
        if hasattr(parallax_column, 'unit'):
            parallax_mas = np.asarray(parallax_column) * parallax_column.unit
        else:
            parallax_mas = u.Quantity(parallax_column, u.mas)

        # Strip to pure float and compute distance
        return (1000.0 / parallax_mas.to(u.mas).value) * u.pc

    parallax = np.clip(results['parallax'], 0.01, np.inf)  # use a small floor to avoid huge distances

    coords = SkyCoord(
        ra=Quantity(results['ra'], u.deg),
        dec=Quantity(results['dec'], u.deg),
        distance=parallax_to_distance(parallax),
        pm_ra_cosdec=Quantity(results['pmra'], u.mas/u.yr),
        pm_dec=Quantity(results['pmdec'], u.mas/u.yr),
        radial_velocity=Quantity(results['radial_velocity'], u.km/u.s)
    )

    # Compute Cartesian positions in parsecs
    positions = coords.cartesian.xyz.to(u.pc).value.T  # Shape: (10000, 3)

    # positions = np.zeros(

    # Compute motion vectors in pc/yr
    velocities = coords.velocity.d_xyz.to(u.pc/u.yr).value.T  # Shape: (10000, 3)

    # convert magnitude to brightness
    magnitude = results['phot_g_mean_mag'].data.data

    return results, positions, velocities, magnitude


def load_hpx():
    cache_file = 'hpx_cache.pkl'
    if os.path.exists(cache_file):
        print("Loading cached XHIP data")
        with open(cache_file, 'rb') as f:
            xhip_bright = pickle.load(f)
    else:
        print("Loading XHIP data from VizieR")
        from astroquery.vizier import Vizier

        # Catalog ID for XHIP in VizieR
        # catalog_id = "V/137D"
        catalog_id = "I/322A"

        # Select useful columns
        columns = ["*"]

        # Load Vizier with all columns
        v = Vizier(columns=columns, row_limit=10000)

        # Query for stars with Vmag < 6.5 (approximate naked-eye brightness)
        result = v.query_constraints(catalog=catalog_id, Vmag="<6.5")

        # Get the main table
        xhip_bright = result[0]

        # Save the results to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(xhip_bright, f)

    return xhip_bright


import coorx

import numpy as np
from coorx import Transform

import numpy as np
from coorx import Transform

class SphericalTransform(Transform):
    """Converts Cartesian (x, y, z) to spherical (lon(rad), lat(rad), r)."""
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, dims=(3, 3), **kwargs):
        super().__init__(dims, **kwargs)

    def _map(self, coords):
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        lon = np.arctan2(y, x)
        lat = np.arcsin(z / r)

        ret = np.empty(coords.shape, dtype=coords.dtype)
        ret[..., 0] = lon
        ret[..., 1] = lat
        ret[..., 2] = r
        return ret

    def _imap(self, coords):
        lon, lat, r = coords[..., 0], coords[..., 1], coords[..., 2]
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)

        ret = np.empty(coords.shape, dtype=coords.dtype)
        ret[..., 0] = x
        ret[..., 1] = y
        ret[..., 2] = z
        return ret

    @property
    def params(self):
        return {}

    def set_params(self, **params):
        return


class MercatorSphericalTransform(Transform):
    """Maps (lon, lat, z) → (x, y, z) using Mercator projection (ignores r)."""
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, dims=(3, 3), **kwargs):
        super().__init__(dims, **kwargs)

    def _map(self, coords):
        lon = coords[..., 0]
        lat = np.clip(coords[..., 1], -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
        x = lon
        y = np.log(np.tan(np.pi / 4 + lat / 2))

        ret = np.empty_like(coords)
        ret[..., 0] = x
        ret[..., 1] = y
        ret[..., 2:] = coords[..., 2:]  # preserve extra axes like brightness
        return ret

    def _imap(self, coords):
        x = coords[..., 0]
        y = coords[..., 1]
        lon = x
        lat = 2 * np.arctan(np.exp(y)) - np.pi / 2

        ret = np.empty_like(coords)
        ret[..., 0] = lon
        ret[..., 1] = lat
        ret[..., 2:] = coords[..., 2:]  # preserve extra axes
        return ret

    @property
    def params(self):
        return {}

    def set_params(self, **params):
        return


class LambertAzimuthalEqualAreaTransform(Transform):
    """Projects (lon, lat, z) → (x, y, z) using Lambert Azimuthal Equal-Area projection."""
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, dims=(3, 3), **kwargs):
        super().__init__(dims, **kwargs)

    def _map(self, coords):
        # Input in radians
        lon = coords[..., 0]
        lat = coords[..., 1]

        # Project relative to north pole (φ₀ = π/2, λ₀ = 0)
        k = np.sqrt(2 / (1 + np.sin(lat)))
        x = k * np.cos(lat) * np.sin(lon)
        y = -k * np.cos(lat) * np.cos(lon)  # y is negative so pole is on top

        ret = np.empty_like(coords)
        ret[..., 0] = x
        ret[..., 1] = y
        ret[..., 2:] = coords[..., 2:]  # preserve z or other axes
        return ret

    def _imap(self, coords):
        x = coords[..., 0]
        y = coords[..., 1]

        rho_sq = x**2 + y**2
        rho = np.sqrt(rho_sq)
        c = 2 * np.arcsin(np.minimum(rho / 2, 1.0))  # clip to avoid domain error

        sin_c = np.sin(c)
        cos_c = np.cos(c)

        # Inverse for center at φ₀ = π/2
        lat = np.arcsin(cos_c)
        lon = np.arctan2(x * sin_c, -y * sin_c)

        ret = np.empty_like(coords)
        ret[..., 0] = lon
        ret[..., 1] = lat
        ret[..., 2:] = coords[..., 2:]  # preserve z or other axes
        return ret

    @property
    def params(self):
        return {}

    def set_params(self, **params):
        return


import math

# from https://stackoverflow.com/a/45497817/643629
def bv_to_temperature(bv):
    return 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62))

redco = np.poly1d([ 1.62098281e-82, -5.03110845e-77, 6.66758278e-72, -4.71441850e-67, 1.66429493e-62, -1.50701672e-59, -2.42533006e-53, 8.42586475e-49, 7.94816523e-45, -1.68655179e-39, 7.25404556e-35, -1.85559350e-30, 3.23793430e-26, -4.00670131e-22, 3.53445102e-18, -2.19200432e-14, 9.27939743e-11, -2.56131914e-07,  4.29917840e-04, -3.88866019e-01, 3.97307766e+02])
greenco = np.poly1d([ 1.21775217e-82, -3.79265302e-77, 5.04300808e-72, -3.57741292e-67, 1.26763387e-62, -1.28724846e-59, -1.84618419e-53, 6.43113038e-49, 6.05135293e-45, -1.28642374e-39, 5.52273817e-35, -1.40682723e-30, 2.43659251e-26, -2.97762151e-22, 2.57295370e-18, -1.54137817e-14, 6.14141996e-11, -1.50922703e-07,  1.90667190e-04, -1.23973583e-02,-1.33464366e+01])
blueco = np.poly1d([ 2.17374683e-82, -6.82574350e-77, 9.17262316e-72, -6.60390151e-67, 2.40324203e-62, -5.77694976e-59, -3.42234361e-53, 1.26662864e-48, 8.75794575e-45, -2.45089758e-39, 1.10698770e-34, -2.95752654e-30, 5.41656027e-26, -7.10396545e-22, 6.74083578e-18, -4.59335728e-14, 2.20051751e-10, -7.14068799e-07,  1.46622559e-03, -1.60740964e+00, 6.85200095e+02])

def temperature_to_rgb(temp):
    return (
        np.clip(redco(temp), 0, 255),
        np.clip(greenco(temp), 0, 255),
        np.clip(blueco(temp), 0, 255),
    )

def btvt_to_rgb(btvt):
    temperature = bv_to_temperature(btvt)
    return temperature_to_rgb(temperature)



if __name__ == '__main__':


    import pyqtgraph as pg
    pg.mkQApp()

    w = pg.GraphicsLayoutWidget()
    w.resize(800, 600)

    class StarViewBox(pg.ViewBox):
        def __init__(self, data):
            pg.ViewBox.__init__(self)
            self.setAspectLocked()
            self.data = data
            
            deg_to_rad = np.pi / 180.0
            mas_to_rad = deg_to_rad / 3600 / 1000

            positions = np.zeros((len(data), 3))
            positions[:, 0] = -data['ra_degrees_j2000'] * deg_to_rad
            positions[:, 1] = data['dec_degrees_j2000'] * deg_to_rad
            positions[:, 2] = 1 / data['parallax_mas'] * 1000.0
            tr = SphericalTransform()
            self.positions = tr.imap(positions)

            positions[:, 0] += data['ra_mas_per_year'] * mas_to_rad
            positions[:, 1] += data['dec_mas_per_year'] * mas_to_rad
            self.vectors = tr.imap(positions) - self.positions

            self.magnitudes = np.asarray(data['magnitude'])
            self.names = np.asarray(data['name'])
            self.time = 0

            self.angle = [0, 0]
            self.rotation_tr = coorx.AffineTransform(dims=(3, 3))
            self.perspective_tr = coorx.linear.PerspectiveTransform()
            self.perspective_tr.set_perspective(fovy=60, aspect=1.0, znear=0.001, zfar=100000.0)
            self.projection = coorx.CompositeTransform([
                # SphericalTransform().inverse,
                self.rotation_tr,
                SphericalTransform(),
                # MercatorSphericalTransform()
                # self.perspective_tr,
                LambertAzimuthalEqualAreaTransform(),
            ])

            self.update_positions()

            magnitudes = self.magnitudes
            brightness = 15 * ((5.5 - magnitudes) / 5.5)**2
            self.sizes = np.clip(brightness, 1, np.inf)
            self.alphas = 255 * np.clip(brightness, 0.0, 1.0)

            self.brushes = []
            for i, (index, row) in enumerate(data.iterrows()):
                bv = row['bv']
                if np.isnan(bv):
                    color = (255, 255, 255)
                else:
                    # B-V = 0.850 * (BT-VT)
                    color = np.array(btvt_to_rgb(bv))
                    mix = 0.5
                    color = (mix * color + (1 - mix) * 255)
                alpha = self.alphas[i]
                self.brushes.append(pg.mkBrush(color[0], color[1], color[2], alpha))

            self.scatter = pg.ScatterPlotItem(
                pos=self.mapped_pos,
                size=self.sizes,
                pen=None,
                brush=self.brushes,
                symbol='o',
                pxMode=True,
                data=np.arange(len(self.data)),
            )
            self.addItem(self.scatter)

            self.scatter.sigClicked.connect(self.scatterClicked)

            self.traveler_lines = pg.PlotCurveItem(pen=(255, 255, 255, 100))
            self.addItem(self.traveler_lines)

            self.update_stars()

            self.setXRange(-2, 2)
            self.setYRange(-2, 2)

            self.timer = pg.QtCore.QTimer()
            self.timer.timeout.connect(self.update_time)
            self.timer.start(16)

        def mouseDragEvent(self, ev, axis=None):
            ev.accept()
            global e
            e = ev
            if ev.isStart():
                return
            if ev.isFinish():
                return
            delta = e.lastScenePos() - e.scenePos()
            self.rotation_tr.rotate(delta.y() * 0.3, axis=(0, 1, 0))
            self.rotation_tr.rotate(-delta.x() * 0.3, axis=(1, 0, 0))
            self.update_stars()

        def update_positions(self):
            pos = self.positions + self.vectors * self.time
            self.mapped_pos = self.projection.map(pos)[..., :2]

        def update_stars(self):
            self.update_positions()
            self.update_lines()
            self.scatter.setData(
                pos=self.mapped_pos,
                size=self.sizes,
                brush=self.brushes,
                data=np.arange(len(self.data)),
            )

        def update_lines(self):
            stars_to_draw = ['Vega', 'Sirius', 'Capella', 'Arcturus', 'Altair', 'Aljanah']
            verts = []
            connect = []
            for star in stars_to_draw:
                ind = np.argwhere(self.names == star)[0,0]
                pos = self.mapped_pos[ind]
                vec = self.vectors[ind] * self.time
                pos = self.positions[ind:ind+1, :] + self.vectors[ind:ind+1, :] * np.linspace(-100000, 100000, 10)[:, np.newaxis]
                verts.append(pos)
                connect.append(np.ones(pos.shape[0], dtype=bool))
                connect[-1][-1] = False
            verts = self.projection.map(np.concatenate(verts))
            connect = np.concatenate(connect)
            self.traveler_lines.setData(verts[:,0], verts[:,1], connect=connect)

        def update_time(self):
            self.time += 1000
            if self.time > 100000:
                self.time = -100000
            self.update_stars()

        def scatterClicked(self, item, points):
            if len(points) == 0:
                return
            print(f"Clicked on {len(points)} points")
            for pt in points:
                rec = self.data.iloc[pt.data()]
                print(rec['name'])

    # hpx_data = load_hpx()
    # positions = np.zeros((len(hpx_data), 3))
    # positions[:, 0] = hpx_data['RAJ2000'].data.data
    # positions[:, 1] = hpx_data['DEJ2000'].data.data
    # positions[:, 2] = 100
    # tr = SphericalTransform()
    # positions = tr.imap(positions)

    # velocities = np.zeros((len(hpx_data), 3))
    # velocities[:, 0] = hpx_data['pmRA'].data.data
    # velocities[:, 1] = hpx_data['pmDE'].data.data

    # magnitude = hpx_data['Vmag'].data.data
    # names = np.array(['']*len(hpx_data), dtype=object)
    # stars = (np.array([
    #     [88.79, 7.41], # betelgeuse
    #     [37.95, 89.26], # polaris
    #     [310.45, 45.28], # vega
    # ]), ['betelgeuse', 'polaris', 'vega'])
    # for i, row in enumerate(hpx_data):
    #     pos = np.array([[row['RAJ2000'], row['DEJ2000']]])
    #     dist = np.linalg.norm(pos - stars[0], axis=1)
    #     min_ind = np.argmin(dist)
    #     if dist[min_ind] < 0.5:
    #         print("Found star: ", stars[1][min_ind])
    #         names[i] = stars[1][min_ind]

    stars = load_bigsky()


    view = StarViewBox(
        data=stars,
    )
    w.addItem(view)




    # g = pg.GridItem()
    # v.addItem(g)
    w.show()


    # pyqtgraph 3D
    # import pyqtgraph.opengl as gl
    # import pyqtgraph as pg
    # app = pg.mkQApp()

    # view = gl.GLViewWidget()

    # scatter = gl.GLScatterPlotItem(pos=positions, size=brightness * 10, color=(1, 1, 1, 0.5), pxMode=True)
    # scatter.setGLOptions('additive')
    # view.addItem(scatter)
    # view.setCameraPosition(distance=.01)

    # view.show()



    # # try vispy instead
    # import vispy
    # from vispy import app, scene

    # # Create a canvas
    # canvas = scene.SceneCanvas(keys='interactive', show=True)
    # canvas.size = (800, 600)

    # view = canvas.central_widget.add_view()

    # # # Create a 3D view
    # # view.camera = scene.cameras.TurntableCamera(fov=45, distance=0.01)

    # # 2d view
    # view.camera = scene.cameras.PanZoomCamera()

    # # Create a scatter plot
    # scatter = scene.Markers(
    #     pos=mapped_pos[..., :2],                    
    #     parent=view.scene, 
    #     size=brightness, 
    #     # scaling='visual', 
    #     edge_color=(0, 0, 0, 0), 
    #     face_color=(1, 1, 1, 0.5)
    # )
    # # scatter.set_data(mapped_pos[..., :2])
    # # additive
    # scatter.set_gl_state('additive', blend=True, depth_test=False)

    # canvas.show()