import os
import numpy as np
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from .transforms import SphericalTransform


class StarCatalog:
    def __init__(self, data):
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
        self.data = data



def load_bigsky():
    """
    Requires the bigsky star catalog from https://github.com/steveberardi/bigsky/releases

    Unzip it to the current directory
    """
    import pandas

    url = 'https://github.com/steveberardi/bigsky/releases/download/v0.4.0/bigsky.0.4.0.stars.csv.gz'
    path = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(path, 'bigsky.0.4.0.stars.csv')
    if not os.path.exists(csv_file):
        import urllib.request
        import gzip
        print("Downloading bigsky star catalog")
        urllib.request.urlretrieve(url, csv_file+'.gz')
        print("Download complete; unpacking..")
        with gzip.open(csv_file+'.gz', 'rb') as f_in:
            with open(csv_file, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(csv_file+'.gz')
    print("Loading star catalog")        
    star_df = pandas.read_csv(csv_file)
    star_df = star_df[star_df['magnitude'] < 5.5]
    return star_df



def load_gaia():
    """NOTE: Gaia does not have data on bright stars; not useful here"""
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
