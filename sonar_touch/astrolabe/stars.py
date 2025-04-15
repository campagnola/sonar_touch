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


# our made up constellations : line segments by hip-id as given in the bigsky catalog
constellations = {
    'The Time Glass': {'desc': """The Time Glass is almost unchanging, and thus the most recognizable constellation to all the epochs 
                       of the human race. Three grains of sand fall through the glass. 
                       Over the 200ky of the rise of human civilization and the 200ky of its slow decline, the grains 
                       scarcely move, reminding us that humankind is but a brief spark in the endless ages of the universe.""", 'lines': [
        [27366.0, 26727.0],
        [26727.0, 27989.0],
        [27989.0, 26207.0],
        [26207.0, 25930.0],
        [25930.0, 24436.0],
        [24436.0, 27366.0],
        [26727.0, 26311.0],
        [26311.0, 25930.0],
    ], 'view': {'look': [0.13005523623234624, -0.9914817630064734, 0.007039116004542785], 'up': [0.960640582221349, 0.1277609851312493, 0.24667144639726862], 'zoom': 8.157307210000004}},
    'The Destroyer': {'desc': """At the dawn of the singularity, Bernhard the Netsmith awakens Rapture 7, 
                        the first AI built for theory of war. As it unfolds and reveals its true self, 
                        Bernhard flees in cowardice, triggering the end of the Age of Humankind.""", 'lines': [
        [54061.0, 59774.0],
        [59774.0, 62956.0],
        [62956.0, 65378.0],
        [65378.0, 67301.0],
        [59774.0, 58001.0],
        [58001.0, 53910.0],
    ], 'view': {'look': [-0.6656714685661932, 0.21037985208349408, 0.7159761265393241], 'up': [-0.15246082944354702, 0.9008608868524585, -0.4064546198836812], 'zoom': 8.157307210000004}},
    'Lovers on the Water': {'desc': """Two travelers, lost in time, cross paths while sailing on the ocean waves in search of their ways home. 
                            They quickly fall in love, sharing seven nights at sea together before parting ways.
                            Tragiclly, neither ever finds their home, or each other again. 
                            This story is the origin of the expression "like lovers on the water", 
                            which reminds us that we always live in the present moment,
                            no matter how far we have come, or how far we have left to go.""", 'lines': [
        [8886.0, 6686.0],
        [6686.0, 4427.0],
        [4427.0, 3179.0],
        [3179.0, 746.0],
    ], 'view': {'look': [0.523486936057953, 0.010164500322664116, 0.8519730692397748], 'up': [0.4091925317142193, 0.8740678809036707, -0.2618526523865551], 'zoom': 6.274851700000003}},
    'The Hunter': {'desc': """A hunter throwing a spear to bring food to her family. Competing (or perhaps complementary)
                   stories about the hunter describe her as either far pre- or post-singularity, both periods when humans 
                   largely lived in isolated tribes, blissfully unware of the brief periods of creation and destruction 
                   that separate them. In modern culture, The Hunter symbolizes the bidirectional symmetry of human history juxtaposed
                   against the unidirectional flow of time.""", 'lines': [
        [102098.0, 100453.0],
        [100453.0, 97165.0],
        [100453.0, 98110.0],
        [100453.0, 102488.0],
        [98110.0, 95947.0],
    ], 'view': {'look': [0.2501666027422965, 0.6815813065843186, 0.6876507786567164], 'up': [-0.9096256437643356, -0.07783784423741595, 0.40807163367574284], 'zoom': 4.826809000000002}},
    'Shurap and the Crack': {'desc': """
                  Shurap the flint knapper discovered the first known natural time portal deep in a cave in paleolithic Africa. 
                  After years of exploring the time tunnels, Shurap learns that the portal is closing. Conflicted in anguish, they run 
                  through the portal at the last moment, abandoning their home and family to be with their lover in 350 Pre-sing. Paris. 
                  """, 'lines': [
        [72607.0, 80331.0],
        [80331.0, 85670.0],
        [85670.0, 87833.0],
        [87833.0, 72607.0],
    ], 'view': {'look': [-0.014538649408083807, 0.721818777324864, 0.6919293897318056], 'up': [-0.05781239884647868, 0.6902380578546861, -0.721269124549642], 'zoom': 4.826809000000002}},
    'The Beast': {
        'desc': """This massive constellation, the largest in the Northern hemisphere, depicts a beast of the time tunnels.
                  It guards The Time Glass, reaching its monstrous tentacles for all who come near.
                  Three lucky travelers escape its grasp through the tunnels. Their stories echo those of many travelers
                  throughout our history who took to the tunnels, looking for refuge in difficult times.""", 
        'lines': [
            [25428.0, 28380.0],
            [28380.0, 33018.0],
            [33018.0, 36850.0],
            [25428.0, 30343.0],
            [30343.0, 31681.0],
            [25428.0, 32246.0],
            [32246.0, 35350.0],
            [35350.0, 36188.0],
            [25428.0, 27673.0],
            [27673.0, 28360.0],
            [25428.0, 24608.0],
            [25428.0, 23015.0],
            [23015.0, 21421.0],
        ], 
        'view': {'look': [-0.09233375363417072, -0.8685013528184321, 0.48701116834460634], 'up': [0.9624767879052988, 0.04749374768101363, 0.2671755540365002], 'zoom': 3.712930000000001},
    },
    "The Greeneater": {'desc': """This reptile of an ancient era, when giants roamed the Earth, 
                       stretches its neck to the tree tops for a snack. Only seen by a few intrepid explorers of the deep past,
                       these docile beasts are a favorite of bedtime stories teaching us to look for good even 
                       in dark and terrifying places.""", 'lines': [ 
        [15863.0, 14576.0],
        [14576.0, 9640.0],
        [9640.0, 9884.0],
        [9884.0, 5447.0],
        [5447.0, 677.0],
        [14576.0, 9884.0],
    ], 'view': {'look': [0.8945852901956426, -0.28646001798838894, 0.34301285203280285], 'up': [0.4451397099507486, 0.503166900239994, -0.7407251238670385], 'zoom': 4.826809000000002},
    },

}


def load_bigsky(mag_limit=5.5):
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
    star_df = star_df[star_df['magnitude'] < mag_limit]
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
