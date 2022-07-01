#from sunpy.coordinates.ephemeris import get_horizons_coord
#from numpy import sqrt, log, pi
#from astropy import constants as const
#from astropy import units as u
#import sunpy.sun.constants as sconst
#import datetime

import ipywidgets as widgets
from IPython.display import display

# a list of available spacecraft:
list_of_sc = ["STEREO-A", "STEREO-B", "Solar Orbiter", "Bepicolombo", "SOHO"]

stereo_instr = ["LET", "SEPT", "HET"]
solo_instr = ["EPT", "HET"]
bepi_instr = ["SIXS-P"]
soho_instr = ["ERNE", ]

sensor_dict = {
    "STEREO-A" : stereo_instr,
    "STEREO-B" : stereo_instr,
    "Solar Orbiter" : solo_instr,
    "Bepicolombo" : bepi_instr,
    "SOHO" : soho_instr
}

view_dict = {
    "SEPT" : ["sun", "asun", "north", "south"],
    "SIXS-P" : [0, 1, 2, 3, 4]
}

# init this as None
view_drop = None

def spacecraft_dropdown():

    global spacecraft_drop

    spacecraft_drop = widgets.Dropdown(
                                options = list_of_sc,
                                value = list_of_sc[0],
                                description = "Spacecraft:",
                                disabled = False,
                                )

    return spacecraft_drop


def sensor_dropdown(spacecraft_key):

    global sensor_drop

    sensor_list = sensor_dict[spacecraft_key]

    sensor_drop = widgets.Dropdown(
                                options = sensor_list,
                                value = sensor_list[0],
                                description = "Sensor:",
                                disabled = False,
                                )

    return sensor_drop


def viewing_dropdown(instrument_key):

    global view_drop

    try:
        viewing_list = view_dict[instrument_key]
    except KeyError:
        errormsg = "No viewing option available for this sensor."
        return errormsg

    view_drop = widgets.Dropdown(
                                options = viewing_list,
                                value = viewing_list[0],
                                description = "Viewing:",
                                disabled = False,
                                )

    return view_drop

def display_input():

    print("You've chosen the following options:")
    print(f"Spacecraft: {spacecraft_drop.value}")
    print(f"Sensor: {sensor_drop.value}")
    if view_drop:
        print(f"Viewing: {view_drop.value}")
    