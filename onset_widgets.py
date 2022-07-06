"""
A library to run the interactive user interface in SEP event onset determination notebooks.
"""


import ipywidgets as widgets

# a list of available spacecraft:
list_of_sc = ["STEREO-A", "STEREO-B", "Solar Orbiter", "Bepicolombo", "SOHO"]

stereo_instr = ["LET", "SEPT", "HET"]
solo_instr = ["EPT", "HET"]
bepi_instr = ["SIXS-P"]
soho_instr = ["ERNE-HED"]

sensor_dict = {
    "STEREO-A" : stereo_instr,
    "STEREO-B" : stereo_instr,
    "Solar Orbiter" : solo_instr,
    "Bepicolombo" : bepi_instr,
    "SOHO" : soho_instr
}

view_dict = {
    ("STEREO-A", "SEPT") : ["sun", "asun", "north", "south"],
    ("STEREO-B", "SEPT") : ["sun", "asun", "north", "south"],
    ("Solar Orbiter", "EPT") : ["sun", "asun", "north", "south"],
    ("Solar Orbiter", "HET") : ["sun", "asun", "north", "south"],
    ("Bepicolombo", "SIXS-P") : [0, 1, 2, 3, 4]
}

species_dict = {
    ("STEREO-A", "LET") : ['p', 'e'],
    ("STEREO-A", "SEPT") : ['p', 'e'],
    ("STEREO-A", "HET") : ['H', 'e'],
    ("STEREO-B", "LET") : ['p', 'e'],
    ("STEREO-B", "SEPT") : ['p', 'e'],
    ("STEREO-B", "HET") : ['H', 'e'],
    ("Solar Orbiter", "EPT") : ['p', 'e'],
    ("Solar Orbiter", "HET") : ['p', 'e'],
    ("Bepicolombo", "SIXS-P") : ['p', 'e'],
    ("SOHO", "ERNE-HED") : ['p'],
}

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
    """
    instrument_key is a 2-tuple consisting of spacecraft identifier and a sensor identifier.
    This is because there are sensors with the same name on board different spacecraft, which have different viewing options.
    """

    # initialize these global variables for the first run of the notebook
    global view_drop
    view_drop = None

    global viewing_opened
    viewing_opened = False

    try:
        viewing_list = view_dict[instrument_key]
        viewing_opened = True

    except KeyError:
        errormsg = "No viewing option available for this sensor."
        if viewing_opened:
            view_drop = None
        return errormsg

    view_drop = widgets.Dropdown(
                                options = viewing_list,
                                value = viewing_list[0],
                                description = "Viewing:",
                                disabled = False,
                                )

    return view_drop


def species_dropdown(instrument_key):

    global species_drop

    species_list = species_dict[instrument_key]

    species_drop = widgets.Dropdown(
                                options = species_list,
                                value = species_list[0],
                                description = "Species:",
                                disabled = False,
                                )

    return species_drop


def update_and_display_input(event_date : int, data_path : str, plot_path : str):

    try:
        if view_drop:
            view_drop_dict_value = view_drop.value
        else:
            view_drop_dict_value = None

    # this is caused by running this function before viewing_drop() has been run,
    # which in itself is not necessarily a problem
    except NameError:
        print("NameError")
        view_drop_dict_value = None

    # this is to be fed into Event class as input
    global input_dict

    # we differentiate between erne-hed and erne, but the main onset analysis class only recognizes 'erne'
    # same for solar orbiter
    if spacecraft_drop.value == "ERNE-HED":
        spacecraft_drop_value = "ERNE"
    elif spacecraft_drop.value == "Solar Orbiter":
        spacecraft_drop_value = "solo"
    else:
        spacecraft_drop_value = spacecraft_drop.value

    input_dict = {
        "Spacecraft" : spacecraft_drop_value,
        "Sensor" : sensor_drop.value,
        "Species" : species_drop.value,
        "Viewing" : view_drop_dict_value,
        "Event_date" : event_date,
        "Data_path" : data_path,
        "Plot_path" : plot_path
    }

    print("You've chosen the following options:")
    print(f"Spacecraft: {input_dict['Spacecraft']}")
    print(f"Sensor: {input_dict['Sensor']}")
    print(f"Species: {input_dict['Species']}")
    print(f"Viewing: {input_dict['Viewing']}")
    print(f"Event_date: {input_dict['Event_date']}")
    print(f"Data_path: {input_dict['Data_path']}")
    print(f"Plot_path: {input_dict['Plot_path']}")
