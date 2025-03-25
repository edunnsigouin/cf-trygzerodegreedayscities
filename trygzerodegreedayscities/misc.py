"""
collection of useful miscellaneous functions                                                                                                                  
"""
import numpy  as np
import xarray as xr


def tic():
    # Homemade version of matlab tic function
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    # Homemade version of matlab tic function
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


def get_city_bboxes(city):
    """
    Returns a dictionary of city bounding boxes.
    Each entry has lat_min, lat_max, lon_min, lon_max.
    """

    bboxes = {
        "Oslo": {
            "lat_min": 59.70, "lat_max": 60.25,
            "lon_min": 10.40, "lon_max": 11.10
        },
        "Kristiansand": {
            "lat_min": 58.00, "lat_max": 58.30,
            "lon_min":  7.80, "lon_max":  8.20
        },
        "Stavanger": {
            "lat_min": 58.80, "lat_max": 59.10,
            "lon_min":  5.50, "lon_max":  6.00
        },
        "Bergen": {
            "lat_min": 60.20, "lat_max": 60.55,
            "lon_min":  5.10, "lon_max":  5.50
        },
        "Ålesund": {
            "lat_min": 62.30, "lat_max": 62.60,
            "lon_min":  5.90, "lon_max":  6.40
        },
        "Trondheim": {
            "lat_min": 63.30, "lat_max": 63.55,
            "lon_min": 10.20, "lon_max": 10.60
        },
        "Bodø": {
            "lat_min": 67.10, "lat_max": 67.40,
            "lon_min": 14.20, "lon_max": 14.60
        },
        "Tromsø": {
            "lat_min": 69.50, "lat_max": 69.80,
            "lon_min": 18.60, "lon_max": 19.20
        },
        "Lillehammer": {
            "lat_min": 61.00, "lat_max": 61.20,
            "lon_min": 10.20, "lon_max": 10.70
        },
        "Alta": {
            "lat_min": 69.80, "lat_max": 70.10,
            "lon_min": 23.00, "lon_max": 23.50
        }
    }

    return bboxes[city]
