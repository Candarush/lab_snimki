import numpy as np
import cv2
import operator
import requests
import xml.etree.ElementTree as ET

# latitude  широта  Y
# longitude долгота X

SCREEN_RES = (1280, 720)

COORD_UL = -124.58436, 49.84859
COORD_UR = -121.22100, 49.84576
COORD_LL = -124.52281, 47.86034
COORD_LR = -121.29009, 47.85770

PIXEL_UL = 1620, 15
PIXEL_UR = 7852, 1376
PIXEL_LL = 50, 5951
PIXEL_LR = 6278, 7321

img = cv2.imread('LE07_L1TP_047026_20011005_20160929_01_T1_B1.tif')

def show_image():
    scale_width = SCREEN_RES[0] / img.shape[1]
    scale_height = SCREEN_RES[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', window_width, window_height)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_lon_coeff(lat, lon):
    n = COORD_UL[0] - COORD_LL[0], COORD_UL[1] - COORD_LL[1]
    print(n)
    x = (lat*n[0]*(COORD_UR[0]-COORD_UL[0]) + n[0]*lon*(COORD_UL[0]-COORD_UR[0]) + n[1]*(COORD_UR[0]*COORD_UL[1] - COORD_UL[0]*COORD_UR[1]))/((n[0]-n[1])*(COORD_UL[0]-COORD_UR[1]))
    return (x - COORD_UL[0]) / (COORD_UR[0] - COORD_UL[0])

def get_lat_coeff(lat, lon):
    return (lon - COORD_UL[1]) / (COORD_LL[1] - COORD_UL[1])

def get_pix_from_coord(lat, lon):
    coeff_lon = get_lon_coeff(lat, lon)
    coeff_lat = get_lat_coeff(lat, lon)
    point_lon = get_point_between_two(PIXEL_UL, PIXEL_UR, coeff_lon)
    point_lat = get_point_between_two(PIXEL_UL, PIXEL_LL, coeff_lat)
    return get_parallelogram_point(PIXEL_UL, point_lon, point_lat)

def draw_dot(pixelx, pixely, color):
    cv2.circle(img,(int(pixelx),int(pixely)), 10, color, -1)
    cv2.circle(img,(int(pixelx),int(pixely)), 200, color, 10)

def draw_line(pixelA, pixelB, color):
    cv2.line(img, (int(pixelA[0]),int(pixelA[1])), (int(pixelB[0]),int(pixelB[1])), color, 7)

def draw_coord_line(coordA, coordB, color):
    pixA = get_pix_from_coord(coordA[0], coordA[1])
    pixB = get_pix_from_coord(coordB[0], coordB[1])
    draw_line((pixA[0], pixA[1]), (pixB[0], pixB[1]), color)

def draw_bbox(minlon, minlat, maxlon, maxlat, color):
    draw_coord_line((minlon,minlat),(maxlon,minlat),color)
    draw_coord_line((minlon,minlat),(minlon,maxlat),color)
    draw_coord_line((maxlon,minlat),(maxlon,maxlat),color)
    draw_coord_line((minlon,maxlat),(maxlon,maxlat),color)

def get_point_between_two(point1, point2, coeff):
    y = point1[1] + (point2[1] - point1[1]) * coeff
    x = -((point2[0] - point1[0]) * y + (point1[0] * point2[1] - point2[0] * point1[1])) / (point1[1] - point2[1])
    return (x, y)

def get_parallelogram_point(origin, point1, point2):
    return point1[0] + point2[0] - origin[0], point1[1] + point2[1] - origin[1]

def download_city_data(city_relation):
    url = 'http://overpass-api.de/api/interpreter?data=rel({0});out geom;'.format(str(city_relation))
    open('city_data.xml', 'wb').write(requests.get(url, allow_redirects=True).content)
    data = open('city_data.xml','r').read()
    data_xmltree = ET.parse('city_data.xml').getroot()
    bbox = data_xmltree[2][0].attrib
    return  {
                'lon' : (float(bbox["minlon"]) + float(bbox["maxlon"])) / 2, 
                'lat' : (float(bbox["minlat"]) + float(bbox["maxlat"])) / 2,
                'minlon' : float(bbox["minlon"]),
                'maxlon' : float(bbox["maxlon"]),
                'minlat' : float(bbox["minlat"]),
                'maxlat' : float(bbox["maxlat"])
            }

if __name__ == '__main__':
    city = download_city_data(1852574)
    print(city)
    #draw_bbox(city["minlon"], city["minlat"], city["maxlon"], city["maxlat"], (0, 0 ,255))
    print(get_lon_coeff(city["minlon"], city["minlat"]))
    show_image()