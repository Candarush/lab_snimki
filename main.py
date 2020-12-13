import numpy as np
import cv2
import operator
import requests
import xml.etree.ElementTree as ET
import math

# Разрешение экрана.
SCREEN_RES = (1366, 768)

PIXEL_UL = 1620, 15
PIXEL_UR = 7852, 1376
PIXEL_LL = 50, 5951
PIXEL_LR = 6278, 7321

TARGET = {'minlon': -123.5039, 'maxlon': -123.0297, 'minlat': 49.3195, 'maxlat': 48.9413}

def parse_arr(array, valuename):
    for item in array:
        if item.find(valuename) != -1:
            return float(item.split('=',1)[1])

# Читаем MTL файл.
try:
    mtl = open('LE07_L1TP_047026_20011005_20160929_01_T1_MTL.txt', 'r')
    mtl_arr = mtl.read().split('\n')
    COORD_UL = parse_arr(mtl_arr, 'CORNER_UL_LON_PRODUCT'), parse_arr(mtl_arr, 'CORNER_UL_LAT_PRODUCT')
    COORD_UR = parse_arr(mtl_arr, 'CORNER_UR_LON_PRODUCT'), parse_arr(mtl_arr, 'CORNER_UR_LAT_PRODUCT')
    COORD_LL = parse_arr(mtl_arr, 'CORNER_LL_LON_PRODUCT'), parse_arr(mtl_arr, 'CORNER_LL_LAT_PRODUCT')
    COORD_LR = parse_arr(mtl_arr, 'CORNER_LR_LON_PRODUCT'), parse_arr(mtl_arr, 'CORNER_LR_LAT_PRODUCT')
finally:
    mtl.close()

# Читаем картинку.
red_img = cv2.imread('LE07_L1TP_047026_20011005_20160929_01_T1_B3.tif')
nir_img = cv2.imread('LE07_L1TP_047026_20011005_20160929_01_T1_B4.tif')

B3_GAIN = 0.621654
B3_BIAS = -5.62

B4_GAIN = 0.639764
B4_BIAS = -5.74

EARTH_SUN_DISTANCE = 0.9999841

B3_SUN_RADIANCE = 1533
B4_SUN_RADIANCE = 1039

ndvi_gradient_img = cv2.imread('ndvi_gradient3.png')

def show_image(mat):
    ''' Показывает текущее состояние прочитанного изображения img
    '''
    scale_width = SCREEN_RES[0] / mat.shape[1]
    scale_height = SCREEN_RES[1] / mat.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(mat.shape[1] * scale)
    window_height = int(mat.shape[0] * scale)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', window_width, window_height)
    cv2.imshow('Result', mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_pix_from_coord(lon, lat):
    ''' Функция высчитывающая положение пикселя по широте и долготе.
        Возвнащаяет кортеж (x,y).
    '''
    UB = (COORD_UL[1] + COORD_UR[1])/2
    BB = (COORD_LL[1] + COORD_LR[1])/2
    LB = (COORD_UL[0] + COORD_LL[0])/2
    RB = (COORD_UR[0] + COORD_LR[0])/2
    coeff_lon = (lon - LB) / (RB - LB)
    coeff_lat = (lat - UB) / (BB - UB)
    point_lon = get_point_between_two(PIXEL_UL, PIXEL_UR, coeff_lon)
    point_lat = get_point_between_two(PIXEL_UL, PIXEL_LL, coeff_lat)
    return get_intersection_point(PIXEL_UL, point_lon, point_lat)

def draw_dot(pixelx, pixely, color):
    ''' Рисует точку на изображении.
    '''
    cv2.circle(img,(int(pixelx),int(pixely)), 10, color, -1)
    cv2.circle(img,(int(pixelx),int(pixely)), 200, color, 10)

def draw_line(pixelA, pixelB, color):
    ''' Рисует линиию на изображении.
    '''
    cv2.line(img, (int(pixelA[0]),int(pixelA[1])), (int(pixelB[0]),int(pixelB[1])), color, 7)

def draw_coord_line(coordA, coordB, color):
    ''' Рисует линию по координатам широты и долготы.
    '''
    pixA = get_pix_from_coord(coordA[0], coordA[1])
    pixB = get_pix_from_coord(coordB[0], coordB[1])
    draw_line((pixA[0], pixA[1]), (pixB[0], pixB[1]), color)

def draw_bbox(minlon, minlat, maxlon, maxlat, color):
    ''' Рисует прямоугольник по координатам широты и долготы.
    '''
    draw_coord_line((minlon,minlat),(maxlon,minlat),color)
    draw_coord_line((minlon,minlat),(minlon,maxlat),color)
    draw_coord_line((maxlon,minlat),(maxlon,maxlat),color)
    draw_coord_line((minlon,maxlat),(maxlon,maxlat),color)

def fill_bbox(mat, minlon, minlat, maxlon, maxlat, color):
    ''' Заполняет прямоугольник цветом.
    '''
    coordA = get_pix_from_coord(minlon,minlat)
    coordB = get_pix_from_coord(maxlon,minlat)
    coordC = get_pix_from_coord(maxlon,maxlat)
    coordD = get_pix_from_coord(minlon,maxlat)

    pointA = np.array([coordA[0], coordA[1]])
    pointB = np.array([coordB[0], coordB[1]])
    pointC = np.array([coordC[0], coordC[1]])
    pointD = np.array([coordD[0], coordD[1]])

    contours = np.array( [ pointA, pointB, pointC, pointD ], dtype=np.int32 )
    cv2.fillPoly(mat, pts=[contours], color=color)

def cut_bbox(mat, minlon, minlat, maxlon, maxlat):
    ''' Вырезает прямоугольный фрагмент. Возвращает новую каринку.
    '''
    mask = np.zeros([mat.shape[0],mat.shape[1],mat.shape[2]],dtype=np.uint8)
    fill_bbox(mask, minlon, minlat, maxlon, maxlat, (1,1,1))
    result = np.array(mat * (mask.astype(mat.dtype)))
    coords = np.argwhere(mask)
    x_min = coords.min(axis=0)
    y_min = coords.min(axis=0)
    x_max = coords.max(axis=0)
    y_max = coords.max(axis=0)
    crop_result = result[x_min[0]:x_max[0]+1, y_min[1]:y_max[1]+1].copy()
    return crop_result;


def get_ndvi(nir_mat, red_mat): 
    nir_mat_float = nir_mat.astype(float)
    red_mat_float = red_mat.astype(float)
    sub_mat = np.subtract(nir_mat_float, red_mat_float)
    sum_mat = np.add(nir_mat_float, red_mat_float)
    result = np.divide(sub_mat,sum_mat)
    return result;

'''
def get_ndvi(nir_mat, red_mat): 
    nir_radiance_mat = np.add(np.multiply(nir_mat.astype(float), B4_GAIN), B4_BIAS)
    red_radiance_mat = np.add(np.multiply(red_mat.astype(float), B3_GAIN), B3_BIAS)
    sun_evaluation = math.sin(math.radians(parse_arr(mtl_arr, 'SUN_ELEVATION')))
    nir_tos_radiance_mat = (nir_radiance_mat * math.pi * (EARTH_SUN_DISTANCE * EARTH_SUN_DISTANCE)) / (sun_evaluation * B4_SUN_RADIANCE)
    red_tos_radiance_mat = (red_radiance_mat * math.pi * (EARTH_SUN_DISTANCE * EARTH_SUN_DISTANCE)) / (sun_evaluation * B3_SUN_RADIANCE)
    sub_mat = np.subtract(nir_tos_radiance_mat, red_tos_radiance_mat)
    sum_mat = np.add(nir_tos_radiance_mat, red_tos_radiance_mat)
    result = np.divide(sub_mat,sum_mat)
    return result;
'''

def apply_gradient(grayscale_mat, gradient_mat):
    ''' Дает цвета пикселям в соответствии с картинкой градиента.
    '''
    return cv2.LUT((grayscale_mat*256).astype("uint8"), gradient_mat.astype("uint8"));

def get_point_between_two(point1, point2, coeff):
    ''' Возвращает точку на прямой между точками point1 и point2 на расстоянии coeff процентов от point1.
        (Пример: coeff = 0.5 - точка посередине между point1 и point2, coeff = 0.4 - чуть ближе к точке point 1 и т.д.)
    '''
    y = point1[1] + (point2[1] - point1[1]) * coeff
    x = -((point2[0] - point1[0]) * y + (point1[0] * point2[1] - point2[0] * point1[1])) / (point1[1] - point2[1])
    return (x, y)

def get_intersection_point(origin, pointX, pointY):
    ''' Находит пересечение координатных линий по правилу параллелограмма.
        Для прямоугольника просто находит координату точки (x,y) отложенную от центра в точке origin.
    '''
    return pointX[0] + pointY[0] - origin[0], pointX[1] + pointY[1] - origin[1]

def download_city_data(city_relation):
    ''' Скачивает данные о городе из overpass api. Изначально использовалось название города,
        но из-за неодназначности (в мире существует два города с названием "Ванкувер") изменил
        функцию на использование параметра "relation" из OpenStreetMap чтобы однозначно
        определять город.
    '''
    url = 'http://overpass-api.de/api/interpreter?data=rel({0});out geom;'.format(str(city_relation))
    open('city_data.xml', 'wb').write(requests.get(url, allow_redirects=True).content)
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
    #city = download_city_data(1852574) # https://www.openstreetmap.org/relation/1852574
    city = TARGET
    print(city)
    red_bbox = cut_bbox(red_img, city["minlon"], city["minlat"], city["maxlon"], city["maxlat"])
    nir_bbox = cut_bbox(nir_img, city["minlon"], city["minlat"], city["maxlon"], city["maxlat"])

    result = get_ndvi(nir_bbox, red_bbox) 
    result_01 = (result + 1.)/2. # переходим от диапазона [-1;1] к [0;1]
    result_gradient = apply_gradient(result_01, ndvi_gradient_img); # накладываем градиент

    show_image(result_gradient)