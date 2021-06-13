
import glob
import pydicom

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import numpy as np
from skimage import measure
import math

import numpy as np
from scipy.optimize import least_squares
from scipy import ndimage


def traslacion(punto, vector_traslacion):
    x, y, z = punto
    t_1, t_2, t_3 = vector_traslacion
    punto_transformado = (x+t_1, y+t_2, z+t_3)
    return punto_transformado


def rotacion_axial(punto, angulo_en_radianes, eje_traslacion):
    x, y, z = punto
    v_1, v_2, v_3 = eje_traslacion
    #   Vamos a normalizarlo para evitar introducir restricciones en el optimizador
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v_1, v_2, v_3]]))
    v_1, v_2, v_3 = v_1 / v_norm, v_2 / v_norm, v_3 / v_norm
    #   Calcula cuaternión del punto
    p = (0, x, y, z)
    #   Calcula cuaternión de la rotación
    cos, sin = math.cos(angulo_en_radianes / 2), math.sin(angulo_en_radianes / 2)
    q = (cos, sin * v_1, sin * v_2, sin * v_3)
    #   Calcula el conjugado
    q_conjugado = (cos, -sin * v_1, -sin * v_2, -sin * v_3)
    #   Calcula el cuaternión correspondiente al punto rotado
    p_prima = multiplicar_quaterniones(q, multiplicar_quaterniones(p, q_conjugado))
    # Devuelve el punto rotado
    punto_transformado = p_prima[1], p_prima[2], p_prima[3]
    return punto_transformado


def transformacion_rigida_3D(punto, parametros):
    x, y, z = punto
    t_11, t_12, t_13, alpha_in_rad, v_1, v_2, v_3, t_21, t_22, t_23 = parametros
    #   Aplicar una primera traslación
    x, y, z = traslacion(punto=(x, y, z), vector_traslacion=(t_11, t_12, t_13))
    #   Aplicar una rotación axial traslación
    x, y, z = rotacion_axial(punto=(x, y, z), angulo_en_radianes=alpha_in_rad, eje_traslacion=(v_1, v_2, v_3))
    #   Aplicar una segunda traslación
    x, y, z = traslacion(punto=(x, y, z), vector_traslacion=(t_21, t_22, t_23))
    punto_transformado = np.array((x, y, z))
    return punto_transformado


def multiplicar_quaterniones(q1, q2):
    """Multiplica cuaterniones expresados como (1, i, j, k)."""
    return (
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    )


def cuaternion_conjugado(q):
    """Conjuga un cuaternión expresado como (1, i, j, k)."""
    return (
        q[0], -q[1], -q[2], -q[3]
    )


def residuos_cuadraticos(lista_puntos_ref, lista_puntos_inp):
    """Devuelve un array con los residuos cuadráticos del ajuste."""
    residuos = []
    for p1, p2 in zip(lista_puntos_ref, lista_puntos_inp):
        p1 = np.asarray(p1, dtype='float')
        p2 = np.asarray(p2, dtype='float')
        residuos.append(np.sqrt(np.sum(np.power(p1-p2, 2))))
    residuos_cuadraticos = np.power(residuos, 2)
    return residuos_cuadraticos

def load_dcm(filename):
    return pydicom.dcmread(filename)


def load_dcm_from_folder(path):
    dcm_files = []
    instance_number = []
    dcm_images = []
    for file in glob.glob(path + '/*.dcm'):
        dcm_file = pydicom.dcmread(file)

        dcm_files.append(dcm_file)
        instance_number.append(dcm_file.InstanceNumber)
        dcm_images.append(dcm_file.pixel_array)

    if len(dcm_files) == 1:
        dcm_images = dcm_file.pixel_array
    elif len(dcm_files) > 1:
        sorted_index = np.argsort(np.array(instance_number))
        dcm_images = np.array(dcm_images)[sorted_index]

    return dcm_files, dcm_images





# dcm_AAL3, image_AAL3 = load_dcm_from_folder('P2 - DICOM/AAL3_1mm')
# dcm_icbm, image_icbm = load_dcm_from_folder('P2 - DICOM/icbm_avg')
# dcm_RM_Brain, image_RM_Brain = load_dcm_from_folder('P2 - DICOM/RM_Brain_3D-SPGR')
# print(image_AAL3.shape)
# print(image_icbm.shape)
# print(image_RM_Brain.shape)
#
# image_RM_Brain_0 = ndimage.zoom(image_RM_Brain, (1/2, 1/0.5078, 1/0.5078))
# image_RM_Brain_0 = ndimage.zoom(image_RM_Brain, (np.array(image_icbm.shape)/np.array(image_RM_Brain.shape)))
#
#
# print(image_icbm.shape)
# print(image_RM_Brain_0.shape)
# img = dcm.pixel_array
# img2 = img[0,:,:]
# plt.imshow(img[:,:,100], cmap=plt.cm.bone)
# plt.show()
# print(dcm)
def my_mse (img1, img2):
    diff = img2-img1
    return np.linalg.norm(diff)

#FLECHA
flecha = np.zeros((2, 10, 7))
flecha[:, 1, 3] = 1
flecha[:, 2, 2:5] = 1
flecha[:, 3, 1:6] = 1
flecha[:, 1:9, 3] = 1

plt.imshow(flecha[1, :, :])
plt.show()

# parametros_iniciales = [0, -10, -7,
#                         np.pi, 1, 0, 0,  # Inicializamos el eje de la rotacion a un vector unitario
#                         0, 0, 0]
parametros_iniciales = [0, 0, 0,
                        np.pi, 1, 0, 0,  # Inicializamos el eje de la rotacion a un vector unitario
                        0, 10, 7]

# d, h, w = flecha.shape
#
# for z in range(d):
#     for y in range(h):
#         for x in range(w):
#             z_transformed, y_transformed, x_transformed =

def position_list(matrix):
    d, h, w = matrix.shape
    positions = []
    for z in range(d):
        for y in range(h):
            for x in range(w):
                positions.append(np.array((z, y, x)))
    return positions
# Posiciones iniciales de lo píxeles
init_pos = position_list(flecha)

flecha_transformada_pos = [transformacion_rigida_3D(landmark, parametros_iniciales) for landmark in init_pos]

flecha_transformada_pos = np.array(flecha_transformada_pos).astype(int)
flecha_transformada_pos = np.round(flecha_transformada_pos)
flecha_transformada = np.zeros(flecha.shape)
print(flecha_transformada.shape)
d, h, w = flecha_transformada.shape

for i, pixel_pos in enumerate(flecha_transformada_pos):
    # print (flecha_transformada.shape)
    # print(pixel_pos.astype(int))
    new_d, new_h, new_w = pixel_pos
    if new_d >= 0 and new_d < d and new_h >= 0 and new_h < h and new_w >= 0 and new_w < w:
        index = init_pos[i]
        value_gray = flecha[tuple(index)]
        flecha_transformada[tuple(pixel_pos)] = value_gray


plt.imshow(flecha_transformada[1, :, :])
plt.show()