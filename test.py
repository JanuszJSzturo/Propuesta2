import math

import numpy as np
from scipy.optimize import least_squares


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
    punto_transformado = (x, y, z)
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


def main():

    #####
    # Representación de una traslación 3D
    punto = (4, 5, 6)
    vector_traslacion = (0, 0, -2)

    print(f'{punto} trasladado según {vector_traslacion} = {traslacion(punto, vector_traslacion)}.')

    #####
    # Representación de una rotación axial.

    #   Multiplicación de cuaterniones
    cos, sin = math.cos(math.radians(45)/2), math.sin(math.radians(45)/2)
    punto = (3, 3, 5)
    p = (0, 3, 3, 5)        # cuaternión correspondiente al punto (x, y, z) = (3, 3, 5)
    q = (cos, 0, 0, sin)    # cuaternión correspondiente al eje axial de 45º de eje Z
    q_conjugado = cuaternion_conjugado(q)

    print(f'{p} * {q} = {multiplicar_quaterniones(p, q)}.')
    print(f'{q} * {p} * {q_conjugado} = {multiplicar_quaterniones(q, multiplicar_quaterniones(p, q_conjugado))}.')

    #   Aplicamos al punto la rotación axial de 45º de eje Z
    print(f'{p} tras rotacion axial según q = {q}: {rotacion_axial(punto, math.radians(45), (0, 0, 1) )}.')

    # Corregistro mediante landmarks:
    #   Supongamos que en dos imágenes, referencia e input, hemos manualmente localizado puntos de interés.
    #   Vamos a buscar el ajuste que mejor envía los puntos de interés de una a la otra.
    landmarks_ref = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    landmarks_inp = [(0.05, 3.22, 0.94),
                     (0.59, 3.56, 1.71),
                     (0.39, 3.45, 0.37),
                     (-0.52, 3.79, 1.23),
                     (0.16, 4.88, 1.22) ]

    #   Calculamos el error como la suma de los errores cuadráticos medios
    residuos = residuos_cuadraticos(landmarks_ref, landmarks_inp)
    error_cuadratico_medio = sum(residuos)/len(residuos)
    print(f'Residuos cuadráticos: {residuos}')
    print(f'Error cuadrático medio antes de la transformación: {error_cuadratico_medio}')

    #   Podemos encontrar una buena inicialización centrando las dos nubes de puntos en el mismo centroide.
    parametros_iniciales = [0, 0, 0,
                            0, 1, 0, 0,     # Inicializamos el eje de la rotacion a un vector unitario
                            0, 0, 0]
    for i in range(3):
        centroide_ref = sum([punto[i] for punto in landmarks_ref])/len(landmarks_ref)
        centroide_inp = sum([punto[i] for punto in landmarks_inp])/len(landmarks_inp)
        parametros_iniciales[i] = centroide_ref - centroide_inp     # ¿Por qué esta diferencia?

    landmarks_inp_transf = [transformacion_rigida_3D(l, parametros_iniciales) for l in landmarks_inp]
    residuos = residuos_cuadraticos(landmarks_ref, landmarks_inp_transf)
    error_cuadratico_medio = sum(residuos) / len(residuos)
    print(f'Residuos cuadráticos: {residuos}')
    print(f'Error cuadrático medio tras inicializar los parámetros: {error_cuadratico_medio}')

    #####
    # Encontrar la mejor transformacion rígida
    #   Mediante minimizar una función objetivo mediante un algoritmo de resolución de mínimos cuadrados

    def funcion_a_minimizar(parametros):
        landmarks_inp_transf = [transformacion_rigida_3D(landmark, parametros) for landmark in landmarks_inp]
        # Debe devolver una array 1-dimensional con los errores cuadráticos medios.
        return residuos_cuadraticos(landmarks_ref, landmarks_inp_transf)

    resultado = least_squares(funcion_a_minimizar,
                              x0=parametros_iniciales,
                              verbose=1)
    x_opt = resultado.x
    print(f'''
    Los mejores parámetros son: 
        1) Traslación respecto al vector ({x_opt[0]}, {x_opt[1]}, {x_opt[2]}).
        2) Rotación axial de {math.degrees(x_opt[3])} grados, eje ({x_opt[4]}, {x_opt[5]}, {x_opt[6]}).
        3) Traslación respecto al vector ({x_opt[7]}, {x_opt[8]}, {x_opt[9]}).
    ''')

    #   Modifica la función para comprobar si los parámetros son redundantes: ¿que implicaría que podamos ajustar bien
    #   los puntos de interés incluso sin hacer alguna de las dos rotaciones?

    def funcion_a_minimizar_v2(parametros_v2):
        parametros_v2 = np.concatenate([parametros_v2, [0, 0, 0]])
        landmarks_inp_transf = [transformacion_rigida_3D(landmark, parametros_v2) for landmark in landmarks_inp]
        # Debe devolver una array 1-dimensional con los errores cuadráticos medios.
        return residuos_cuadraticos(landmarks_ref, landmarks_inp_transf)

    resultado = least_squares(funcion_a_minimizar_v2,
                              x0=parametros_iniciales[0:7],
                              verbose=1)
    x_opt = resultado.x
    print(f'''
    Los mejores parámetros son: 
        1) Traslación respecto al vector ({x_opt[0]}, {x_opt[1]}, {x_opt[2]}).
        2) Rotación axial de {math.degrees(x_opt[3])} grados, eje ({x_opt[4]}, {x_opt[5]}, {x_opt[6]}).
        3) Traslación respecto al vector: la hemos eliminado!.
    ''')


if __name__ == '__main__':
    main()