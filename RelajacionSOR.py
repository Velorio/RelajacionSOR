import math
import numpy as np
from fractions import Fraction
from numpy.linalg import eig


def pedir_coeficientes(n):
    while True:
        try:
            coeficientes = input("Ingrese los coeficientes separados por espacios: ").split()
            coeficientes = [Fraction(coef) for coef in coeficientes]
            if len(coeficientes) != n:
                raise ValueError("Debe ingresar exactamente {} coeficientes.".format(n))
            break
        except ValueError as e:
            print(e)
            print("Inténtelo de nuevo.")
    return coeficientes

def definir_sistema():
    n = 0
    while True:
        try:
            n = int(input("Ingrese el tamaño del sistema (n): "))
            if n <= 0:
                raise ValueError("El tamaño del sistema debe ser un número entero positivo.")
            break
        except ValueError:
            print("Ingrese un número entero válido.")

    A = []
    b = []
    for i in range(n):
        print("Ingrese los coeficientes de la fila", i + 1)
        coeficientes = pedir_coeficientes(n)
        A.append(coeficientes)
        while True:
            try:
                termino_independiente = Fraction(input("Ingrese el término independiente b[{}]: ".format(i + 1)))
                break
            except ValueError:
                print("Ingrese un valor válido para el término independiente.")
        b.append(termino_independiente)

    return np.array(A), np.array(b)

def imprimir_matriz(matriz):
    for fila in matriz:
        imprimir_vector(fila)

def imprimir_vector(vector):
    for elemento in vector:
        elemento = Fraction(elemento)
        if elemento.denominator == 1:
            print(elemento.numerator, end=" ")
        else:
            if elemento.numerator < 0:
                print("-{}/{} ".format(abs(elemento.numerator), elemento.denominator), end="")
            else:
                print("{}/{} ".format(elemento.numerator, elemento.denominator), end="")
    print()

def solicitar_tolerancia():
    try:
        print("La tolerancia es un valor 1*10^-n")
        n = int(input("Ingrese el valor de n para la tolerancia (debe ser un entero): "))
        if n < 0:
            raise ValueError("El valor de n debe ser un entero positivo o cero.")
        
        tolerancia = 10 ** -n
        print("Valor numérico de la tolerancia:", tolerancia)
        return tolerancia
    except ValueError as e:
        print("Error:", e)

def norma_euclidiana(vector1,vector2):
    c = [ai - bi for ai, bi in zip(vector1, vector2)]
    diferencia = np.array(c) ##c = Vector1-Vector2
    sumacuadrada = Fraction(0)
    for elemento in diferencia:
        sumacuadrada += (elemento**2)##sumatoria de ci^2
    return math.sqrt(float(sumacuadrada))##raiz de la sumatoria

def gauss_seidel(A,b,k,x_previo,tolerancia):##A:=matriz; b:=terminos independientes; k:=iteración actual; x_previo:=vector x en iteración k-1
    n=len(A)

    if k == 1:
        print("X^(0) : ")
        imprimir_vector(np.array(x_previo))

    x=x_previo[:]##crea el vector x para la k-esima iteración
    print("Iteracion: ",k)
    print()
    for i in range(n):
        aii = A[i][i] ## Selecciona el valor del denominador A_ii
        xi=Fraction(0)
        for j in range(n):
            xi -= (Fraction(A[i][j], aii) * x[j]) if j != i else 0 ## (Aij/Aii)*xj 
        xi += Fraction(b[i],aii)
        x[i]=xi ##Agrega xi a x // en las siguientes filas se opera con el nuevo valor para x[i]
    
    print(f"X^({k}) : ")
    imprimir_vector(np.array(x))
    print()

    n_e = norma_euclidiana(np.array(x),np.array(x_previo))

    if n_e < tolerancia:
        print("El sistema converge en la iteración ",k)
        print(f"La norma euclidiana es {n_e} que es menor que la tolerancia: {tolerancia}")
        return x
    else:
        return gauss_seidel(A,b,k+1,x,tolerancia)

def imprimir_solucion(x):
    for i, xi in enumerate(x, start=1):
        print("X{} = {}".format(i, float(xi)))

def obtenerTj(A):
    # Inicializar una lista vacía para almacenar la matriz Tj
    Tj = []
    # Iterar sobre las filas de la matriz A
    for i in range(len(A)):
        # Inicializar una lista vacía para almacenar la fila resultante de Tj
        fila_resultante = []
        # Obtener el elemento diagonal de la fila actual de A
        aii = A[i][i]
        # Iterar sobre las columnas de la matriz A
        for j in range(len(A)):
            # Calcular el elemento de la fila resultante de Tj
            f_ij = float(((-1)*A[i][j])/aii) if j != i else 0
            # Agregar el elemento calculado a la fila resultante de Tj
            fila_resultante.append(f_ij)
        # Agregar la fila resultante de Tj a la matriz Tj
        Tj.append(fila_resultante)
    return(np.array(Tj))

def relajacionSOR(A,b,k,x_previo,tolerancia,w): ## A: matriz; b:t_independientes; k:iteración; x_previo:vector incognitas; w:factor de relajación
    n=len(A)

    if k == 1:
        print("X^(0) : ")
        imprimir_vector(np.array(x_previo))
        print()

    x=x_previo[:]##crea el vector x para la k-esima iteración
    print("Iteracion: ",k)
    print()
    for i in range(n):
        aii = A[i][i] ## Selecciona el valor del denominador A_ii
        xi=Fraction(0)
        for j in range(n):
            xi -= (Fraction(A[i][j], aii) * x[j]) if j != i else 0 ## (Aij/Aii)*xj 
        xi += Fraction(b[i],aii)
        xi = ( w * xi ) + ((1-w)*x_previo[i])
        x[i]=xi ##Agrega xi a x // en las siguientes filas se opera con el nuevo valor para x[i]
    
    print(f"X^({k}) : ")
    imprimir_vector(np.array(x))
    print()

    n_e = norma_euclidiana(np.array(x),np.array(x_previo))

    if n_e < tolerancia:
        print("El sistema converge en la iteración ",k)
        print(f"La norma euclidiana es {n_e} que es menor que la tolerancia: {tolerancia}")
        return x
    else:
        return relajacionSOR(A,b,k+1,x,tolerancia,w)


def factorDeRelajacion(A):
    # Obtener la matriz Tj
    Tj = obtenerTj(A)
    # Obtener los eigenvalores de Tj
    eigen_valores, eigen_vectores = eig(np.array(Tj))
    # Calcular el módulo de cada número complejo
    modulos = np.abs(eigen_valores)
    # Obtener el mayor módulo
    radio_espectral = max(modulos)
    # Obtener el factor de relajacion
    factor_de_relajacion = 2 / ( 1 + math.sqrt(1-( radio_espectral ** 2 ) ) )

    return factor_de_relajacion

if __name__ == '__main__':

    A, b = definir_sistema()
    n = len(A)

    print("tamaño sistema: ",n)
    
    print("Matriz A:")
    imprimir_matriz(A)
    print("Vector b:")
    imprimir_vector(b)

    print("Factor de relajación:")
    factor_relajacion = factorDeRelajacion(A)
    print(factor_relajacion)

    tolerancia = solicitar_tolerancia()
    solucion = relajacionSOR(A,b,1,[Fraction(0) for _ in range(n)],tolerancia,factor_relajacion)
    imprimir_solucion(solucion)

    finalizar = input("FIN")