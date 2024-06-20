from __future__ import division
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import networkx as nkx
import time

def ESTADO_INICIAL(L):
    state = 2*np.random.randint(2, size=(L,L))-1
    return state

def MOV_MONTE_CARLO(conf, beta):
    L=np.shape(conf)[0]
    for i in range(L):
        for j in range(L):
            x=np.random.randint(0,L)
            y=np.random.randint(0,L)
            spin=conf[x,y]
            neighspin=conf[(x+1)%L,y]+conf[(x-1)%L,y]+conf[x,(y-1)%L]+conf[x,(y+1)%L]
            changeE = 2*spin*neighspin
            if changeE<0: #si se baja energia, se permite el cambio
                spin*=-1
            elif rand()<np.exp(-changeE*beta): #si se aumenta energia, se permite el cambio aleatoriamente
                spin*=-1
            conf[x,y]=spin
    return conf

def MOV_MONTE_CARLO_COMPLETO(conf, beta):
    L=np.shape(conf)[0]
    for i in range(L):
        for j in range(L):
            x=np.random.randint(0,L)
            y=np.random.randint(0,L)
            spin=conf[x,y]
            neighspin=CALC_ABSOLUTE_MAGNETIZACION(conf)/L/L
            changeE = 2*spin*neighspin
            if changeE<0: #si se baja energia, se permite el cambio
                spin*=-1
            elif rand()<np.exp(-changeE*beta): #si se aumenta energia, se permite el cambio aleatoriamente
                spin*=-1
            conf[x,y]=spin
    return conf

def MOV_MONTE_CARLO_CN(conf,red,beta):
    #conf es array con todos los agentes y sus valores d espin
    #red es array con todos los agentes y sus vecinos
    #se podria hacer tb una red con vecinos y fuerza de enlace por así decirlo, influencia
    N = np.shape(conf)[0]
    for i in range(N):
        x = np.random.randint(0, N)
        spin = conf[x, y]

        neighspin = 0
        for k in range(red[x]):
            neighspin=neighspin+conf[red[k]]

        changeE = 2 * spin * neighspin
        if changeE < 0:  # si se baja energia, se permite el cambio
            spin *= -1
        elif rand() < np.exp(-changeE * beta):  # si se aumenta energia, se permite el cambio aleatoriamente
            spin *= -1
        conf[x, y] = spin
    return conf


def CALC_ABSOLUTE_MAGNETIZACION(conf):
    mag=np.sum(np.sum(conf))
    return mag

def CALC_ENERGIA(config):
    energia = 0
    L=np.shape(config)[0]
    for i in range(len(config)):
        for j in range(len(config)):
            spin = config[i,j]
            nb = config[(i+1)%L, j] + config[i,(j+1)%L] + config[(i-1)%L, j] + config[i,(j-1)%L]
            energia += -nb*spin
    return energia/4. #el sumatorio que hacemos es sobre
    #todo ij y sus 4 vecinos, eso multiplica por 4 el sumatorio


nt = 40 #puntos de temperatura
Larray = [10,20,30,40,60,80] #lado de la configuracion
eqSteps=2000 #era 1024 #numero de movimientos MC para equlibracion (asegurarse de que se ha iterado lo suficiente en una configuracion
#para que haya un equilibrio?? y no se llegue a un resultado intermedio
mcSteps=2300 #era 1024 #numero de movimientos MC para calculo (cuantas iteraciones se quiere, de las que hacer luego la media)
T = np.linspace(0.4,1.6,nt)

#establecer los plots que luego rellenaremos
fig, (magplot, susplot) = plt.subplots(2)
magplot.set(xlabel='Temperatura', ylabel='Magnetización')
susplot.set(xlabel='Temperatura', ylabel='Susceptibilidad')

startime=time.time()
f = open("archivo_isingtodos.txt", "a")

for m in range(np.shape(Larray)[0]):
    L=Larray[m]
    #E,M = np.zeros(nt)
    M = np.zeros(nt, dtype=np.float64)
    X = np.zeros(nt, dtype=np.float64)
    n1,n2=1/(mcSteps*L*L), 1/(mcSteps*mcSteps*L*L)
    print(m)
    f.write('cambio de L' + '\n')
    config = ESTADO_INICIAL(L)
    beta=1/T[1]
    for i in range(eqSteps):
        MOV_MONTE_CARLO_COMPLETO(config, beta)  # equilibrar
    for tt in range(nt):
        M1=M2=np.int64(0)
        #E1=E2=0
        beta=1/T[tt] #beta, iT en el codigo
        #iT2=iT*iT
        f.write('Temperatura     ' + str(T[tt]) + '\n')
        for i in range(mcSteps):
            MOV_MONTE_CARLO_COMPLETO(config, beta)
            Mag = abs(CALC_ABSOLUTE_MAGNETIZACION(config))
            M1=M1+Mag
            M2=M2+Mag**2
        M[tt] = n1*M1
        X[tt] = (n1*M2-n2*M1*M1)*beta
        print(M[tt])
        f.write(str(M[tt]) + "     " + str(X[tt])+ "\n")

    magplot.scatter(T,abs(M),s=10, label='L =%s' %L)
    susplot.scatter(T,X,s=10, label='L =%s' %L)

endtime=time.time()
print(endtime-startime)
f.write('se acaba')
f.write(str(endtime-startime))
f.close()

magplot.legend()
susplot.legend()
plt.show(block=True)
