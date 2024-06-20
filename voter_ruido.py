import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nkx


def GRAPH_A_DICT(graph):
    #convertir un graph de networkx en un dictionary red con red[i] los vecinos del nodo i-esimo
    red={}
    n=np.shape(graph)[0]
    for i in range(0,n):
        gni=graph[i]
        it = gni.items()
        lista = list(it)
        numpyarray = np.array(lista)
        vec_i=np.zeros(np.shape(numpyarray)[0])
        for k in range(0,np.shape(numpyarray)[0]):
            vec_i[k]=np.int64(numpyarray[k][0])
        vec_i=np.asarray(vec_i, dtype=int)
        red.update({i:vec_i})
    return red

def INIT (L) :
    cfg = np.random.randint(2 , size = (L , L) ) #saca 0 y 1
    #reescribo la configuracion cambiando los 0 por -1
    for i in range(L):
        for j in range(L):
            if cfg[i,j]==0:
                cfg[i,j]=-1
    return cfg

def INIT_CN(n):
    cfg=np.random.randint(2,size=n)
    for i in range(n):
        if cfg[i]==0:
            cfg[i]=-1
    return cfg

def MOV_MONTE_CARLO_4vecinos(conf, temp): #hacer
    L = np.shape(conf)[0]
    for i in range(L):
        for j in range(L):
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)
            if np.random.rand() < temp:  # ver si cambia por el ruido
                conf[x, y] = (conf[x, y]) * -1
            else:
                suma = conf[(x + 1) % L, y] + conf[x, (y + 1) % L] + conf[(x - 1) % L, y] + conf[x, (y - 1) % L]
                num1s=(suma+4)/2
                #c sera el numero de vecinos con la misma opinion que conf[x,y]
                if conf[x,y]==1:
                    c=num1s
                else:
                    c=4-num1s
                if np.random.rand() < c / 4: #si todos tienen la misma, c=4, c/4=1 y siempre es este caso
                    conf[x, y] = conf[x,y]
                else:
                    conf[x, y] = conf[x,y]*-1
    return conf

def MOV_MONTE_CARLO_CN(conf, red, temp): #hacer
    n= np.shape(conf)[0]
    for i in range(n):
            x = np.random.randint(0, n)
            if np.random.rand() < temp:  #ver si cambia por el ruido
                conf[x] = (conf[x]) * -1
            else:
                suma=0
                nvecinos=np.shape(red[x])[0]
                for j in range(nvecinos):
                    suma=suma+conf[red[x][j]]
                num1s=(suma+nvecinos)/2
                #c sera el numero de vecinos con la misma opinion que conf[x,y]
                if conf[x]==1:
                    c=num1s
                else:
                    c=nvecinos-num1s
                if np.random.rand() < c / nvecinos: #si todos tienen la misma, c=4, c/4=1 y siempre es este caso
                    conf[x] = conf[x]
                else:
                    conf[x] = conf[x]*-1
    return conf

#iteramos L^2 veces aleatoriamente
def MOV_MONTE_CARLO_alltoall(conf, temp):
    L=np.shape(conf)[0]
    for i in range(L):
        for j in range(L):
            x=np.random.randint(0,L)
            y=np.random.randint(0,L)
            if np.random.rand() < temp:  # ver si cambia por el ruido
                conf[x, y] = (conf[x, y])*-1
            else:  # si no cambia por el ruido, hacemos lo demas
                # #caso interaccion local
                # si hay N 1's y L^2-N -1's, entonces la suma de todos será igual a 2N-L^2
                num1s = (L**2+sum(sum(conf)))/2
                #c será la fraccion de de voters con la misma opinion que conf[x,y]
                if conf[x,y]==1:
                    c=num1s/L/L
                else:
                    c=(L**2-num1s)/L/L
                if np.random.rand() < c: #OJO cambiar esto a c si estamos haciendo all-to-all o c/4 si local
                    conf[x, y] = conf[x,y]
                else:
                    conf[x, y] = conf[x,y]*-1
    return conf

def CALC_MAGNETIZACION(conf):
    mag=np.sum(sum(conf))
    return mag #magnetizacion absoluta, no está fraccionada

def CALC_MAGNETIZACION_CN(conf):
    mag=np.sum(conf)
    return mag #magnetizacion absoluta, no está fraccionada

def GRAFICO_ALLTOALL(Larray,nt,mcSteps,eqSteps):
    magplot = plt.subplot()
    magplot.set(xlabel='Temperatura', ylabel='Magnetización')
    for m in range(np.shape(Larray)[0]):
        L=Larray[m]
        n2=1/(mcSteps**2*L*L)
        M = np.zeros(nt)
        M2a = np.zeros(nt)
        n1=1/(mcSteps*L*L)
        f.write('cambio de L'+'\n')
        config = INIT(L)
        for i in range(eqSteps):
            MOV_MONTE_CARLO_alltoall(config, 0)  # equilibrar
        for tt in range(nt):
            M1=0
            M2=np.int64(0)
            M4=np.int64(0)
            temp=temparr[tt]
            f.write(str(temp)+'\n')

            for i in range(mcSteps):
                MOV_MONTE_CARLO_alltoall(config, temp)
                Mag = CALC_MAGNETIZACION(config)
                M1=M1+abs(Mag)
                M2=M2+Mag**2
            M[tt] = n1*M1
            M2a[tt]=n2*M2
            print(tt)
            f.write(str(M[tt]) + "    " + str(M2a[tt]) + "\n")
            #se normaliza M con n1, que contiene 1/mcsteps/L^2. El L^2 por la magnetizacion normalizada al tamaño del sistema y el mcsteps por hacer la media.

        magplot.scatter(temparr, M, s=10, label='L =%s' % L)
    magplot.legend()
    plt.show(block=True)

def GRAFICO_4vecinos(Larray,nt,mcSteps,eqSteps):
    magplot = plt.subplot()
    magplot.set(xlabel='Temperatura', ylabel='Magnetización')
    for m in range(np.shape(Larray)[0]):
        L=Larray[m]
        M = np.zeros(nt)
        n1=1/(mcSteps*L*L)
        n2=1/(mcSteps**2*L*L)
        f.write('cambio de L'+'\n')
        config = INIT(L)

        for i in range(eqSteps):
            MOV_MONTE_CARLO_4vecinos(config, 0)  # equilibrar

        for tt in range(nt):
            M1=0
            M2=np.int64(0)

            temp=temparr[tt]
            f.write(str(temp)+'\n')

            for i in range(mcSteps):
                MOV_MONTE_CARLO_4vecinos(config, temp)
                Mag = CALC_MAGNETIZACION_CN(config)
                M1=M1+abs(Mag)
                M2=M2+Mag**2
            M[tt] = n1*M1
            M2[tt] = n2*M2
            print(tt)
            f.write(str(M[tt]) + "     "+str(M2[tt]) + "     "+str(M4[tt])+"\n")

            #se normaliza M con n1, que contiene 1/mcsteps/L^2. El L^2 por la magnetizacion normalizada al tamaño del sistema y el mcsteps por hacer la media.

        magplot.scatter(temparr, M, s=10, label='L =%s' % L)
    magplot.legend()
    plt.show(block=True)


def GRAFICO_CN(narray,p,nt,mcSteps,eqSteps):
    for m in range(np.shape(narray)[0]):
        n=narray[m]
        M = np.zeros(nt)
        n1=1/(mcSteps*n)
        graphbar = nkx.barabasi_albert_graph(n, p)  # red barabasi albert
        # n nodos y p edges que añadir en cada paso.
        red = GRAPH_A_DICT(graphbar)
        f.write('cambio de n a '+ str(n)+'\n')
        config = INIT_CN(n)
        for i in range(eqSteps):
            MOV_MONTE_CARLO_CN(config, red, 0)  # equilibrar

        for tt in range(nt):
            M1=0
            temp=temparr[tt]
            f.write('Temperatura     '+str(temp)+'\n')
            for i in range(mcSteps):
                MOV_MONTE_CARLO_CN(config, red, temp)
                Mag = CALC_MAGNETIZACION(config)
                M1=M1+abs(Mag)
            M[tt] = n1*M1
            print(M[tt])
            f.write(str(M[tt]) + "\n")

            #se normaliza M con n1, que contiene 1/mcsteps/L^2. El L^2 por la magnetizacion normalizada al tamaño del sistema y el mcsteps por hacer la media.

#codigo para ejecutar la simulacion del modelo del votante con ruido en una red compleja libre de escala
eqSteps = 2000
mcSteps = 3000
narray=[1000,3000,6000,10000,30000,60000]
m=3
startime=time.time()
f = open("voter_cn_2006.txt", "a")
f.write("Now the file has more content!")
p=3
nt=30
temparr = np.linspace(0,0.02,nt)
GRAFICO_CN(narray,p,nt,mcSteps,eqSteps)
endtime=time.time()
print(endtime-startime)
f.write('se acaba')
f.write(str(endtime-startime))
f.close()
