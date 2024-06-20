import numpy as np
import matplotlib.pyplot as plt
import networkx as nkx

def INIT (L) :
    cfg = np.random.randint(2 , size = (L , L) ) #saca 0 y 1
    #reescribo la configuracion cambiando los 0 por -1
    for i in range(L):
        for j in range(L):
            if cfg[i,j]==0:
                cfg[i,j]=-1
    return cfg

def ESTADO_INICIAL(n):
    state = 2*np.random.randint(2, size=n)-1
    return state

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

def MOV_MONTE_CARLO_alltoall_edad(conf, a, edad):
    L=np.shape(conf)[0]
    for i in range(L):
        for j in range(L): #un mov de monte carlo son N=LxL sweeps
            x=np.random.randint(0,L)
            y=np.random.randint(0,L)
            if np.random.rand() < a:  # ver si cambia por el ruido
                conf[x, y] = (conf[x, y])*-1
                edad[x,y]=0 #actualizo a 0 porque ha cambiado opinion
            else:  # si no cambia por el ruido (1-a), hacemos lo demas
                # si hay N 1's y L^2-N -1's, entonces la suma de todos será igual a 2N-L^2
                num1s = (L**2+sum(sum(conf)))/2
                #c será la fraccion de de voters con la misma opinion que conf[x,y]
                if conf[x,y]==1:
                    c=num1s/L/L
                else:
                    c=(L**2-num1s)/L/L
                if np.random.rand() < 1/(2+edad[x,y]): #se supera la edad, se hace el herding
                    if np.random.rand() < c: #OJO cambiar esto a c si estamos haciendo all-to-all o c/4 si local
                        conf[x, y] = conf[x,y]
                        edad[x,y]=edad[x,y]+1 #se ha copiado la misma opinion, no cambia
                    else:
                        conf[x, y] = conf[x,y]*-1
                        edad[x,y] = 0 #se ha copiado la opinion contraria, edad cero
                else: #no se ha superado la edad, no se cambia, sumo edad
                    edad[x,y]=edad[x,y]+1
    return conf
def MOV_MONTE_CARLO_4vecinos_edad(conf, a, edad):
    L=np.shape(conf)[0]
    for i in range(L):
        for j in range(L): #un mov de monte carlo son N=LxL sweeps
            x=np.random.randint(0,L)
            y=np.random.randint(0,L)
            if np.random.rand() < a:  # ver si cambia por el ruido
                conf[x, y] = (conf[x, y])*-1
                edad[x,y]=0 #actualizo a 0 porque ha cambiado opinion
            else:  # si no cambia por el ruido (1-a), hacemos lo demas
                # si hay N 1's y L^2-N -1's, entonces la suma de todos será igual a 2N-L^2
                suma = conf[(x + 1) % L, y] + conf[x, (y + 1) % L] + conf[(x - 1) % L, y] + conf[x, (y - 1) % L]
                num1s=(suma+4)/2
                #c sera el numero de vecinos con la misma opinion que conf[x,y]
                if conf[x,y]==1:
                    c=num1s
                else:
                    c=4-num1s
                if np.random.rand() < 1/(2+edad[x,y]): #se supera la edad, se hace el herding
                    if np.random.rand() < c/4: #OJO cambiar esto a c si estamos haciendo all-to-all o c/4 si local
                        conf[x, y] = conf[x,y]
                        edad[x,y]=edad[x,y]+1 #se ha copiado la misma opinion, no cambia
                    else:
                        conf[x, y] = conf[x,y]*-1
                        edad[x,y] = 0 #se ha copiado la opinion contraria, edad cero
                else: #no se ha superado la edad, no se cambia, sumo edad
                    edad[x,y]=edad[x,y]+1
    return conf


def Mprueba(conf, red, temp, edad): #hacer
    n= np.shape(conf)[0]
    for i in range(n):
            x = np.random.randint(0, n)
            if np.random.rand() < temp:  # ver si cambia por el ruido
                conf[x] = (conf[x]) * -1
                edad[x] = 0 #edad==0 porque ha cambiado
            else:
                #veamos si supera la edad, ya que NO ha funcionado el ruido
                if np.random.rand() < 1/(2+edad[x]):
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
                        #conf[x] = conf[x] no se cambia
                        edad[x]=edad[x]+1
                    else:
                        conf[x] = conf[x]*-1
                        edad[x]=0 #se cambia
                else:
                    #no se ha superado edad, no cambia
                    #conf[x]=conf[x]
                    edad[x]=edad[x]+1
    return conf

def MOV_MONTE_CARLO_COMPLEJO_edad(conf, red, a, edad):
    N=np.shape(conf)[0]
    for i in range(N):#un mov de monte carlo son N sweeps
        x=np.random.randint(0,N)
        if np.random.rand() < a:  # ver si cambia por el ruido
            conf[x] = (conf[x])*-1
            edad[x]=0 #actualizo a 0 porque ha cambiado opinion
        else:  # si no cambia por el ruido (1-a), hacemos lo demas
            if np.random.rand() < 1: #se supera la edad, se hace el herding
                neighspin = 0
                numvec=np.shape(red[x])[0]
                for k in range(numvec):
                    neighspin = neighspin + conf[int(red[x][k])]
                num1s=(numvec+neighspin)/2
                if conf[n]==1:
                    c=num1s/numvec
                else:
                    c=(numvec-num1s)/numvec
                if np.random.rand() < c:
                    #conf[n]=conf[n]
                    edad[x]=edad[x]+1 #se ha copiado la misma opinion, no cambia
                else:
                    conf[x] = conf[x]*-1
                    edad[x] = 0 #se ha copiado la opinion contraria, edad cero
            else: #no se ha superado la edad, no se cambia, sumo edad
                edad[x]=edad[x]+1
    return conf

def CALC_MAGNETIZACION(conf):
    mag=np.sum(sum(conf))
    return mag #magnetizacion absoluta, no está fraccionada

def CALC_MAGNETIZACION_CN(conf):
    mag=np.sum(conf)
    return mag #magnetizacion absoluta, no está fraccionada


def GRAFICO_ALLTOALL_edad(Larray, nb, mcSteps, eqSteps):
    for m in range(np.shape(Larray)[0]):
        L = Larray[m]
        M = np.zeros(nb)
        M2t = np.zeros(nb)

        n1 = 1 / (mcSteps * L * L)
        n2 = 1 / (mcSteps ** 2 * L * L)
        f.write('cambio de L' + '\L')

        for bb in range(nb):
            M1 = 0
            M2 = np.int64(0)
            config = INIT(L)
            b = barr[bb]
            f.write('Parametro     ' + str(b) + '\n')
            # como lo estoy haciendo aqui, el paso de equilibrio NO tiene aging realmente
            for i in range(eqSteps):  # para un primer paso aleatorio
                MOV_MONTE_CARLO_alltoall(config, 0)  # equilibrar

            edad = np.zeros([L, L])
            for i in range(mcSteps):
                MOV_MONTE_CARLO_alltoall_edad(config, b, edad)  # lo voy a hacer con temp 0.5 primero
                Mag = CALC_MAGNETIZACION(config)
                M1 = M1 + abs(Mag)
                M2 = M2 + Mag**2
            M[bb] = n1 * M1
            M2t[bb] = n2 * M2
            print(M[bb])
            f.write(str(M[bb]) + "   " + str(M2t[bb]) + "\n")
            # se normaliza M con n1, que contiene 1/mcsteps/L^2. El L^2 por la magnetizacion normalizada al tamaño del sistema y el mcsteps por hacer la media.


def GRAFICO_COMPLEJO_edad(Narray,nb,mcSteps,eqSteps):
    for d in range(np.shape(Narray)[0]):
        N=Narray[d]
        M = np.zeros(nb)
        M2t = np.zeros(nb)
        n1=1/(mcSteps*N)
        n2=1/(mcSteps**2*N)
        f.write('cambio de L'+'\n')

        graphbar = nkx.barabasi_albert_graph(N, m)  # red barabasi albert
        # n nodos y m edges que añadir en cada paso. tengo q ver como cambiar los otros parametros
        red = GRAPH_A_DICT(graphbar)

        for bb in range(nb):
            edad = np.zeros(N) #reiniciamos la edad
            conf = ESTADO_INICIAL(N) #reiniciamos tb la configuracion, pq la edad no cambia continuamente como b??PREGUNTAR
            M1=0
            M2=np.int64(0)
            b=barr[bb]
            f.write('Parametro     '+str(b)+'\n')
            #como lo estoy haciendo aqui, el paso de equilibrio NO tiene aging realmente
            for i in range(eqSteps): #para un primer paso aleatorio
                Mprueba(conf,red, 0,edad) #equilibrar
            for i in range(mcSteps):
                Mprueba(conf,red, b,edad) #lo voy a hacer con temp 0.5 primero
                Mag = CALC_MAGNETIZACION_CN(conf)
                M1=M1+abs(Mag)
                M2=M2+Mag**2
            M[bb] = n1*M1
            M2t[bb]= n2*M2
            print(M[bb])
            f.write(str(M[bb]) + "   " + str(M2t[bb])+ "        " + "\n")




def GRAFICO_4vecinos(Larray,nt,mcSteps,eqSteps):
    magplot = plt.subplot()
    magplot.set(xlabel='Temperatura', ylabel='Magnetización')
    for m in range(np.shape(Larray)[0]):
        L=Larray[m]
        M = np.zeros(nt)
        n1=1/(mcSteps*L*L)

        for tt in range(nt):
            M1=0
            config=INIT(L)
            temp=temparr[tt]

            for i in range(eqSteps):
                MOV_MONTE_CARLO_4vecinos(config, 0) #equilibrar

            for i in range(mcSteps):
                MOV_MONTE_CARLO_4vecinos(config, temp)
                Mag = CALC_MAGNETIZACION(config)
                M1=M1+abs(Mag)
            M[tt] = n1*M1
            print(M[tt])
            #se normaliza M con n1, que contiene 1/mcsteps/L^2. El L^2 por la magnetizacion normalizada al tamaño del sistema y el mcsteps por hacer la media.

        magplot.scatter(barr, M, s=10, label='L =%s' % L)
    magplot.legend()
    plt.show(block=True)


def GRAFICO_4vecinos_edad(Larray, nb, mcSteps, eqSteps):
    for m in range(np.shape(Larray)[0]):
        L = Larray[m]
        M = np.zeros(nb)
        M2t = np.zeros(nb)

        n1 = 1 / (mcSteps * L * L)
        n2 = 1 / (mcSteps ** 2 * L * L)
        f.write('cambio de L' + '\L')

        for bb in range(nb):
            M1 = 0
            M2 = np.int64(0)
            config = INIT(L)
            b = barr[bb]
            f.write('Parametro     ' + str(b) + '\n')
            # como lo estoy haciendo aqui, el paso de equilibrio NO tiene aging realmente
            for i in range(eqSteps):  # para un primer paso aleatorio
                MOV_MONTE_CARLO_4vecinos(config, 0)  # equilibrar

            edad = np.zeros([L, L])
            for i in range(mcSteps):
                (MOV_MONTE_CARLO_4vecinos_edad(config, b, edad))  # lo voy a hacer con temp 0.5 primero
                Mag = CALC_MAGNETIZACION(config)
                M1 = M1 + abs(Mag)
                M2 = M2 + Mag**2
            M[bb] = n1 * M1
            M2t[bb] = n2 * M2
            print(M[bb])
            f.write(str(M[bb]) + "   " + str(M2t[bb]) + "\n")
            # se normaliza M con n1, que contiene 1/mcsteps/L^2. El L^2 por la magnetizacion normalizada al tamaño del sistema y el mcsteps por hacer la media.

#código para ejecutar la simulación del modelo del votante con ruido y edad en red cuadrada
eqSteps = 2000
mcSteps = 4000
nb=30
m=3
barr = np.linspace(0,0.045,nb)
f = open("aging_4vec_0806.txt", "a")
f.write("Now the file has more content!")
Larray=[10,20,30,40,60,80]
GRAFICO_4vecinos_edad(Larray,nb,mcSteps,eqSteps)
f.write('se acaba')