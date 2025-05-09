from classes import *
from graph import *

gridsize = 7000
iterations = 100

ES = EcoSystem(N=gridsize)
print(ES.N)
# p_0, z_0, n_0, y_0, I_0 = starts
starts = 1, 1, 1, 1, I_0
ends = p_N, z_N, n_N, y_N 
phyto, zoo, nitro, organic, I = ES.going(starts=starts, ends=ends)
z = ES.z
plt.plot(z, phyto)

for i in range(iterations):
    phyto_new, zoo_new, nitro_new, organic_new, I_new = ES.going(starts=starts, ends=ends, 
                                            phyto=phyto, zoo=zoo, nitro=nitro, organic=organic, I=I)
    
    # delta = max(np.max(np.abs(phyto_new - phyto)),
    #             np.max(np.abs(zoo_new - zoo)),
    #             np.max(np.abs(nitro_new - nitro)),
    #             np.max(np.abs(organic_new - organic)),
    #             np.max(np.abs(I_new - I)))
    print(f"Итерация {i+1}, изменение: {np.max(np.abs(phyto_new - phyto))}")
    phyto, zoo, nitro, organic, I = phyto_new.copy(), zoo_new.copy(), nitro_new.copy(), organic_new.copy(), I_new.copy()
    if i%10==0:
        plt.plot(z, phyto)

z = ES.z
T = ES.T
phyto = ES.phyto
zoo = ES.zoo
nitro = ES.nitro
organic = ES.organic
I = ES.I

print(ES.h)

# plt.plot(phyto, z)
plt.title('phyto')
plt.show()

# plt.plot(zoo, z)
# plt.title('zoo')
# plt.show()

# plt.plot(nitro, z)
# plt.title('nitro')
# plt.show()

# plt.plot(organic, z)
# plt.title('organic')
# plt.show()

# plt.plot(I, z)
# plt.title('I')
# plt.show()
