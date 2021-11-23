
def mol(t,C,n,CA0in,dz,Da,U,k):
     dCdt = zeros(n,1)
     C[1] = CA0in + 1/900*(60*t-t*t) #Inlet boundary condition 
     C[n+1] = 1/3*(4*C(n)-C(n-1)) #at the end dp/dt = 0

     for i in 1+range (i):
        dCdz[i]  = (C[n+1]-C[n-1])* 1/[2*dz] #centered
        d2Cdz2[i] = 1/(dz^2)*(C[i+1]-2*C[i] + C[i-1])
        dCdt[i] = Da*d2Cdz2(i) - U*dCdz[i]-k*C[i]^2
    
    
     
     
    
