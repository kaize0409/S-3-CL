import numpy as np

class DPMM:
    def __init__(self,X):                    
        self.K = 1
        self.d = X.shape[1]
        self.z = np.mod(np.random.permutation(X.shape[0]),self.K)+1
        self.mu = np.random.standard_normal((self.K, self.d))
        self.sigma = 1
        self.nk = np.zeros(self.K)
        self.pik = np.ones(self.K)/self.K 

        self.mu = np.array([np.mean(X,0)])
        self.Lambda = 0.15
        self.max_iter = 10
        self.obj = np.zeros(self.max_iter)
        self.em_time = np.zeros(self.max_iter)   
        
    def fit(self,X):
        max_iter = self.max_iter        
        [n,d] = np.shape(X)      
        for iter in range(max_iter):
            dist = np.zeros((n,self.K))
            for kk in range(self.K):
                Xm = X - np.tile(self.mu[kk,:],(n,1))
                dist[:,kk] = np.sum(Xm*Xm,1)            
            dmin = np.min(dist,1)
            self.z = np.argmin(dist,1)
            idx = np.where(dmin > self.Lambda)
            
            if (np.size(idx) > 0):
                self.K = self.K + 1
                self.z[idx[0]] = self.K-1 
                self.mu = np.vstack([self.mu,np.mean(X[idx[0],:],0)])                
                Xm = X - np.tile(self.mu[self.K-1,:],(n,1))
                dist = np.hstack([dist, np.array([np.sum(Xm*Xm,1)]).T])
            self.nk = np.zeros(self.K)
            for kk in range(self.K):
                self.nk[kk] = self.z.tolist().count(kk)
                idx = np.where(self.z == kk)
                self.mu[kk,:] = np.mean(X[idx[0],:],0)
            
            self.pik = self.nk/float(np.sum(self.nk))

        return self.z, self.K
