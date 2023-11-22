
import numpy
import matplotlib.pyplot as plt





from statsmodels.tsa.stattools import acf as autocorr



class evaluator():
    def __init__(self,nb_auto_correl):
        self.nb_auto_correl=nb_auto_correl
        #self.time=time
        self.count=0
    def correl(self,samples,i):#à corder
        
        n = len(samples)
        
        acf = autocorr(samples, nlags=self.nb_auto_correl, fft=True)
        sums = 0
        for k in range(1, len(acf)):
            sums = sums + (n-k)*acf[k]/n

        return n/(1+2*sums)
        

    def ESS(self,samples):
        """
        computes the ESS related to samples
        """
        
        n = len(samples)
        
        
        acf = autocorr(samples, nlags=self.nb_auto_correl, fft=True)
        sums = 0
        for k in range(1, len(acf)):
            sums = sums + (n-k)*acf[k]/n

        self.ESS_val=n/(1+2*sums)
        
        
        
        return self.ESS_val
    


    def summary(self,samples,time,title,plot=True,true_param=None,true=False):
        print("ESS")
        print(self.ESS(samples))
        Ess=self.ESS(samples)
        print("ESS/t")
        print(Ess/time)
        if true:
            print("error mean")
            print(self.error_true(true_param,samples))
        print("plot")
        if plot:
            plt.figure(self.count)
            plt.plot(samples[3:])
            plt.title(title)
            plt.ylabel("F.X")
            plt.xlabel("iterations")
            plt.legend()
            plt.show()
        self.count+=1
        return Ess
        #self.plot(samples)