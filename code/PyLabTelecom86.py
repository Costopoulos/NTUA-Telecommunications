#!/usr/bin/env python
# coding: utf-8

# In[2]:


#O kwdikas tha grafei antikeimenostrefws
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sympy.combinatorics.graycode import GrayCode    


# In[3]:


#For Question 1
#Every plot is scatter

#For sub-questions a)i), a)ii), b)
#Samples and plots squared triangular
def sampled_squared_triangular(fm,fs,n,title_of_graph):
    t=np.linspace(0,n/fm,n*fs/fm) #linear space for n=4 periods 	    
    triangular=2*signal.sawtooth(2*np.pi*fm/2*t,0.5) #trig=sawtooth with width=0.5, squared signal's f=f/2
    squared_triangular=triangular*triangular
    plt.scatter(t,squared_triangular,label='Squared Triangular Sampled')
    plt.legend(loc='upper right')
    plt.title("Sampled Squared Triangular Pulse Train for Frequency of" + str(fs) + "Hz")
    plt.xlabel("Time (Sec)")
    plt.ylabel("Amplitute (Volts)")
    plt.grid()
    plt.xlim(-0.0,n/fm) #if there was no xlim the signal would last from -nTm/2 till nTm/2
    plt.savefig(title_of_graph+".jpg")
    plt.show()
    return t,squared_triangular


# In[5]:


#For sub-questions 1)a)iii),1)c)i)a')iii)
#Merges plots
def fuse_plots(signal1,t1,fs1,signal2,t2,fs2,title_of_graph):
    plt.scatter(t2,signal2,label='Sampled Triangular for Frequency of' + str(fs2) +"Hz")
    plt.scatter(t1,signal1,label='Sampled Triangular for Frequency of' + str(fs1) + "Hz")
    plt.legend(loc='upper right')
    plt.xlabel("Time (Sec)")
    plt.ylabel("Amplitude (Volts")
    plt.title("Fused Signals")
    plt.grid()
    plt.xlim(-0.0,max(t2[-1],t1[-1])) #could have wrote xlim(-0.0,n/fm) for n=4 periods since I set rightlim=4/fm in each signal
    plt.savefig(title_of_graph+".jpg")
    plt.show()

#For subquestion c)i)
#Samples the sine function z(t)=Asin(2pifmt)
def sampled_zsine(fm,fs,title_of_graph):
    t=np.linspace(0,4/fm,4*fs/fm) #samples for linspace of n=4 periods
    zsine=np.sin(2*np.pi*fm*t)
    plt.scatter(t,zsine,label='Sampled Sine') #plot the sine func z
    plt.legend(loc='upper right')
    plt.title("Sampled Sine for Frequency="+str(fs)+"Hz")
    plt.xlabel("Time (Sec)")
    plt.ylabel("Amplitude (Volts)")
    plt.grid()
    plt.xlim(-0.0,4/fm)    #could also have used plt.plot(t,zsine,'o') and avoid xlim
    plt.savefig(title_of_graph+".jpg")
    plt.show()
    return t, zsine


# In[ ]:





# In[6]:


#For subquestion c)ii)
#Samples the sum of the two sines
def sampled_sum_of_sines(fm,fs,title_of_graph):
    t=np.linspace(0,0.001,fs/1000) #period of sum=1/1000=1/L
    sum=np.sin(2*np.pi*fm*t)+np.sin(2*np.pi*(fm+1000)*t)
    plt.scatter(t,sum,label='Sum of Sampled Sines')
    plt.legend(loc='upper right')
    plt.title("Sum of the Sampled Sines with Frequency="+str(fs)+"Hz")
    plt.xlabel("Time (Sec)")
    plt.ylabel("Amplitude (Volts)")
    plt.grid()
    plt.xlim(-0.0,0.001) #xlim(-0.0,1/L)
    plt.savefig(title_of_graph+".jpg")
    plt.show()
    return t,sum


            #For Question 2
#For subquestion a
#Quantization for q=5 bits
def quantized_sampled_signal(signal,t,title_of_graph):
    L=2**5 #total number of levels in binary -> L=2**q for q=5 bits
    D=8/L #step size=D=2*A/L for A=4 Volts
    quantized_signal=D*np.round(signal/D)
    g=GrayCode(5) #the 5-dimensional cube is created
    code_gray=list(g.generate_gray()) #its values are put in an array
    plt.yticks(np.arange(-4,4+D,step=D),code_gray) #from -A to A+D
    plt.step(t,quantized_signal,label='Quantized Sampled Signal')
    plt.legend(loc='upper right')
    plt.xlabel("Time (Sec)")
    plt.ylabel(str(5)+"Gray Code bits")
    plt.title("Output of the Quantizer")
    plt.grid()
    plt.savefig(title_of_graph+".jpg")
    plt.show()
    return quantized_signal


# In[ ]:





# In[7]:


#For subquestion b
#Computes the standard deviation for the N first specimens in an array
def standard_deviation(a,N):
    sum=0
    for i in range(0,N):
        sum=sum+a[i]
    average=sum/N
    deviation=0
    for i in range(0,N):
        deviation=deviation+(a[i]-average)**2
    deviation=(deviation/(N-1))**0.5
    return deviation

#For subquestions b)i),b)ii)
#Computes the error of the quantization for the first N values 
def quantization_error(quantized_signal,original_signal,N):
    error=np.zeros(N)
    for i in range (0,N):
        error[i]=quantized_signal[i]-original_signal[i]
    return error

#For subquestion b)iii)
#Computes the signal to noise ratio for the first N values
def SNR(signal,noise,N):
    squared_values_of_signal=np.zeros(N)
    squared_values_of_noise=np.zeros(N)
    sum_of_squared_values_of_signal=0
    sum_of_squared_values_of_noise=0
    for i in range (0,N):
        squared_values_of_signal[i]=signal[i]**2
        squared_values_of_noise[i]=noise[i]**2
        sum_of_squared_values_of_signal+=squared_values_of_signal[i]
        sum_of_squared_values_of_noise+=squared_values_of_noise[i]
    avg_of_the_sum_of_the_squared_values_of_signal=sum_of_squared_values_of_signal/N
    avg_of_the_sum_of_the_squared_values_of_noise=sum_of_squared_values_of_noise/N	
    rms=(avg_of_the_sum_of_the_squared_values_of_signal/avg_of_the_sum_of_the_squared_values_of_noise)**0.5
    snr=rms**2
    return snr


# In[61]:


#For subquestion c

#def bipolar_rz (quantized_signal,t,title_of_graph):
    #result_graph=[]
    #thislist=np.zeros(1/fm)
    #for i in range (0,225)
    #flag=0
    #for i in range (0,225):
        #if quantized_signal[i]==0:
            #result_graph.append(0)
            #result_graph.append(0)
        #elif (quantized_signal[i]=1 and flag==0):
            #result_graph.append(45) #A=45V
            #result_graph.append(0) #Half of a 0 bit is in +V and the other half in zero
            #flag=1 #So we know we are about to go negative
        #elif (quantized_signal[i]=1 and flag==1):
            #result_graph.append(0)
            #result_graph.append(-45) #Half of a 1 bit is in -V and the other in one
    #t=np.linspace(0,0.225,450) #45 samples *5 bits=225 
    #plt.step(t,result_graph,where='post')
    #plt.grid()
    #plt.savefig(title_of_graph+".jpg")
    #plt.show()


# In[10]:


#For Question 3
#For subquestion a
#Modulates a carrier signal with frequency fc and amplitude 1 for a modulation coefficient ka=0.5
def AM_modulation (carrier_signal,t,ka,title_of_graph):
    carrier_signal=np.sin(2*np.pi*fm*t)
    modulated_signal=(1+ka*np.sin(2*np.pi*35*t))*carrier_signal
    plt.plot(t,modulated_signal,label='Modulated Signal')
    plt.legend(loc='upper right')
    plt.xlabel("Time (Sec)")
    plt.ylabel("Amplitude (Volts)")
    plt.title("Modulated Signal")
    plt.grid()
    plt.savefig(title_of_graph+".jpg")
    plt.show()
    return modulated_signal

#Sample for f=35.0 Hz
def sampled_zsine2(f,fs,title_of_graph):
    t=np.linspace(0,4/f,4*fs/f) #linspace stop is 4/f cause (4/f)>(4/fm) 
    zsine=np.sin(2*np.pi*fm*t)
    plt.plot(t,zsine) #plot the sine func z
    plt.title("Sampled Sine for Frequency="+str(fs)+"Hz")
    plt.xlabel("Time (Sec)")
    plt.ylabel("Amplitude (Volts)")
    plt.grid()
    plt.xlim(-0.0,4/fm)    #could also have used plt.plot(t,zsine,'o') and avoid xlim
    plt.savefig(title_of_graph+".jpg")
    plt.show()
    return t, zsine

#For subquestion b
#Average of an array a out of the N first values
def avg (a,N):
    average=0
    for i in range(0,N):
        average+=a[i]
    return average/N

#Demodulate a modulated signal with sampling frequency fs5, carrier freq fm, amplitude of carrier and info signal=1
def AM_demodulation(modulated_signal,t,A,fs,fc,Ac,title_of_graph):
    fn=fs/2.0
    cut_off=50 #freq of info is 35 so I need something higher
    demodulated=4*modulated_signal*np.sin(2*np.pi*fc*t)
    DC=avg(demodulated,len(t))
    demodulated=demodulated-DC #-DC Value of signal
    taps=signal.firwin(1003,cut_off/fn)
    demodulated=signal.lfilter(taps,1.0,demodulated)
    plt.plot(t,demodulated, label='Demodulated Signal')
    plt.legend(loc='upper right')
    plt.xlabel("Time[sec]")
    plt.ylabel("Amplitude[Volt]")
    plt.title("Demodulated Signal with a cut-off frequency of "+str(cut_off)+"Hz")
    plt.grid()
    plt.savefig(title_of_graph+".jpg")
    plt.show()


# In[ ]:





# In[44]:


#START OF EXERCISE 03117043
#def AM_03117043():
    #ARXIKOPOIHSEIS
    #AM=7.0=4+3
fm=7000.0 #Hz
fs1=25*fm
fs2=60*fm
fs3=5*fm
fs4=45*fm
fs5=130*fm
fi=35.0
f=35.0
quantizing_bits=5  #AM=odd


# In[45]:


##EXERCISE 1##


                #QUESTION a
#subquestion i

t1,first_sampled_squared_triangular_signal=sampled_squared_triangular(fm,fs1,4,"1)a)i)")


# In[46]:


#subquestion ii

t2,second_sampled_squared_triangular_signal=sampled_squared_triangular(fm,fs2,4,"1)a)ii)")


# In[47]:


#subquestion iii

fuse_plots(first_sampled_squared_triangular_signal,t1,fs1,second_sampled_squared_triangular_signal,t2,fs2,"1)a)iii)")


# In[48]:


#QUESTION b
t3,third_sampled_squared_triangular_signal=sampled_squared_triangular(fm,fs3,4,"1)b")


# In[49]:


#QUESTION c
#subquestion i

#sub-subquestion a'

t1,first_sampled_zsine=sampled_zsine(fm,fs1,"1)c)i)a')i") #i
t2,second_sampled_zsine=sampled_zsine(fm,fs2,"1)c)i)a')ii") #ii
fuse_plots(first_sampled_zsine,t1,fs1,second_sampled_zsine,t2,fs2,"1)c)i)a')iii") #iii


# In[50]:


#sub-subquestion b'

t3,third_sampled_zsine=sampled_zsine(fm,fs3,"1)c)b'")


# In[51]:


#subquestion ii

#sub-subquestion a'

t1,first_sampled_sum_of_sines=sampled_sum_of_sines(fm,fs1,"c)ii)a')i") #i
t2,second_sampled_sum_of_sines=sampled_sum_of_sines(fm,fs2,"c)ii)a')ii") #ii
fuse_plots(first_sampled_sum_of_sines,t1,fs1,second_sampled_sum_of_sines,t2,fs2,"c)ii)a')iii")#iii


# In[52]:


#sub-subquestion b'

t3,third_sampled_sum_of_sines=sampled_sum_of_sines(fm,fs3,"c)ii)b'")


# In[53]:


##EXERCISE 2##

#QUESTION a
t4,fourth_sampled_squared_triangular=sampled_squared_triangular(fm,fs4,1,"2)a)1st") #I will quantize in 1 period
quantized_fourth_sampled_squared_triangular=quantized_sampled_signal(fourth_sampled_squared_triangular,t4,"2)a)2nd")


# In[63]:


#QUESTION b
#subquestion i

quantization_error_10=quantization_error(quantized_fourth_sampled_squared_triangular,fourth_sampled_squared_triangular,10)
standard_deviation_10=standard_deviation(quantization_error_10,10)
print("The standard deviation of the first 10 values of the quantization error is 0.0406593598725838")


# In[64]:


#subquestion ii

quantization_error_20=quantization_error(quantized_fourth_sampled_squared_triangular,fourth_sampled_squared_triangular,20)
standard_deviation_20=standard_deviation(quantization_error_20,20)
print("The standard deviation of the first 20 values of the quantization error is 0.0279836906516282")


# In[65]:


#subquestion iii

#sub-subquestion 1

snr_10=SNR(fourth_sampled_squared_triangular,quantization_error_10,10)
print("The SNR of the first 10 values of the quantization error is: 4777.8119525619")


# In[66]:


#sub-subquestion 2

snr_20=SNR(fourth_sampled_squared_triangular,quantization_error_20,20)
print("The SNR of the first 20 values of the quantization error is: 3319.96574413596")


# In[62]:


#bipolar_rz(quantized_fourth_sampled_squared_triangular,t4,"2)c")


# In[58]:


##EXERCISE 3##

                #Question a
t4, fourth_sampled_zsine=sampled_zsine2(f,fs5,"3)a)1st") #f=35Hz -> linspace
modulated_signal=AM_modulation(fourth_sampled_zsine,t4,0.5,"3)a)2nd")


# In[59]:


#Question b
A=1
demodulated_signal=AM_demodulation(modulated_signal,t4,A,fs5,fm,A,"3)b")


# In[28]:


#START OF EXERCISE 03117012
#def AM_03117012():
    #ARXIKOPOIHSEIS
    #AM=3.0=1+2
fm=3000.0 #Hz
fs1=25*fm
fs2=60*fm
fs3=5*fm
fs4=45*fm
fs5=130*fm
fi=35.0
f=35.0
quantizing_bits=5  #AM=odd


# In[29]:


##EXERCISE 1##


                #QUESTION a
#subquestion i

t1,first_sampled_squared_triangular_signal=sampled_squared_triangular(fm,fs1,4,"croto1)a)i)")


# In[30]:


#subquestion ii

t2,second_sampled_squared_triangular_signal=sampled_squared_triangular(fm,fs2,4,"croto1)a)ii)")


# In[31]:


#subquestion iii

fuse_plots(first_sampled_squared_triangular_signal,t1,fs1,second_sampled_squared_triangular_signal,t2,fs2,"croto1)a)iii)")


# In[32]:


#QUESTION b
t3,third_sampled_squared_triangular_signal=sampled_squared_triangular(fm,fs3,4,"croto1)b")


# In[33]:


#QUESTION c
#subquestion i

#sub-subquestion a'

t1,first_sampled_zsine=sampled_zsine(fm,fs1,"croto1)c)i)a')i") #i
t2,second_sampled_zsine=sampled_zsine(fm,fs2,"croto1)c)i)a')ii") #ii
fuse_plots(first_sampled_zsine,t1,fs1,second_sampled_zsine,t2,fs2,"croto1)c)i)a')iii") #iii


# In[34]:


#sub-subquestion b'

t3,third_sampled_zsine=sampled_zsine(fm,fs3,"croto1)c)b'")


# In[35]:


#subquestion ii

#sub-subquestion a'

t1,first_sampled_sum_of_sines=sampled_sum_of_sines(fm,fs1,"crotoc)ii)a')i") #i
t2,second_sampled_sum_of_sines=sampled_sum_of_sines(fm,fs2,"crotoc)ii)a')ii") #ii
fuse_plots(first_sampled_sum_of_sines,t1,fs1,second_sampled_sum_of_sines,t2,fs2,"crotoc)ii)a')iii")#iii


# In[36]:


#sub-subquestion b'

t3,third_sampled_sum_of_sines=sampled_sum_of_sines(fm,fs3,"costoc)ii)b'")


# In[37]:


##EXERCISE 2##

                #QUESTION a
t4,fourth_sampled_squared_triangular=sampled_squared_triangular(fm,fs4,1,"croto2)a)1st") #I will quantize in 1 period
quantized_fourth_sampled_squared_triangular=quantized_sampled_signal(fourth_sampled_squared_triangular,t4,"croto2)a)2nd")


# In[67]:


#QUESTION b
#subquestion i

quantization_error_10=quantization_error(quantized_fourth_sampled_squared_triangular,fourth_sampled_squared_triangular,10)
standard_deviation_10=standard_deviation(quantization_error_10,10)
print("The standard deviation of the first 10 values of the quantization error is 0.051418180198324975")


# In[68]:


#subquestion ii

quantization_error_20=quantization_error(quantized_fourth_sampled_squared_triangular,fourth_sampled_squared_triangular,20)
standard_deviation_20=standard_deviation(quantization_error_20,20)
print("The standard deviation of the first 20 values of the quantization error is 0.03538841863346253")


# In[70]:


#subquestion iii

#sub-subquestion 1

snr_10=SNR(fourth_sampled_squared_triangular,quantization_error_10,10)
print("The SNR of the first 10 values of the quantization error is: 2702.132831073378")


# In[71]:


#sub-subquestion 2

snr_20=SNR(fourth_sampled_squared_triangular,quantization_error_20,20)
print("The SNR of the first 20 values of the quantization error is: 1041.2201954243662")


# In[42]:


##EXERCISE 3##

                #Question a
t4, fourth_sampled_zsine=sampled_zsine2(f,fs5,"croto3)a)1st") #f=35Hz -> linspace
modulated_signal=AM_modulation(fourth_sampled_zsine,t4,0.5,"croto3)a)2nd")


# In[43]:


#Question b
A=1
demodulated_signal=AM_demodulation(modulated_signal,t4,A,fs5,fm,A,"croto3)b")


# In[ ]:




