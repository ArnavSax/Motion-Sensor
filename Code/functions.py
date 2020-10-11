import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read(filename):
    titles=["accelX", "accelY", "accelZ", "magneX", "magneY", "magneZ", "gyrosX", "gyrosY", "gyrosZ", "time(ms)"]
    dataframe = pd.read_csv(str(filename), names=titles).iloc[1:-1].astype(float)
    dataframe["time(ms)"] -= dataframe.iloc[0, dataframe.columns.get_loc('time(ms)')] # scale time on the sensor down
    return dataframe

#Mean average value
def meanABS(d, col):
    return d[col].abs().mean()
    
#Root mean square
def rootMS(d, col):
    return np.sqrt(np.mean([i**2 for i in d[col]]))

#Slow sign change
def slopeSignChange(d, col):
    times = []
    data = d[col].diff().to_numpy()
    for i in range(1, len(data)-1):
        if data[i]*data[i+1] <= 0:
            times.append(df.iloc[i]["time(ms)"])        
    return times
    
#Positive Peak
def posPeak(d, col):
    times = []
    data = d[col].diff().to_numpy()
    for i in range(1, len(data)-1):
        if data[i]*data[i+1] < 0 and data[i] > data[i+1]:
            times.append(df.iloc[i]["time(ms)"])
    return times

#Negative Peak
def negPeak(d, col):
    times = []
    data = d[col].diff().to_numpy()
    for i in range(1, len(data)-1):
        if data[i]*data[i+1] < 0 and data[i] < data[i+1]:
            times.append(df.iloc[i]["time(ms)"])   
    return times

#Zero crossing
def zeroCrossing(d, col):
    times = []
    data = d[col].to_numpy()
    for i in range(0, len(data)-1):
        if data[i]*data[i+1] < 0 or data[i] == 0:
            times.append(df.iloc[i]["time(ms)"])
    return times

#Clip Dataframe
def clip(d, num):
    return d.iloc[num:-num]

#chunk dataframe
def chunk(d, window, overlap):
    out = []
    for i in range(0, d.size, window-overlap):
        try:
            out.append(d.iloc[i:i+window])
        except:
            out.append(d.iloc[i:])
            break
    return out

df = read("right arm straight reach.log")

plt.plot(df["time(ms)"], df["gyrosX"])
    


            
    
    
    

