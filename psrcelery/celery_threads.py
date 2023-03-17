import numpy as np

def init(_cel_gp,_x,_y):
    global cel_gp
    global x
    global y
    print("init thread")
    cel_gp = _cel_gp
    x=_x
    y=_y

def run(i):
    global cel_gp
    global x
    global y
    print(f"{i}")
    pred_y, cov = cel_gp.predict(y, x[i], return_cov=True)
    pred_yerr = np.sqrt(np.diag(cov))
    return pred_y,pred_yerr,cov
