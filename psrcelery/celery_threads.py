import numpy as np
import dill

def init(_cel_gp,_x,_y):
    global cel_gp
    global x
    global y
    print("init thread")
    cel_gp = dill.loads(_cel_gp)
    x=_x
    y=_y

def run(i):
    global cel_gp
    global x
    global y
    print(f"{i}")
    pred_y = cel_gp.predict(y, x[i], return_cov=False)
    return pred_y
