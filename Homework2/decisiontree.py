pred_PW=[]
pred_PL=[]
accu_PW = []
accu_PL=[]
classes=np.array([[int(y_train[0])] for y_train in lines]) 
PW=np.array([[int(p_width[1])] for p_width in lines])
PL =np.array([[int(p_length[2])] for p_length in lines])
#SW = np.array([[int(s_width[1])] for s_width in lines])

#print(PW.T)
#Decision stump for Petal Width Feature
for j in range(len(PW)):
    if(int(PW[j])<8):
        pred_PW=np.append(pred_PW,0)
    elif(int(PW[j])>8 and int(PW[j])<=16):
        pred_PW=np.append(pred_PW,2)
    elif(int(PW[j])>16):
        pred_PW=np.append(pred_PW,1)
#Test for accuracy with Decision Stump with Petal Width        
for test in range(len(classes)):
    if (int(pred_PW[test]) is int(classes[test])):
        accu_PW=np.append(accu_PW,1)
    else:
        accu_PW=np.append(accu_PW,0)
        
#Decision stump for Petal Length Feature
for j in range(len(PL)):
    if(int(PL[j])<25):
        pred_PL=np.append(pred_PL,0)
    elif(int(PL[j])>=25 and int(PL[j])<=50):
        pred_PL=np.append(pred_PL,2)
    elif(int(PL[j])>50):
        pred_PL=np.append(pred_PL,1)
#Test for accuracy with Decision Stump with Petal Length        
for test in range(len(classes)):
    if (int(pred_PL[test]) is int(classes[test])):
        accu_PL=np.append(accu_PL,1)
    else:
        accu_PL=np.append(accu_PL,0)
                
        
accu_mean_PW = np.mean(accu_PW)
accu_mean_PL = np.mean(accu_PL)
 
print('Accuracy of Decision stump using Petal Width: %f'%accu_mean_PW)
print('Accuracy of Decision stump using Petal Length: %f'%accu_mean_PL)

