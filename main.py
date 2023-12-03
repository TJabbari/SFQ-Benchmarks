"""
File name: Main.py
Author: Tahereh Jabbari
Date created: 12/1/2023

Description:
This script demonstrates ...

"""

################ imports ##################
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import copy

########### Parameters ##########
DFF_Width = 60
Cell_Width = 80
Cell_height = 40
Min_Space = 20   # Min Space requirement between the cells
M_X_b = 0.0 # margin of X
M_X_e = 75.84 # margin of X
M_Y_b = 0.0
M_Y_e = 74.56
Max_of_X = 212.96 # Maxium of Xs in Cmos data + 2.88 ( Width of cells in cmos benchmark )

########## Functions ###########
def Draw(n,size,x,y,H_data,i):

    if n==0:
        return

    x0= x-size/2.
    x1 = x + size / 2.
    y0 = y - size / 2.
    y1 = y + size / 2.

    #H_data = np.zeros(1, dtype={'names': ('H_name', 'P_UL_X', 'P_UL_Y', 'P_L_X', 'P_L_Y', 'P_DL_X', 'P_DL_Y','P_M_X'
    #                                      ,'P_M_Y','P_UR_X','P_UR_Y','P_R_X','P_R_Y','P_DR_X','P_DR_Y','Size'),
    #                         'formats': ('U10', 'f8', 'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8')})

    H_data[i]['H_name']= 'H'+str(i)
    H_data[i]['P_UL_X']= x0
    H_data[i]['P_UL_Y']= y1
    H_data[i]['P_L_X'] = x0
    H_data[i]['P_L_Y'] = y
    H_data[i]['P_DL_X']= x0
    H_data[i]['P_DL_Y'] = y0
    H_data[i]['P_M_X'] = x
    H_data[i]['P_M_Y'] = y
    H_data[i]['P_UR_X'] = x1
    H_data[i]['P_UR_Y'] = y1
    H_data[i]['P_R_X'] = x1
    H_data[i]['P_R_Y'] = y
    H_data[i]['P_DR_X'] = x1
    H_data[i]['P_DR_Y'] = y0
    H_data[i]['Size']= size
    i=4*i+1
    #print(H_data[0])


   # plt.plot([x, x], [y, (y - size / 2)], )
    plt.imshow
    #plt.pause(0.1)
    plt.plot([x0,x1],[y,y], color='yellow', linewidth= 3)
    plt.imshow
    #plt.pause(0.1)
    plt.plot([x0,x0],[y0,y1],color='yellow', linewidth= 3)
    plt.imshow
    #plt.pause(0.1)
    plt.plot([x1, x1], [y0, y1], color='yellow', linewidth= 3)
    plt.imshow
    #plt.pause(0.1)
    Draw(n-1,size/2.,x0,y0,H_data,i)
    Draw(n-1,size/2.,x0,y1,H_data,i+1)
    Draw(n-1,size/2.,x1,y0,H_data,i+2)
    Draw(n-1,size/2.,x1,y1,H_data,i+3)

### convert the string to the float value
def Stirng_to_Float(input):
   Test=input
   Test=Test.translate({ord('{'): None})
   Test=Test.translate({ord('}'): None})
   #Test=Test.replace(' ', ',')
   Test=list(map(float, Test.split()))
   return Test
# draw function
def func_draw(index,X):

    Y=0
    plt.plot([X[index, 0], X[index+1, 0]],[X[index, 1], X[index+1, 1]], color='black')  ###
    plt.plot((X[index+1, 0], X[index+2, 0]), (X[index + 1, 1], X[index + 2, 1]), color='black')
    plt.plot((X[index+2, 0], X[index+3, 0]), (X[index + 2, 1], X[index + 3, 1]), color='black')
    plt.plot((X[index, 0], X[index+3, 0]), (X[index , 1], X[index + 3, 1]), color='black')

    return Y


########## Reading Input #######
df = pd.read_csv('newdataB09.csv')   # Read  floorplan info of CMOS benchmark

######### Variables #############
Name_array = np.asarray(df['base_name'] ) # Array of Circuit Name
DFF_index = np.zeros(4*(len(df['base_name'])), dtype=float) # index of DFF
x = np.zeros(4*(len(df['boundary'])), dtype=float) #
point_index = np.zeros(4*(len(df['boundary'])), dtype=float)
clock_index = np.zeros(4*(len(df['boundary'])), dtype=float)
clock_index_d = np.zeros(2*(len(df['boundary'])), dtype=float)
y = np.zeros(4*len((df['boundary'])), dtype=float)
xd = np.zeros(2*(len(df['boundary'])), dtype=float)
yd = np.zeros(2*len((df['boundary'])), dtype=float)
d  = np.zeros(len((df['boundary']))+2*186, dtype=float)



################## Adjustment of size of DFF #########################
for i in range(len(Name_array)):
    if ((np.char.find(Name_array[i], "DFF", start=0, end=None)== 1) or (np.char.find(Name_array[i], "dff", start=0, end=None)== 1)):
        DFF_index[i] = DFF_Width
        DFF_index[i+1] = DFF_Width
        DFF_index[i+2] = DFF_Width
        DFF_index[i+3] = DFF_Width
    else:
        DFF_index[i] = Cell_Width
        DFF_index[i+1] = Cell_Width
        DFF_index[i+2] = Cell_Width
        DFF_index[i+3] = Cell_Width

################ extract the X,Y of four boundry points ##########
for i in range(len(df['boundary'])):
    F_Boundary= Stirng_to_Float(df['boundary'][i])
    C_name = df['base_name'][i]
    for j in range(4):
        if (C_name.find('split') != -1):
            clock_index[4*i+j]= 0.0
        else:
            clock_index[4*i+j]= 1.0
        x[4*i+j]= F_Boundary[2*j]
        y[4*i+j]= F_Boundary[2*j+1]

############### extarct two right points ###########################
for i in range(len(df['boundary'])):
    F_Boundary= Stirng_to_Float(df['boundary'][i])
    Cd_name = df['base_name'][i]
    for j in range(2):
        if (Cd_name.find('split') != -1):
            clock_index_d[2* i + j] = 0.0
        else:
            clock_index_d[2* i + j] = 1.0
        xd[2*i+j]= F_Boundary[6*j]
        yd[2*i+j]= F_Boundary[6*j+1]

####### data points ( all four points ) #############
data = pd.DataFrame({'x': x, 'y': y})

# Getting the values
f1 = data['x'].values
f2 = data['y'].values
for i in range(len(point_index)):
    point_index[i] = i + 1
X = np.array(list(zip(f1, f2, point_index))) # points with index
X_WI = np.array(list(zip(f1, f2))) # points without index

###### sort points by the Y cordinate ################
y1_sort = np.unique(np.sort(f2))
ind1 = np.lexsort((X[:,0],X[:,1]))
temp= X[ind1]


##### adjust the row height ########################
j = 0
for i in range(len(temp)-1):
    if (temp[i][1] == temp [i+1][1]):
        temp[i][1] = j* Cell_height
    else:
        temp[i][1] = j * Cell_height
        j = j+1

temp[len(temp)-1][1] = j*Cell_height
max_fig = j*Cell_height + Cell_height

####################### Intial Space Ratio for distrubution of the cells ########################
j = 0
space = np.zeros((len(temp)), dtype=float)
space_ratio = np.zeros((len(temp)), dtype=float)
total_sum = 0
for i in range(len(temp)-1):
    if (temp[i][1] == temp[i + 1][1]):
        if (j % 2 == 0):  # First point of cells
            if (j == 0):  # First point of first cell
                space[i] = temp[i][0] - Min_Space
            else:
                space[i] = temp[i][0] - pervious_temp
        else:
            pervious_temp = temp[i][0]
        j = j + 1
        print("temp is", (temp[i]))
        flag2 = j
    else:
        space[i] = Max_of_X - Min_Space - temp[i][0]
        total_sum = np.sum(space[i-j:i+1])
        for k in range(i-j,i+1):
            space_ratio[k]= space[k] / total_sum
        #print(total_sum)
        j = 0

temp[len(temp)-1][0] = temp[i+1][0]
space[i+1] = Max_of_X - Min_Space - temp[i+1][0]
total_sum = np.sum(space[i+1-j:i+1+1])
for k in range(i+1-flag2,i+1+1):
        space_ratio[k]= space[k] / total_sum     ####### space ratio in cmos style cells ######


#################### fake temp for compute the maxium value after adjustment ######################
#################### This part of the code is just written to compute the maxium value ###########
old_temp = temp
temp_fake = temp
j = 0
for i in range(len(temp)-1):
    if (temp_fake[i][1] == temp_fake[i + 1][1]):
       if ( j % 2 == 0 ): # First point of cells
           if (j == 0 ): # First point of first cell
               temp_fake[i][0] = temp_fake[i][0]
           else:
                temp_fake[i][0] = ((temp[i][0] - pervious_temp)) + temp_fake[i-1][0]
       else:
           pervious_temp = temp_fake[i][0]
           temp_fake[i][0] = temp_fake[i-1][0] + Cell_Width
       j = j + 1
    else:
       temp_fake[i][0] = temp_fake[i - 1][0] + Cell_Width
       j = 0

temp_fake[len(temp)-1][0] = temp_fake[i][0] + Cell_Width # check this
max_new = np.max(temp_fake[:,0])
###########################################################

################### update the temp value ########################

j = 0
for i in range(len(temp)-1):
    if (temp[i][1] == temp[i + 1][1]):
       if ( j % 2 == 0 ): # First point of cells
           if (j == 0 ): # First point of first cell
               temp[i][0] = temp[i][0]
           else:
                temp[i][0] = ((temp[i][0] - pervious_temp)) + temp[i-1][0]
       else:
           pervious_temp = temp[i][0]
           temp[i][0] = temp[i-1][0] + Cell_Width
       j = j + 1
       print("temp is", (temp[i]))
    else:
       temp[i][0] = temp[i - 1][0] + Cell_Width
       j = 0

temp[len(temp)-1][0] = temp[i][0] + Cell_Width # check this

################# compute the new space ratio to apply ###########################
j = 0
space_new = np.zeros((len(temp)), dtype=float)
space_ratio_new = np.zeros((len(temp)), dtype=float)
total_sum = 0
for i in range(len(temp)-1):
    if (temp[i][1] == temp[i + 1][1]):
        if (j % 2 == 0):  # First point of cells
            if (j == 0):  # First point of first cell
                space_new[i] = temp[i][0]
            else:
                space_new[i] = temp[i][0] - pervious_temp
        else:
            pervious_temp = temp[i][0]
        j = j + 1
        flag = j
    else:
        space_new[i] = max_new - temp[i][0]
        total_sum = np.sum(space_new[i-j:i+1])
        for k in range(i-j,i+1):
            space_ratio_new[k]= space_new[k] / total_sum
        j = 0

space_new[i+1] = max_new - temp[i+1][0]
total_sum = np.sum(space_new[i+1 - flag:i+1 + 1])
for k in range(i+1 - flag, i+1 + 1):
    space_ratio_new[k] = space_new[k] / total_sum   ####### space ratio in sfQ style cells ######

############## compute the ratio of space ratio ##############################
change_space_ratio = np.zeros((len(temp)), dtype=float)
for i in range(len(temp)):
    if (space_ratio[i] != 0.0):
        change_space_ratio[i] = space_ratio[i]/space_ratio_new[i]



################################ apply the ratio of space ratio between cmos and SFQ ###########################
j = 0
for i in range(len(old_temp)-1):
    if (old_temp[i][1] == old_temp[i + 1][1]):
       if ( j % 2 == 0 ): # First point of cells
           if (j == 0 ): # First point of first cell
               old_temp[i][0] = old_temp[i][0] * change_space_ratio[i]
           else:
               check = (old_temp[i][0] - pervious_temp) * change_space_ratio[i]
               if( check < Min_Space and  check > 0):  #### ADD Min Space between cells
                   old_temp[i][0] = Min_Space + old_temp[i - 1][0]
               else:  #### ADD  Min Space between cells
                   old_temp[i][0] = ((old_temp[i][0] - pervious_temp) * change_space_ratio[i]) + old_temp[i-1][0]
       else:
           pervious_temp = old_temp[i][0]
           old_temp[i][0] = old_temp[i-1][0] + Cell_Width
       j = j + 1
    else:
       old_temp[i][0] = old_temp[i - 1][0]  + Cell_Width
       j = 0

old_temp[len(old_temp)-1][0] = old_temp[i][0] + Cell_Width # check this
sorted_temp = old_temp[temp[:, 2].argsort()]
X_WI = sorted_temp[:,0:2]
X = X_WI

############################### reading the 2 right points for plotting ##############################
data_d = pd.DataFrame({'x': xd, 'y': yd})

# Getting the values and plotting it
f1_d = data_d['x'].values
f2_d = data_d['y'].values
X_d = np.array(list(zip(f1_d, f2_d, clock_index_d)))

###################### sort the y values #########################################
y_sort = np.unique(np.sort(f2_d))
ind = np.lexsort((X_d[:,0],X_d[:,1]))
temp= X_d[ind]

d_y0 = temp[0,1] - M_Y_b
d[0]= temp[0,0] - M_X_b

for i in range(0,len(X),4):
    Y=func_draw(i,X)

plt.ylim(0, max_fig)
plt.show()



