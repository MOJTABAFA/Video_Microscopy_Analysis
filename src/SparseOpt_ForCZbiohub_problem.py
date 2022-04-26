''' An optical-flow based tracker, designed for problem #2 of CZBiohub Interview Questions 
    Developed by Mojtaba Fazli : mfazli@meei.harvard.edu'''  

import numpy as np
import cv2
import pandas as pd 
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import time
import os 

def save(fname, outname):
    os.system("ffmpeg -r 20 -i "+ str(fname)+ "%d.jpg -vcodec mpeg4 -y " +str(outname))

def transpose_jagged_list(a):
    ''' It receives a jagged list of the features extracted using KLT-tracker
    and tramsposes it in away that the objects will be the rows and frames 
    will be in columns. '''
    max_col = len(a[0])
    for row in a:
        row_length = len(row)
        if row_length > max_col:
            max_col = row_length
    a_trans = []
    for col_index in range(max_col):
        a_trans.append([])
        for row in a:
            if col_index < len(row):
                a_trans[colIndex].append(row[col_index])
    return a_trans

def distance( a , b ):
    return  math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def Trajectory_2d_plot( x, y, Invert):

    for i in range(number_of_points):

        plt.plot(x[i], y[i])#, marker='o', color='r', ls='')
    if Invert : 
        plt.gca().invert_yaxis()
        plt.show()
    else : 
        plt.show()

def intensity_kernel(frame_gray, x, y, Image_height, Image_width ):
    frvalue = 0
    cont = 0
    neighbors = 18

    if x > Image_height - neighbors : 
        x = Image_height - (neighbors + 8)
    
    if x < neighbors : 
        x = x + (neighbors + 8)
    
    if y > Image_width - neighbors : 
        y = Image_width - (neighbors + 8)
    
    if y < neighbors : 
        y = y + (neighbors + 8)
    
#    if ((x > neighbors) and ( x < Image_width - neighbors) and (y > neighbors) and (y < Image_height - neighbors)):
#    if(x > neighbors and x < Image_width - neighbors):
    for i in range(int(x - neighbors), int(x + neighbors)):
        for j in range(int(y - neighbors), int(y + neighbors)):
            if ((i < Image_width) and (j<Image_height)):  
                vs = frame_gray.item(j , i)
            else : 
                vs = 0 
            frvalue = frvalue + vs
            if vs >= 5 :
                cont += 1

    if cont == 0 : 
        avgIntensity = 0
    else:
        avgIntensity = frvalue / cont
#        print (str(frvalue) + ', cont = '+str(cont) + '  avg = ',avgIntensity)
    return avgIntensity

def Tracker( video_file, out_video_file, frame_threshold):
    
    cap = cv2.VideoCapture(video_file)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 250,
                           qualityLevel = 0.01,
                           minDistance = 11,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (12,12),
                      maxLevel = 5,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0.0003))
    # Create some random colors
    color = np.random.randint(0,255,(250,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    fnum = 0

    old_points = [[]]
    new_points = [[]]
    old_2x = [[]]
    old_2y = [[]]
    X = int(cap.get(3))
    Y = int(cap.get(4))
    font = cv2.FONT_HERSHEY_SIMPLEX

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_video_file, fourcc, 25.0, (X, Y))
    
    while(True):
       
        if (fnum <= frame_threshold): 
            
            ret,frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()

                old_points[fnum].append((a, b))
                new_points[fnum].append((c, d))

                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                cv2.putText(frame, str(i), (int(a),int(b)), font, 0.7, color[i].tolist())


            old_points.append([])
            new_points.append([])

            img1 = cv2.add(frame,mask)
            cv2.putText( img1, 'Frame # '+ str(fnum), (10, 40), font, 0.5, (0, 255, 50), 1)

            cv2.imshow('frame', img1)
            k = cv2.waitKey(30) & 0xff
            fnum += 1
        else : 
            break
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        out.write(img1)

    out.release()
    cv2.destroyAllWindows()
    cap.release()

    return(old_points)

def post_processing(old_points, video_file, out_video_file, frame_threshold, single_flag, desired_muliple, point_num, desired_traj_points) :

    rational_thresh = 10
    #to_skip = []

    objects = [ [x] for x in old_points[0]]

    #print( objects )
    # fix larger array
    for i in range (1,len(old_points) ): 
        print(i)
        used_objects = []
        for j in range (len(old_points[i])):
            dists = []
            
            for k in range( len(objects) ):
                if k in used_objects:
                    dists.append( 100000000000 )
                else :
                    #print( objects[k][-1] , old_points[i][j]  , j , i , k  )
                    d =  distance( objects[k][-1] , old_points[i][j] )
                    dists.append(d)

            closest =np.argmin( np.array(dists) ) 
            used_objects.append( closest )
            objects[ closest ].append( old_points[i][j] )
        
        for m in range (len(objects)):
            # print( len(objects[m]), m, i)
            # print( objects[m] )
            if (len (objects[m]) < i) :
                objects[m].append(objects[m][-1])

    olds = np.array(objects)
    #print (olds[1]) 
    #print (np.shape(olds))



    o_x = [] 
    o_y = []
    # o_x = [  [x[0]] for x in item for item in objects ]
    # o_y = [  [x[1]] for x in item for item in objects ]

    for i in range(len (objects)): 
        temp_x = []
        temp_y = []
        for j in range( len(objects[i])):
            temp_x.append( objects[i][j][0] )
            temp_y.append( objects[i][j][1] )
        o_x.append(temp_x)
        o_y.append(temp_y)

    x = np.array(o_x)
    y = np.array(o_y)  
    
    number_of_points = np.shape(x)[0]
    #++++++++{ Improved tracking visualization }+++++++++++
    fnum = 0
    avg_frame_intensity = []
 
    cap = cv2.VideoCapture(video_file)
    X = int(cap.get(3))
    Y = int(cap.get(4))
    objects_dimensions = np.shape(x)[0]
    color = np.random.randint(0,255,(250,3))
    font = cv2.FONT_HERSHEY_SIMPLEX

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_video_file, fourcc, 25.0, (X, Y))
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(old_frame)
    width, height = old_gray.shape
    objects_intensities = np.zeros(shape = (objects_dimensions, frame_threshold + 1))
    #print(np.shape(objects_intensities))

    if single_flag : 
            tracking_list = [point_num]
            status = 0

    elif desired_muliple : 
        tracking_list = desired_traj_points
        status = 1

    else : 
        tracking_list = np.zeros(number_of_points)
        status = 2
        for points in range(number_of_points):
            tracking_list[points] = points

    while(1):
       
        if (fnum <= frame_threshold): 
            
            ret,frame2 = cap.read()
            frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            if (fnum == 0):
                for i in range(0, np.shape(x)[0]):
                    objects_intensities[i, 0] = intensity_kernel(frame_gray, int(x[i, 0] + 1), int(y[i, 0] + 1), width, height)

            if (fnum > 0 ):
                for i in range(0, np.shape(x)[0]):

                    if i in tracking_list : 

                        objects_intensities[i, fnum] = intensity_kernel(frame_gray, int(x[i, fnum] + 1), int(y[i, fnum] + 1), width, height)

                        mask = cv2.line(mask, (x[i, fnum - 1], y[i, fnum - 1]), (x[i, fnum], y[i, fnum]), color[i].tolist(), 2)
                        frame2 = cv2.circle(frame2,(x[i, fnum - 1] ,y[i, fnum - 1]), 5, color[i].tolist(),-1)
                        cv2.putText(frame2, str(i), (int(x[i, fnum]),int(y[i, fnum])), font, 0.7, color[i].tolist())

            img2 = cv2.add(frame2,mask)
            cv2.putText( img2, 'Frame # '+ str(fnum), (10, 40), font, 0.5, (0, 255, 50), 1)

            cv2.imshow('frame2',img2)
            k = cv2.waitKey(30) & 0xff
            fnum += 1
        else : 
            break
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        avg_int_fr = average_Intensity(X, Y, frame_gray, 2)
        avg_frame_intensity.append(avg_int_fr)

        out.write(img2)

    olds = np.array(old_points)
    df_olds1 = pd.DataFrame(olds)
    df_olds1.to_csv("old_points.csv")

    df_olds = pd.DataFrame(x)
    df_olds.to_csv("x.csv")

    df_intensity = pd.DataFrame(objects_intensities)
    df_intensity.to_csv("Intensity.csv")

    df_news = pd.DataFrame(y)
    df_news.to_csv("y.csv")

    out.release()
    cv2.destroyAllWindows()
    cap.release()

    return (x, y, objects_intensities, avg_frame_intensity, status, tracking_list)

def average_Intensity(X, Y, frame_gray, Intensity_threshold):

    frvalue = 0
    cont = 0
    neighbors = 5

    for i in range(X):
        for j in range(Y):
                frvalue = frvalue + frame_gray.item(j, i)
                if frame_gray.item(j, i) > Intensity_threshold :
                    cont += 1

    if cont == 0 : 
        avgIntensity = 0
    else:
        avgIntensity = frvalue / cont

    return avgIntensity

def average_frame_inetnsity_plot(avg_int, z, frame_num, sliding_size):

    slidingwindow = sliding_size
    avg_velovalues = [] 
    avgframe = frame_num - slidingwindow
    slidingpoint = 0
    
    int_norm = []
    traj_points = frame_num
    max_pixel_value = max(avg_int)

    for i in range (traj_points):        
        
        if max_pixel_value != 0 :
        
            int_norm.append( [ avg_int[i] / (max_pixel_value )])
        else : 
            int_norm[i].append(0) 
    int_norm = np.array(int_norm)
    int_norm2 = np.array(int_norm[:, 0])
    # computing the sliding average velocity of objects
    while avgframe > 0 :

        endpoint = slidingpoint + slidingwindow
        sumwin= sum(int_norm2[slidingpoint : endpoint])
        avgvel = sumwin / slidingwindow
        avg_velovalues.append(avgvel)
        slidingpoint += 1
        avgframe -= 1
    #slidingpoint = 0


    # Removing the last blank element from the avg velo list 
    avg_velovalues = np.array(avg_velovalues)
    intensity_max_values = np.max(avg_velovalues)
    pixelnorm = [x / intensity_max_values for x in avg_velovalues]
    #print(pixelnorm.shape)

    fr = z[ : -sliding_size ]

    plt.plot(fr, pixelnorm)#, marker='o', color='r', ls='')
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Values')
    plt.legend(['Average Intensity'], loc='upper left')
    plt.show()


def customDataframe(frame_num, Desired_Points, x, y):
    Xdesire = np.zeros(shape=(len(Desired_Points), frame_num))
    Ydesire = np.zeros(shape=(len(Desired_Points), frame_num))
    #print(Xdesire.shape, Ydesire.shape)

    for points in range(len(Desired_Points)):
        for elements in range(frame_num):
            Xdesire [points][elements] = x[Desired_Points[points]][elements]
            Ydesire [points][elements] = y[Desired_Points[points]][elements]
    #print("x = " + str(x))
    #print("Xdesire = " + str(Xdesire))
    #print("y = " + str(y))
    #print("Ydesire = " + str(Ydesire))
    
    return Xdesire, Ydesire

def instantVelocity(frame_num, status, point_num, x, y, Xdesire, Ydesire, z, pixel, desired_points):
    ''' *******************************************************
    This function computes the instant velocity of the objects.
    ********************************************************'''

    
    objectVelocities=[[]]
    m = np.shape(Xdesire)

    # Determinig if we are seeking for 1 point velocity or all the points
    if status == 0 :
        traj_points = 1
   
    elif status == 1 :
        traj_points = m[0]

    else : 
        traj_points = number_of_points

        
    for points in range(traj_points):
    
        # set the first frame velocity with 0
        objectVelocities[points].append(0)
        for index in range(frame_num - 1):

            if status == 0 :             
                velocity = math.sqrt(( x[point_num][index + 1] - x[point_num][index])**2 + (y[point_num][index + 1] - y[point_num][index])**2)
                objectVelocities[points].append(velocity)

            elif status == 1 : 
                velocity = math.sqrt(( Xdesire[points][index + 1] - Xdesire[points][index])**2 + (Ydesire[points][index + 1] - Ydesire[points][index])**2)
                objectVelocities[points].append(velocity)

            else : 
                velocity = math.sqrt(( x[points][index + 1] - x[points][index])**2 + (y[points][index + 1] - y[points][index])**2)
                objectVelocities[points].append(velocity)

        objectVelocities.append([])

    # This part is to remove the last blank element of the objectvelocities 2D list 
    objectVelocities = [pick for pick in objectVelocities if len(pick) > 0 ]
    
    velocities = np.array(objectVelocities)

    # Now we need to find the maximum intensity of each object
    intensity_max_values = pixel.max(axis=1)
    pixelnorm = pixel / intensity_max_values[:, np.newaxis]
   
    # ok! now we need to normalize the intensity and velocity values by dividing all the values in different frames by max values of each object
    velocity_max_values = velocities.max(axis=1)
    velonorm = velocities / velocity_max_values[:, np.newaxis]
 
    if (status == 0 ):
        plt.plot(z, pixelnorm[point_num])
        plt.plot(z[:-1], velonorm[0])

    elif (status == 1) :
        for points in range (traj_points):
            plt.plot(z, pixelnorm[desired_points[points]])
            plt.plot(z[:-1], velonorm[points])

    else : 
        for points in range (traj_points):
            plt.plot(z, pixelnorm[points])
            plt.plot(z[:-1], velonorm[points])


    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Values')
    plt.legend(['Intensity', 'InstantVelocity'], loc='upper left')
    plt.show()

    return ( velocities, velonorm )

def avgSlidingVelocity(instvalues, status, frame_num, Single_flag, point_num, x, y, z, pixel, desired_points, sliding_size):

    slidingwindow = sliding_size
    avg_velovalues = [[]] 
    avgframe = frame_num - slidingwindow
    slidingpoint = 0

    m = len(desired_points)

    # Determinig if we are seeking for 1 point velocity or all the points
    if status == 0 :
        traj_points = 1
   
    elif status == 1 :
        traj_points = m

    else : 
        traj_points = number_of_points
 
    # computing the sliding average velocity of objects
    for points in range(traj_points):
        while avgframe > 0 :

            endpoint = slidingpoint + slidingwindow
            sumwin= sum(instvalues[points][slidingpoint: endpoint])
            avgvel = sumwin / slidingwindow
            avg_velovalues[points].append(avgvel)
            slidingpoint += 1
            avgframe -= 1
        avg_velovalues.append([])
        avgframe = frame_num - slidingwindow
        slidingpoint = 0


    # Removing the last blank element from the avg velo list 
    avg_velovalues = [ pick for pick in avg_velovalues if len(pick) > 0 ] 
    avg_velovalues = np.array(avg_velovalues)

    intensity_max_values = pixel.max(axis=1)
    pixelnorm = pixel / intensity_max_values[:, np.newaxis]
   
    # ok! now we need to normalize the intensity and average inst velocity values by dividing all the values in different frames by max values of each object
    velocity_max_values = avg_velovalues.max(axis=1)
    velonorm = avg_velovalues / velocity_max_values[:, np.newaxis]
    print (np.shape(velonorm))
    # print(velonorm[1])
    # print(velonorm[2])
   

    if (status == 0 ):
        plt.plot(z, pixelnorm[point_num])
        plt.plot(z[:(frame_num - slidingwindow)], velonorm[0])

    elif (status == 1) :
        for points in range (traj_points):
            plt.plot(z, pixelnorm[desired_points[points]])
            plt.plot(z[:(frame_num - slidingwindow)], velonorm[points])

    else : 
        for points in range (traj_points):
            plt.plot(z, pixelnorm[points])
            plt.plot(z[:(frame_num - slidingwindow)], velonorm[points])


    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Values')
    plt.legend(['Intensity', 'SlidingVelocity'], loc='upper left')
    plt.show()
    return velonorm

def computePointsAngels(NormalizedInstVal, frame_num, Single_flag, point_num, x, y, z, pixel):

    points_angles = [[]]
    AngularVelocities = [[]]
    
    if Single_flag :
        traj_points = 1

    else : 
        traj_points = number_of_points
    
        
    for points in range(traj_points):
    
        for index in range(frame_num + 1):

            if Single_flag :             
                point_angle = np.arctan( y[point_num][index] / x[point_num][index] )
                points_angles[points].append(point_angle)
            else : 
                point_angle = np.arctan( y[points][index] / x[points][index] )
                points_angles[points].append(point_angle)
        points_angles.append([])
    #print points_angles

    for obj_num in range (traj_points):
        
        plt.plot(z, np.degrees(points_angles[obj_num]))#, marker='o', color='r', ls='')
        plt.plot(z[:-1], np.degrees(NormalizedInstVal[obj_num]))

    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Values')
    plt.legend(['Point Angles', 'InstantVelocity'], loc='upper left')
    plt.show()

    # for obj_num in range (traj_points):
        
    #     plt.scatter(NormalizedInstVal[obj_num], np.degrees(points_angles[obj_num]))#, marker='o', color='r', ls='')
        
    # plt.xlabel('Normalized velocity')
    # plt.ylabel('Point Angles')
    # #plt.legend(['Point Angles', 'InstantVelocity'], loc='upper left')
    # plt.show()

    return points_angles

def computeInstantAngularVelo(InstantVelocity, points_angles, frame_num, Single_flag, point_num, x, y, z, pixel):

    pixelNorm = [[]]
    max_pixel_value = []
    Apoints = points_angles
    AngularVelocities=[[]]
    # Determinig if we are seeking for 1 point velocity or all the points
    if Single_flag :
        traj_points = 1

    else : 
        traj_points = Number_of_points
    
        
    for points in range(traj_points):
    
        # set the first frame velocity with 0
        AngularVelocities[points].append(0)

        for index in range(frame_num):
            angvelo = Apoints[points][index + 1] - Apoints[points][index]
            AngularVelocities[points].append(angvelo)
        AngularVelocities.append([])


    for obj_num in range (traj_points):
        
        max_pixel_value.append(max(pixel[obj_num]))        
        
        if max_pixel_value[obj_num] != 0 :
        
            pixelNorm[obj_num] = [ q / (max_pixel_value[obj_num] ) for q in pixel[obj_num]]
        else : 
            pixelNorm[obj_num] = 0 
        
        plt.plot(z, pixelNorm[obj_num])#, marker='o', color='r', ls='')
        plt.plot(z, np.degrees(AngularVelocities[obj_num]))
        plt.plot(z[:-1], InstantVelocity[obj_num])#, marker='o', color='r', ls='')
        
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Values')
    plt.legend(['Intensity', 'InstantVelocity', 'InstantAngularVelocity'], loc='upper left')
    plt.show()

    for obj_num in range (traj_points):
        
        plt.scatter(pixelNorm[obj_num], np.degrees(AngularVelocities[obj_num]), c = z)#, marker='o', color='r', ls='')
#    plt.xlabel('Normalized Intensity')
    plt.ylabel('Instant Angular Velocity')
    plt.colorbar()
    plt.show()

    return AngularVelocities

def computeSlidingAngularVelo(avgVelo, instAngVelo,frame_num, Single_flag, point_num, x, y, z, pixel):

    pixelNorm = [[]]
    max_pixel_value = []
    slidingWindow = 5
    AvgInstValues = [[]] 
    avgFrame = frame_num - slidingWindow
    slidingpoint = 0

    if Single_flag :
        traj_points = 1

    else : 
        traj_points = Number_of_points
    
        
    for points in range(traj_points):
        while avgFrame != 0 :
            endpoint = slidingpoint + slidingWindow
            SumWin= sum(instAngVelo[points][slidingpoint: endpoint])
            AvgVel = SumWin / slidingWindow
            AvgInstValues[points].append(AvgVel)
            slidingpoint += 1
            avgFrame -= 1
        AvgInstValues.append([])
    #print AvgInstValues

    for obj_num in range (traj_points):

        max_pixel_value.append(max(pixel[obj_num]))        
        
        if max_pixel_value[obj_num] != 0 :
        
            pixelNorm[obj_num] = [ q / (max_pixel_value[obj_num] ) for q in pixel[obj_num]]
        else : 
            pixelNorm[obj_num] = 0 
        
        plt.plot(z, pixelNorm[obj_num])#, marker='o', color='r', ls='')
        plt.plot(z[:(frame_num - slidingWindow)], np.degrees(AvgInstValues[obj_num]))
        plt.plot(z[:(frame_num - (slidingWindow))], avgVelo[obj_num])
        pixelNorm.append([])

    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Values')
    plt.legend(['Intensity', 'SlidingAngularVelocity', 'SlidingAverageVelocity'], loc='best')
    plt.show()


    for obj_num in range (traj_points):
        
        plt.scatter(pixelNorm[obj_num][:(frame_num - slidingWindow)], np.degrees(AvgInstValues[obj_num]))#, marker='o', color='r', ls='')
        
    plt.xlabel('Normalized Intensity')
    plt.ylabel('Sliding Angular Velocity')

    plt.show()

def rotational_3D_axis(frame_num, Single_flag, point_num, s, x, y, z, pixel):



    # Create a figure and a 3D Axes
    fig = plt.figure()
    ax = Axes3D(fig)

    def init():
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        # ax.scatter(xx, yy, zz, marker='o', s=20, c="goldenrod", alpha=0.6)

    def animate(i):
        ax.view_init(elev=25., azim=i)

    plt.figure(figsize=(7,7))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plt.axes(projection='3d')

    n = frame_num
    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

    plt.ion()

    fig.show()
    fig.canvas.draw()
    ax.view_init(60, 35)

    if Single_flag :
        traj_points = 1
        T = np.linspace(0,1,np.size(x[point_num]))

    else : 
        traj_points = Number_of_points
        T = np.linspace(0,1,np.size(x))

    for i in range(traj_points):
        for j in range(0, n-s, s):
            if Single_flag :             
                ax.plot(z[j: j+s+1],  y[point_num][j: j+s+1] ,x[point_num][j:j+s+1], linewidth =5, color = (0.0,(pixel[point_num][j]/max(pixel[point_num])),0.0))
            
            else : 
                ax.plot(z[j:j+s+1],  x[i][j:j+s+1], y[i][j:j+s+1] ,linewidth =2, color = (0.0,pixel[i][j]/max(pixel[i]) if max(pixel[i]) != 0 else pixel[i][j]/150 , 0.0))
            ax.set_xlabel('X')
            # ax.set_xlim(-20, 600)
            ax.set_ylabel('Y')
            # ax.set_ylim(-20, 600)
            ax.set_zlabel('Frames')
            ax.grid(False)
            #ax.view_init(abs(35) , j)
            fig.canvas.draw()        
            plt.savefig('contourff%d.jpg' % j, dpi=600)
            plt.pause(0.001)
    save('contourff', '3dtrajectory.mp4')


def calcAnglesHistogram(frame_num, status, point_num, desired_point, x, y, Xdesire, Ydesire):
   
    from math import atan2, degrees, pi, hypot
    m = Xdesire.shape

    if status == 0 :
        PointsNum = 1

    elif status == 1 :
        PointsNum = m[0]
    
    else :  
        PointsNum = number_of_points
    
    tetha_Table1 = np.zeros(shape=(PointsNum, desired_point))
    
    if ((frame_num - 1) < desired_point ): 
        tetha_Table2 = np.ones(shape=(1,1))
    else :
        tetha_Table2 = np.zeros(shape=(PointsNum, ((frame_num) - desired_point) + 1))
    #print (np.shape(tetha_Table1), frame_num)
    #print (np.shape(tetha_Table2), frame_num)
    
    slidingpoint = 20

    for objects in range(PointsNum):
        for frameNum in range(frame_num - slidingpoint):
            
            # Determining deltaX and deltaY in both cases ( single object or multiple objects)
            

            if status == 0 :
                X2 = x[point_num][frameNum + slidingpoint] 
                X1 = x[point_num][frameNum]
                Y2 = y[point_num][frameNum + slidingpoint] 
                Y1 = y[point_num][frameNum]

            elif status == 1 :
                X2 = Xdesire[objects][frameNum + slidingpoint] 
                X1 = Xdesire[objects][frameNum]
                Y2 = Ydesire[objects][frameNum + slidingpoint]
                Y1 = Ydesire[objects][frameNum]

            elif status == 2 :
                X2 = x[objects][frameNum + slidingpoint] 
                X1 = x[objects][frameNum]
                Y2 = y[objects][frameNum + slidingpoint]
                Y1 = y[objects][frameNum]

            V2 = (1, 0)
            V1 = (X2 - X1, Y2 - Y1)
            if (X1 == X2 and Y1 == Y2):
                tetha = 0
            else : 
                len1 = math.sqrt(np.dot(V1, V1))
                len2 = math.sqrt(np.dot(V2, V2))

                # inner_product = X1*X2 + Y1*Y2

                inner_product = (np.dot(V1, V2))

                #v[0]*w[1]-v[1]*w[0]
                determinant = V1[0]*V2[1] - V1[1]*V2[0]

                # len1 = math.hypot (X1, Y1)
                # len2 = math.hypot (X2, Y2)
                #print( X1, X2, Y1, Y2)
                #print( inner_product, len1, len2)
                rad = np.arccos(np.clip(inner_product / (len1 * len2), -1.0, 1.0))
#                rad = math.acos(inner_product / (len1 * len2))
                rad = rad*180 / math.pi

                #rad = rad * 180 / math.pi  
                if determinant < 0 :
                    tetha = rad #np.degrees(rad)
                else : 
                    tetha = 360 - rad

            #print (tetha, objects, frame_num)
            
            if frameNum < desired_point : 
                tetha_Table1[objects][frameNum] = tetha

            else : 
                tetha_Table2[objects][frameNum - desired_point]= tetha
            
        #print (' tetha_Table1 = ' + str (tetha_Table1))
        #print (' tetha_Table2 = ' + str (tetha_Table2))
    
    general1 = np.array(tetha_Table1)
    general2 = np.array(tetha_Table2)
    gen1 = np.zeros(desired_point * PointsNum)
    gen2 = np.zeros((frame_num - desired_point)*PointsNum)

    for i in range(PointsNum):
        for j in range(desired_point):
                gen1[ (i * (desired_point) + j) ] =  general1[i][j]

    for i in range(PointsNum):
        for j in range((frame_num - slidingpoint) - desired_point):
                gen2[ i * ((frame_num - slidingpoint) - desired_point) + j] =  general2[i][j]
    

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    n, bins, rectangles = ax1.hist(gen1, 90, density=True, stacked=True)
    ax1.set_xlabel('Theta (Degrees)')
    ax1.set_ylabel('Normalized Frequency')
    fig.canvas.draw()


    ax2 = fig.add_subplot(212)
    n, bins, rectangles = ax2.hist(gen2, 90, density=True, stacked=True)
    ax2.set_xlabel('Theta (Degrees)')
    ax2.set_ylabel('Normalized Frequency')
    fig.canvas.draw()
    plt.show()    

    bins = np.linspace(0, 360, 90)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    n, bins, rectangles = ax1.hist(gen1, bins, alpha = .5 , density=True, stacked=True, label = 'Before desired_time')
    n, bins, rectangles = ax1.hist(gen2, bins, alpha = .5 , density=True, stacked=True, label = 'After desired_time')
    ax1.set_xlabel('Theta (Degrees)')
    ax1.set_ylabel('Normalized Frequency')
    plt.legend(loc='upper right')
    plt.show() 

def anglesHistogram(frame_num, status, point_num, desired_point, x, y, Xdesire, Ydesire, Tracking_list):

    from math import acos, degrees, pi
    m = Xdesire.shape

    if status == 0 :
        PointsNum = 1

    elif status == 1 :
        PointsNum = m[0]
    
    else :  
        PointsNum = number_of_points
    
    tetha_Table1 = np.ones(shape=(PointsNum, desired_point))
    
    if ((frame_num - 1) < desired_point ): 
        tetha_Table2 = np.ones(shape=(1,1))

    else :
        tetha_Table2 = np.zeros(shape=(PointsNum, ((frame_num) - desired_point) + 1))

    #print (np.shape(tetha_Table1), frame_num)
    #print (np.shape(tetha_Table2), frame_num)

    tetha45multiples = np.zeros(PointsNum)
    slowmovements = np.zeros(PointsNum)
    slidingpoint = 20

    for objects in range(PointsNum):
        tethacounter = 0
        counter = 0

        for frameNum in range(frame_num - slidingpoint):
            
            # Determining deltaX and deltaY in both cases ( single object or multiple objects)
            
            if status == 0 :
                deltaX = x[point_num][frameNum + slidingpoint] - x[point_num][frameNum]
                deltaY = y[point_num][frameNum + slidingpoint] - y[point_num][frameNum]
            
            elif status == 1 :
                deltaX = Xdesire[objects][frameNum + slidingpoint] - Xdesire[objects][frameNum]
                deltaY = Ydesire[objects][frameNum + slidingpoint] - Ydesire[objects][frameNum]
            
            else :
                deltaX = x[objects][frameNum + slidingpoint] - x[objects][frameNum]
                deltaY = y[objects][frameNum + slidingpoint] - y[objects][frameNum]

            if (np.abs(deltaX) <= slidingpoint and np.abs(deltaY) <= slidingpoint) :
                counter += 1 
            
            if np.abs(deltaX) > 0.0 : 
                if ( deltaY == 0 and  deltaX < 0 ) :
                    tetha = 180
                else : 
                    rads = np.arctan2( deltaY , deltaX )
                    #rads = math.acos( deltaY / deltaX)
                    rads %= 2 * pi
                    tetha = np.rad2deg(rads)
                    #if deltaY < 0 :
                    #    tetha += 180
            elif ( deltaX == 0 and  deltaY < 0 ) : 
                tetha = 270
            elif ( deltaX == 0 and  deltaY > 0 ) :
                tetha = 90
            else : 
                tetha = 0 
                #continue
            #print (tetha, objects, frame_num)

            if frameNum < desired_point : 
                tetha_Table1[objects][frameNum] = tetha

            else : 
                tetha_Table2[objects][frameNum - desired_point]= tetha
            if tetha % 45 == 0 : 
                tethacounter += 1  
        tetha45multiples[objects] = tethacounter
        slowmovements[objects] = counter
        #print (' tetha_Table1 = ' + str (tetha_Table1))
        #print (' tetha_Table2 = ' + str (tetha_Table2))
    
    general1 = np.array(tetha_Table1)
    general2 = np.array(tetha_Table2)
    gen1 = np.ones(desired_point * PointsNum)
    gen2 = np.ones((frame_num - desired_point)*PointsNum)

    for i in range(PointsNum):
        for j in range(desired_point):
                gen1[ (i * (desired_point) + j) ] =  general1[i][j]

    for i in range(PointsNum):
        for j in range((frame_num - slidingpoint) - desired_point):
                gen2[ i * ((frame_num - slidingpoint) - desired_point) + j] =  general2[i][j]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    n, bins, rectangles = ax1.hist(gen1, 90, density=True, stacked=True)
    ax1.set_xlabel('Theta (Degrees)')
    ax1.set_ylabel('Frequency')
    fig.canvas.draw()

    ax2 = fig.add_subplot(212)
    n, bins, rectangles = ax2.hist(gen2, 90, density=True, stacked=True)
    ax2.set_xlabel('Theta (Degrees)')
    ax2.set_ylabel('Frequency')
    fig.canvas.draw()
    plt.show()    

    bins = np.linspace(0, 360, 90)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    n, bins, rectangles = ax1.hist(gen1, bins, alpha = .5 , density=True, stacked=True, label = 'Before desired')
    n, bins, rectangles = ax1.hist(gen2, bins, alpha = .5 , density=True, stacked=True, label = 'After  desired')
    ax1.set_xlabel('Theta (Degrees)')
    ax1.set_ylabel('Normalized Frequency')
    plt.legend(loc='upper right')
    plt.show()    


    return tetha45multiples, slowmovements, gen1, gen2, general1, general2
def RadialHistogram(frame_num, status, point_num, desired_point, general1, general2, gen1, gen2, gen1Mag, gen2Mag, general1Mag, general2Mag, Xdesire, Ydesire, Tracking_list):

    from math import acos, degrees, pi

    m = Xdesire.shape

    if status == 0 :
        PointsNum = 1

    elif status == 1 :
        PointsNum = m[0]
    
    else :  
        PointsNum = number_of_points

    Nbins = 90
    BinRatio = 360 / Nbins
    N = 90
    bottom = 8
    max_height = 6
    
    HBinsBefore = np.zeros(shape=(PointsNum,Nbins))
    HBinsAfter = np.zeros(shape=(PointsNum,Nbins))
    HBinSpecialBefore = np.zeros(shape=(Nbins))
    HBinSpecialAfter = np.zeros(shape=(Nbins))
    HBinsBeforeAll = np.zeros(Nbins)
    HBinsAfterAll = np.zeros(Nbins)

    HBinsWeightedBefore = np.zeros(shape=(PointsNum,Nbins))
    HBinsWeightedAfter = np.zeros(shape=(PointsNum,Nbins))
    HBinWeightedSpecialBefore = np.zeros(shape=(Nbins))
    HBinWeightedSpecialAfter = np.zeros(shape=(Nbins))
    HBinsWeightedBeforeAll = np.zeros(Nbins)
    HBinsWeightedAfterAll = np.zeros(Nbins)


    for i in range(PointsNum):
        for j in range(desired_point):
            m1 = int(general1[i][j] / BinRatio)
            HBinsBefore[i][m1] += 1
            HBinsWeightedBefore[i][m1] += general1Mag[i][j]
    
    for i in range(PointsNum):
        for j in range(frame_num - desired_point):
            m1 = int(general2[i][j] / BinRatio)
            HBinsAfter[i][m1] += 1
            HBinsWeightedAfter[i][m1] += general2Mag[i][j]

    for i in range(PointsNum * desired_point):
            m1 = int(gen1[i] / BinRatio)
            HBinsBeforeAll[m1] += 1 
            HBinSpecialBefore[m1] += 1
            HBinsWeightedBeforeAll[m1] += gen1Mag[i]
            HBinWeightedSpecialBefore[m1] += gen1Mag[i]

    for i in range(PointsNum * (frame_num - desired_point)):
            m1 = int(gen2[i] / BinRatio)
            HBinsAfterAll[m1] += 1 
            HBinSpecialAfter[m1] += 1
            HBinsWeightedAfterAll[m1] += gen2Mag[i]
            HBinWeightedSpecialAfter[m1] += gen2Mag[i]
    #print ('I ='+str(i))
    # print ('HBinsaFTER = ' +str(HBinsAfter[4]))
    # print ('General2 = ' +str(general2[4][:]))
    # print ('HBinsaFTERAll = ' +str(HBinsAfterAll))
    # print ('General2All = ' +str(HBinsBeforeAll))

    
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = (2*np.pi) / N
    # l= np.zeros(80)
    
    ax1 = plt.subplot(221, polar=True)
    objectRadial = point_num
    
    if status == 0 : 
        l1 = HBinsBefore[0]
        l2 = HBinsAfter[0]
        l3 = HBinsWeightedBefore[0]
        l4 = HBinsWeightedAfter[0]
        objectRadial = point_num
      
    elif status == 1 : 
        l1 = HBinSpecialBefore
        l2 = HBinSpecialAfter
        l3 = HBinWeightedSpecialBefore
        l4 = HBinWeightedSpecialAfter
        objectRadial = Tracking_list

    else : 
        l1 = HBinsBeforeAll
        l2 = HBinsAfterAll
        l3 = HBinsWeightedBeforeAll
        l4 = HBinsWeightedAfterAll
        objectRadial = ' (All) '

    ax1.set_xlabel('|Hist of angles obj#'+ str(objectRadial) +' Before desired')
    
    bars = ax1.bar(theta, l1, width=width, bottom=bottom)

    # Use custom colors and opacity
    radii = max_height*np.random.rand(N)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.85)

    # ax2 = plt.subplot(222, polar=True)
    # ax2.set_xlabel('|Hist of angles obj#'+ str(objectRadial) +' After desired')
    
    # bars = ax2.bar(theta, l2, width=width, bottom=bottom)

    # # Use custom colors and opacity
    # radii = max_height*np.random.rand(N)
    # for r, bar in zip(radii, bars):
    #     bar.set_facecolor(plt.cm.jet(r / 10.))
    #     bar.set_alpha(0.85)

    ax3 = plt.subplot(223, polar=True)
    ax3.set_xlabel('|Weighted Hist obj#'+ str(objectRadial) +' Before desired')
    
    bars = ax3.bar(theta, l3, width=width, bottom=bottom)

    # Use custom colors and opacity
    radii = max_height*np.random.rand(N)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.85)

    # ax4 = plt.subplot(224, polar=True)
    # ax4.set_xlabel('|Weighted Hist obj#'+ str(objectRadial) +' After desired')
    
    # bars = ax4.bar(theta, l4, width=width, bottom=bottom)

    # # Use custom colors and opacity
    # radii = max_height*np.random.rand(N)
    # for r, bar in zip(radii, bars):
    #     bar.set_facecolor(plt.cm.jet(r / 10.))
    #     bar.set_alpha(0.85)

    plt.show()
def objectsMagnitudes(frame_num, status, point_num, desired_point, x, y, Xdesire, Ydesire, tracking_list):

    m = Xdesire.shape

    if status == 0 :
        PointsNum = 1

    elif status == 1 :
        PointsNum = m[0]
    
    else :  
        PointsNum = number_of_points
    
    magnitude_Table1 = np.ones(shape=(PointsNum, desired_point))
    totalBefore = np.zeros(PointsNum)
    totalAfter = np.zeros(PointsNum)
    if ((frame_num - 1) < desired_point ): 
        magnitude_Table2 = np.ones(shape=(1,1))

    else :
        magnitude_Table2 = np.zeros(shape=(PointsNum, ((frame_num) - desired_point) + 1))

    #print (np.shape(magnitude_Table1), frame_num)
    #print (np.shape(magnitude_Table2), frame_num)

    slidingpoint = 20

    for objects in range(PointsNum):

        totalMagnitudeOf_obj_beforedesired = 0
        totalMagnitudeOf_obj_afterdesired = 0
        
        for frameNum in range(frame_num - slidingpoint):
            
            # Determining deltaX and deltaY in both cases ( single object or multiple objects)
            if status == 0 :             
                Magnitude = math.sqrt(( x[point_num][frameNum + slidingpoint] - x[point_num][frameNum])**2 + (y[point_num][frameNum + slidingpoint] - y[point_num][frameNum])**2)

            elif status == 1 : 
                Magnitude = math.sqrt(( Xdesire[objects][frameNum + slidingpoint] - Xdesire[objects][frameNum])**2 + (Ydesire[objects][frameNum + slidingpoint] - Ydesire[objects][frameNum])**2)

            else : 
                Magnitude = math.sqrt(( x[objects][frameNum + slidingpoint] - x[objects][frameNum])**2 + (y[objects][frameNum + slidingpoint] - y[objects][frameNum])**2)

            if Magnitude >= 50 : 
                Magnitude = 1
            #print (Magnitude, objects, frame_num)

            if frameNum < desired_point : 
                magnitude_Table1[objects][frameNum] = Magnitude
                totalMagnitudeOf_obj_beforedesired = totalMagnitudeOf_obj_beforedesired + Magnitude
            else : 
                magnitude_Table2[objects][frameNum - desired_point]= Magnitude
                totalMagnitudeOf_obj_afterdesired = totalMagnitudeOf_obj_afterdesired + Magnitude
        totalBefore[objects] = totalMagnitudeOf_obj_beforedesired
        totalAfter[objects] = totalMagnitudeOf_obj_afterdesired
    general1Mag = np.array(magnitude_Table1)
    general2Mag = np.array(magnitude_Table2)
    gen1Mag = np.ones(desired_point * PointsNum)
    gen2Mag = np.ones((frame_num - desired_point)*PointsNum)

    for i in range(PointsNum):
        for j in range(desired_point):
                gen1Mag[ (i * (desired_point) + j) ] =  general1Mag[i][j]

    for i in range(PointsNum):
        for j in range((frame_num - slidingpoint) - desired_point):
                gen2Mag[ i * ((frame_num - slidingpoint) - desired_point) + j] =  general2Mag[i][j]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    n, bins, rectangles = ax1.hist(gen1Mag, 60, density=True, stacked=True)
    ax1.set_xlabel('Magnitude( Pixel )')
    ax1.set_ylabel('Frequency')
    fig.canvas.draw()

    ax2 = fig.add_subplot(212)
    n, bins, rectangles = ax2.hist(gen2Mag, 60, density=True, stacked=True)
    ax2.set_xlabel('Magnitude( Pixel )')
    ax2.set_ylabel('Frequency')
    fig.canvas.draw()
    plt.show()    

    #===================================================================================
    #                                                                                  #
    #===================================================================================

    bins = np.linspace(0, 20, 20)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    n, bins, rectangles = ax1.hist(gen1Mag, bins, alpha = .5 , density=True, stacked=True, label = 'Before desired')
    n, bins, rectangles = ax1.hist(gen2Mag, bins, alpha = .5 , density=True, stacked=True, label = 'After desired')
    ax1.set_xlabel('Magnitude ( Pixel )')
    ax1.set_ylabel('Normalized Frequency')
    plt.legend(loc='upper right')
    plt.show()    

    bins = np.linspace(0, PointsNum, PointsNum)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(bins, totalBefore, marker='o', alpha = .5 , label = 'Before desired')
    ax1.scatter(bins, totalAfter,  marker='x', alpha = .5 , label = 'After desired')

    ax1.set_xlabel('Magnitude ( Pixel )')
    ax1.set_ylabel('object#')
    plt.legend(loc='upper right')
    plt.show()    

    N = PointsNum
    x1 = totalBefore.copy()
    x2 = totalAfter.copy()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    maxx = PointsNum #[max_object_show if max_object_show < 18 else 18]
    ## the data
    rects1 = ax.bar(ind, x1[:maxx], width, color='green')

    rects2 = ax.bar(ind + width, x2[:maxx], width, color='magenta')

    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0,max(max(x1),max(x2)) + 2)
    ax.set_ylabel('Magnitude ( total No. of pixel / video)')
    ax.set_title('Total Magnitude of each object before and After addition')
    xTickMarks = ['Object'+str(tracking_list[i]) for i in range(PointsNum)]
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)

    ## add a legend
    ax.legend( (rects1[0], rects2[0]), ('Total Magnitude of the object BEFORE desired', 'Total magnitude of the objectas AFTER the desired'), loc ='upper right' )

    plt.show()

    return gen1Mag, gen2Mag, general1Mag, general2Mag


def main(): 

    global number_of_points
    path = '/Users/mojtaba/Desktop/N/videos/'
    file = 'video3_normed.mp4'
    video_file = path + file
    out_video_file1 = 'normed_3channel.avi'
    out_video_file2 = 'Enhanced_normed_3channel.avi'
    frames_to_be_tracked = 118
    single_flag = False
    desired_muliple = False
    point_num = 0
    desired_traj_points = [0, 1, 2]#, 10, 7, 6, 8, 1, 2]
    desired_point = 118
    sliding_size = 10
    z = []
    for i in range (frames_to_be_tracked + 1):
        z.append(i)

    #===========================================================================================

    fetaure_set = Tracker( video_file, out_video_file1, frames_to_be_tracked)
    x, y, intensity, avg_frame_intensity, status, tracking_list = post_processing(fetaure_set, video_file, out_video_file2, frames_to_be_tracked, single_flag, desired_muliple, point_num, desired_traj_points)
    number_of_points = np.shape(x)[0]
    

    Xdesire, Ydesire = customDataframe(frames_to_be_tracked, desired_traj_points, x, y)
    # print(avg_frame_intensity)
    # print(z)

    Trajectory_2d_plot(x, y, True)
    average_frame_inetnsity_plot(avg_frame_intensity, z, frames_to_be_tracked + 1, sliding_size)
    instvalues, normalizedinstval = instantVelocity(frames_to_be_tracked, 0, point_num, x, y, Xdesire, Ydesire, z, intensity, desired_traj_points)    
    Trajectory_2d_TimeVarying( frames_to_be_tracked, single_flag, point_num , 1, x, y)
    avgvelo = avgSlidingVelocity(instvalues, 0, frames_to_be_tracked, single_flag, point_num, x, y, z, intensity, desired_traj_points, sliding_size)
    apoints = computePointsAngels(normalizedinstval, frames_to_be_tracked, True, point_num, x, y, z, intensity)
    instangvelo = computeInstantAngularVelo(normalizedinstval, apoints, frames_to_be_tracked, True, point_num, x, y, z, intensity)
 #   computeSlidingAngularVelo(avgvelo, instangvelo,frames_to_be_tracked, True, point_num, x, y, z, intensity)
#    rotational_3D_axis(frames_to_be_tracked, True, point_num, 1, x, y, z, intensity)
    calcAnglesHistogram(frames_to_be_tracked, status , point_num, desired_point, x, y, Xdesire, Ydesire)
    teth, slow, general1, general2, gen1, gen2 = anglesHistogram(frames_to_be_tracked, status, point_num, desired_point, x, y, Xdesire, Ydesire, tracking_list)
    gen1mag, gen2mag, general1mag, general2mag = objectsMagnitudes(frames_to_be_tracked, status, point_num, desired_point, x, y, Xdesire, Ydesire, tracking_list)
    RadialHistogram(frames_to_be_tracked, status, point_num, desired_point, gen1, gen2, general1, general2, gen1mag, gen2mag, general1mag, general2mag, Xdesire, Ydesire, tracking_list)
    RadialHistogram(frames_to_be_tracked, 0, point_num, desired_point, gen1, gen2, general1, general2, gen1mag, gen2mag, general1mag, general2mag, Xdesire, Ydesire, tracking_list)

if __name__ == "__main__":
    main()



    