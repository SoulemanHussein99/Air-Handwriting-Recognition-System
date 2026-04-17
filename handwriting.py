import random

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# load trained CNN model
model = tf.keras.models.load_model('model.h5')

clf = mp.solutions.hands
mp_hand = clf.Hands(max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils

chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
draw_mode = False
text_data = []
points = []
start_x = 0
start_y = 0
cursor_x = start_x
cursor_y = start_y
cursor_index = 0

prev_x, prev_y = 0, 0
prev_x12, prev_y12 = 0, 0
prev_x16, prev_y16 = 0, 0
prev_x4, prev_y4 = 0, 0 
alpha = 0.2  # smoothing factor


camera = cv2.VideoCapture(0)


while True:
    success, frame = camera.read()
    if not success:
        break
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    frame_masked = np.zeros_like(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = mp_hand.process(frame_rgb)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
                
                draw.draw_landmarks(frame_masked, hand, clf.HAND_CONNECTIONS)
                lm4 = hand.landmark[4]
                lm8 = hand.landmark[8]
                lm12 = hand.landmark[12]
                lm16 = hand.landmark[16]
                cx16, cy16 = int(lm16.x * w), int(lm16.y * h)
                cx16 = int(alpha * cx16 + (1 - alpha) * prev_x16)
                cy16 = int(alpha * cy16 + (1 - alpha) * prev_y16)
                prev_x16, prev_y16 = cx16, cy16
                cx12, cy12 = int(lm12.x * w), int(lm12.y * h)
                cx12 = int(alpha * cx12 + (1 - alpha) * prev_x12)
                cy12 = int(alpha * cy12 + (1 - alpha) * prev_y12)
                prev_x12, prev_y12 = cx12, cy12
                cx, cy = int(lm8.x * w), int(lm8.y * h)
                cx = int(alpha * cx + (1 - alpha) * prev_x)
                cy = int(alpha * cy + (1 - alpha) * prev_y)
                prev_x, prev_y = cx, cy
                cx4, cy4 = int(lm4.x * w), int(lm4.y * h)
                cx4 = int(alpha * cx4 + (1 - alpha) * prev_x4)
                cy4 = int(alpha * cy4 + (1 - alpha) * prev_y4)
                prev_x4, prev_y4 = cx4, cy4
                    
                # fingers status
                # ------------------------
                fingers = [
                    0 if hand.landmark[8].y > hand.landmark[6].y else 1,
                    0 if hand.landmark[12].y > hand.landmark[10].y else 1,
                    0 if hand.landmark[16].y > hand.landmark[14].y else 1,
                    0 if hand.landmark[20].y > hand.landmark[18].y else 1
                ]

                # ------------------------
                # moving cursor
                # ------------------------
                if not draw_mode:
                    for i, (char, x, y) in enumerate(text_data):
                        if abs(cx - x) < 25 and abs(cy - y) < 25:
                            cursor_index = i
                            cursor_x, cursor_y = x-20, y
                           

                
                # Draw Mode
                # ------------------------
                draw_mode = fingers[0]==1 and fingers[1]==1 and sum(fingers[2:])==0 and abs(cx - cx12)<20 and abs(cy - cy12)<20

                if draw_mode:
                    # glowing
                    #-----------------
                    cv2.line(frame_masked, (prev_x, prev_y), (cx, cy), (255, 180 + int(3*random.random()), 255), 6)
                    # light shine
                    cv2.circle(frame_masked, (cx, cy), 10 + int(3*random.random()), (255, 200 , 255), -1)
                   
                    # cursor location
                    #-------------------
                    points.append((cx,cy))
                    
                    # draw the line
                    #----------------
                    for i in range(1,len(points)):
                        cv2.line(frame_masked, points[i-1], points[i], (255,255,255), 5)
                        
                # prediction  Mode  
                # ------------------------        
                if  not draw_mode and len(points)>5:
                    canvas = np.zeros((h, w), dtype=np.uint8)
                    for i in range(1, len(points)):
                        cv2.line(canvas, points[i-1], points[i], 255, 5)
                    # 
                    canvas = cv2.GaussianBlur(canvas, (3,3), 0)   
                    canvas = cv2.flip(canvas, 1)
                    
                    # crop the draw
                    
                    # pixels that have drawing 
                    # (xs is for lightning pixel for x position)(xy is for lightning pixel for y position)
                    ys, xs = np.where(canvas > 0)

                    if len(xs) > 0 and len(ys) > 0:
                        x_min, x_max = xs.min(), xs.max()
                        y_min, y_max = ys.min(), ys.max()
                        
                        # taking the part that has draws
                        crop = canvas[y_min:y_max, x_min:x_max]

                        #  aspect ratio with saving resize
                        crop = cv2.resize(crop, (20, 20))

                        #  put it in the text  (28x28)
                        new_img = np.zeros((28,28), dtype=np.uint8)
                        # putting the draw in the picture from 4pixel to 24pixel
                        new_img[4:24, 4:24] = crop

                        img = new_img
                    else:
                        continue
                    img = img.astype('float32') / 255.0
                    img = img.reshape(1, 28, 28, 1)
                    pred = model.predict(img, verbose=0)[0]
                    conf = np.max(pred)
                    char_idx = np.argmax(pred)
                    
                    
                    new_char = chars[char_idx]
                    
                    # Insert character
                    #-----------------------
                    text_data.insert(cursor_index, (new_char, cursor_x, cursor_y))
                    cursor_index += 1
                    cursor_x += 30
                    for i in range(cursor_index, len(text_data)):
                        char, x, y = text_data[i]
                        text_data[i] = (char, x + 30, y)

                    points.clear()
           


                # space
                #----------------------
                if sum(fingers)== 3 and not draw_mode and abs(cx4 - cx12)<25 and abs(cy4 - cy12)<25:
                    text_data.insert(cursor_index, (" ", cursor_x, cursor_y))
                    cursor_index += 1
                    cursor_x += 30  
                    time.sleep(0.3)
                
                # Remove
                #----------------------
                if sum(fingers)== 3 and not draw_mode and abs(cx4 - cx16)<20 and abs(cy4 - cy16)<20:
                    if cursor_index >0 and len(text_data)>0:
                        cursor_index = min(cursor_index, len(text_data))
                        cursor_index -= 1
                        text_data.pop(cursor_index)
                        for i in range(cursor_index, len(text_data)):
                            char, x, y = text_data[i]
                            text_data[i] = (char, x - 30, y)
                                
                                # update cursor
                                #------------------
                        if cursor_index > 0:
                             cursor_x, cursor_y = text_data[cursor_index -1][1] + 30, text_data[cursor_index -1][2]
                        else:
                            cursor_x, cursor_y = start_x, start_y
                            time.sleep(0.5)
                    
                # new line
                if sum(fingers)== 0 :
                        cursor_y += 40
                        cursor_x=start_x
                        cursor_index = len(text_data)
                        time.sleep(0.5)
                    
       
        # showing text
        # ------------------------            
        for char, x, y in text_data:
            cv2.putText(frame_masked, char.lower(), (x,y+20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

       
        # drawing cursor 
        # ------------------------
        cv2.line(frame_masked, (cursor_x+10, cursor_y),
                                (cursor_x+10, cursor_y+20), (0,255,0), 2)

    cv2.imshow("Hand Writing AI - Cursor Mode", frame_masked)
                    
                    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        current_y =0
        final_text =""
        for char, x, y in text_data:
            if y > current_y:
                final_text += "\n"
                current_y = y
            final_text += char
        with open("output.txt", "w") as f:
            f.write(final_text.lower())    
        break
camera.release()
cv2.destroyAllWindows()
