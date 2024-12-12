import cv2
from multiprocessing.connection import Client


faceCascade = cv2.CascadeClassifier("/home/jann/maoarm/cascades/haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
video_capture.isOpened()

width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

conn = Client(('localhost', 6282))

try:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw a rectangle around the faces
        (lx, ly, lw, lh) = (width/2,height/2,0,0)
        largest_size = 0
        for (x, y, w, h) in faces:
            if w * h > largest_size:
                (lx, ly, lw, lh) = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (x+int(w/2),y+int(h/2)), radius=3, color=(0,0,255))
        
        # convert the coor so that 0,0 is in the center
        (lx, ly) = (lx - width/2, ly - height/2)

        # return the middle coordinate of the face
        (lx, ly) = ((lx+int(lw/2),ly+int(lh/2)))
        conn.send((lx, -ly, lw, lh, width, height))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
finally:
    conn.send('close')
    conn.close()
