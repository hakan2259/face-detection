import cv2
from facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

image = cv2.imread( 'bill.jpg' , cv2.IMREAD_UNCHANGED)


face_locations, face_names = sfr.detect_known_faces(image)
for face_loc, name in zip(face_locations, face_names):
    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

    cv2.putText(image, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("image", image)

key = cv2.waitKey(0)
cv2.destroyAllWindows()