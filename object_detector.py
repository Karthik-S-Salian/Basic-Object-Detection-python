import cv2
import numpy as np
import time
import requests

def hsv_to_rgb(h, s=1, v=1):
        h=h/360
        if s == 0.0: v *= 255; return (v, v, v)
        i = int(h * 6.)
        f = (h * 6.) - i
        p, q, t = int(255 * (v * (1. - s))), int(255 * (v * (1. - s * f))), int(255 * (v * (1. - s * (1. - f))))
        v *= 255
        i %= 6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)

class Detector():
    def __init__(self,config_path,model_path:str,class_names_path:str) -> None:
        self.config_path=config_path
        self.model_path=model_path
        self.class_names_path=class_names_path
        self.cap=None
        self.is_url=False
        self.DEFAULT_FPS=10
        self.fps=self.DEFAULT_FPS
        self.iter_count=0
        self.updating=False
        self.success=False

        ###########################################################################

        # initialising network

        self.net=cv2.dnn_DetectionModel(self.model_path,self.config_path)
        self.net.setInputSize(320,320)  # trained image size 

        # normalizing input to [-1,1]
        mean=2.0/255.0  # rgb 0-255
        self.net.setInputScale(mean)  
        self.net.setInputMean((mean,mean,mean))

        self.net.setInputSwapRB(True)  # in opencv image format is bgr which here changed to rgb

        ###########################################################################

        self.read_classes()


    def set_video_source(self,source):
        self.updating=True
        self.video_path=source
        self.validate_source()

    def validate_source(self):
        if isinstance(self.video_path,str):
            if self.video_path.startswith(("http:","https:")):
                self.is_url=True
                self.success=True
                self.updating=False
                return
            if not self.video_path.endswith((".mp4",".avi",".jpg",".png",".jfif")):
                self.cap=None
                self.video_path=None
                self.success=False
                self.updating=False
                return 

        self.cap=cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.cap=None
            self.video_path=None
            self.success=False
            self.updating=False
            return
        self.success=True
        self.fps=self.cap.get(cv2.CAP_PROP_FPS)
        self.updating=False


    def read_classes(self):
        with open(self.class_names_path,"r") as file_handle:
            self.classes_list = file_handle.read().splitlines()

             # endpoint excluded so 360+1
            self.color_list=list(map(hsv_to_rgb,((np.linspace(0,361,num=len(self.classes_list))*20)%361).tolist()))

    def get_image(self):
        if self.video_path is not None:
            if self.is_url:
                try:
                    img_resp = requests.get(self.video_path)
                except:
                    return False,None
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                return True,cv2.imdecode(img_arr, -1)
            else:
                if self.cap:
                    return self.cap.read()
        return False,None

    def detect(self,image):
        class_label_ids,confidences,bounding_boxes = self.net.detect(image,confThreshold=0.5)

        bounding_box_index = cv2.dnn.NMSBoxes(bounding_boxes,confidences,score_threshold=.5,nms_threshold=.2)

        
        for index in bounding_box_index:
            class_label_index = class_label_ids[index]
            class_color = self.color_list[class_label_index]

            x,y,w,h = bounding_boxes[index]
            cv2.rectangle(image,(x,y),(x+w,y+h),color=class_color,thickness=2)

            display_text ="{}:{:.0f}%".format(self.classes_list[class_label_index],confidences[index]*100)
            cv2.putText(image,display_text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,color=class_color,thickness=2)

        return image



    def on_video(self,callback,detect,keep_thread_alive,size,pause):

        start_time=time.time()

        while keep_thread_alive():
            self.iter_count+=1
            if (not pause()) and self.success and not self.updating:
                self.success,image=self.get_image()
                
                if not self.success:
                    self.fps=self.DEFAULT_FPS
                    continue
                size_x,size_y=size()
                image=cv2.resize(image,(size_x,size_y))
                if detect():
                    image=self.detect(image)


                current_time=time.time()
                delta_time=current_time-start_time
                delta_time = delta_time if delta_time>0 else 1
                print("real : '{}', actual {}".format(self.cap.get(cv2.CAP_PROP_FPS),1.0/delta_time))
                cv2.putText(image,'FPS: {:.2f}'.format(1.0/delta_time),(size_x-200,50),cv2.FONT_HERSHEY_PLAIN,2,color=[0,255,0],thickness=2)

                start_time=current_time

                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 0)
                
                callback(image)