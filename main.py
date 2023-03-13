from kivy.app import  App
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import Clock , ObjectProperty
from kivy.graphics.texture import Texture
from kivy.animation import Animation
from threading import Thread
import os
from kivy.uix.popup import Popup
from tkinter import filedialog
from object_detector import Detector


class LoadDialog(Popup):
    load_file = ObjectProperty(None)
    load_url = ObjectProperty(None)
    load_webcam = ObjectProperty(None)
    cancel=ObjectProperty(None)



class MainLayout(RelativeLayout):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.detector_thread=None
        self.keep_thread_alive=True
        self.detect=False
        self.pause=False
        self.drawed=False
        self.draw_duration=.3
        config_path=os.path.join("resources\\model_data","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
        model_path=os.path.join("resources\\model_data","frozen_inference_graph.pb")
        class_names_path=os.path.join("resources\\model_data","coco.names")
        self.detector=Detector(config_path,model_path,class_names_path)

    def on_parent(self,j,parent):
        if parent:
            self.imageView=self.ids.imageView
            self.nav_drawer=self.ids.nav_drawer
            self.start_thread()
        else:
            self.stop_thread()

    def update_image(self,image):
        Clock.schedule_once(lambda dt: self.update_image_main_thread(image))

    def update_image_main_thread(self,image):
        texture = Texture.create(size=(image.shape[1], image.shape[0]))
        texture.blit_buffer(image.flatten(), colorfmt='rgb', bufferfmt='ubyte')
        self.imageView.texture=texture


    def start_thread(self):
        if (not self.detector_thread) or  (not self.detector_thread.is_alive()):
            self.detector_thread=Thread(target=self.detector.on_video,args=(self.update_image,lambda : self.detect,lambda: self.keep_thread_alive ,lambda:self.imageView.size,lambda:self.pause),daemon=True)
            self.detector_thread.start()


    def stop_thread(self):
        if self.detector_thread and  self.detector_thread.is_alive():
            self.keep_thread_alive=False
            self.detector_thread.join()

    def on_draw_request(self):
        if not self.drawed:
            self.drawed=True
            animation = Animation(pos=(0, 0),duration=self.draw_duration)
            animation.start(self.nav_drawer) 

    def on_touch_down(self, touch):
        if self.drawed:
            if touch.x>self.nav_drawer.width:
                animation = Animation(pos=(-self.nav_drawer.width, 0),duration=self.draw_duration)
                self.drawed = False
                animation.start(self.nav_drawer)
        return super().on_touch_down(touch)

    def on_new_prompt(self):
        self.pause=True
        self._popup = LoadDialog(title="NEW",size_hint=(0.9, 0.9),load_file=self.set_file,load_url=self.set_url, load_webcam=self.set_webcam,cancel=self.dismiss_popup)
        self._popup.open()

    def set_file(self):
        file=filedialog.askopenfilename()
        if file:
            self.detector.set_video_source(file)
        self._popup.dismiss()

    def set_url(self,url):
        self.detector.set_video_source(url)
        self._popup.dismiss()

    def set_webcam(self):
        self.detector.set_video_source(0)
        self._popup.dismiss()

    def dismiss_popup(self):
        print("dismiss")
        self.pause=False


if __name__=="__main__":

    class ObjectDetector(App):
        pass

    ObjectDetector().run()