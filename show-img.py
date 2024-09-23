import cv2
import numpy as np
import os
import random



class ImageViewer:
    def __init__(self, window_size=(800, 600), zoom=1.0):

        self.window_size = window_size
        self.zoom = zoom
        self.camera_pos = np.array([0, 0], dtype=np.float32)
        self.dragging = False
        self.last_mouse_pos = np.array([0, 0])
        self.images = []

        cv2.namedWindow('Image Viewer')
        cv2.setMouseCallback('Image Viewer', self.mouse_callback)



    # 鼠标回调函数，用于处理鼠标事件
    def mouse_callback(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下事件
            self.dragging = True
            self.last_mouse_pos = np.array([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:  # 鼠标移动事件
            delta = (np.array([x, y]) - self.last_mouse_pos) / self.zoom
            self.camera_pos -= delta
            self.last_mouse_pos = np.array([x, y])
        elif event == cv2.EVENT_LBUTTONUP:  # 左键抬起事件
            self.dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:  # 鼠标滚轮事件
            if flags > 0:
                self.zoom *= 2
            else:
                self.zoom /= 1.1

        center_after_zoom = self.camera_pos + np.array([self.window_size[0], self.window_size[1]]) / (2 * self.zoom)
        self.camera_pos += center_before_zoom - center_after_zoom
    def add_image(self, path, pos):
        self.images.append({'path': path, 'pos': pos, 'is_active': False, 'img': None})

    def run(self):
        while True:
            # 创建黑色背景
            view = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)

            # 计算相机视图范围
            top_left = self.camera_pos
            bottom_right = self.camera_pos + np.array([self.window_size[0], self.window_size[1]]) / self.zoom
            extended_top_left = top_left - np.array([self.window_size[0], self.window_size[1]]) / self.zoom
            extended_bottom_right = bottom_right + np.array([self.window_size[0], self.window_size[1]]) / self.zoom

            # 动态加载和卸载图像
            for image in self.images:
                pos = image['pos']
                img_top_left = pos
                img_bottom_right = pos + np.array([1000, 1000])  # 假设图像大小为1000x1000

                # 判断图像是否在视图范围内
                if (img_bottom_right[0] > extended_top_left[0] and img_top_left[0] < extended_bottom_right[0] and
                    img_bottom_right[1] > extended_top_left[1] and img_top_left[1] < extended_bottom_right[1]):
                    if not image['is_active']:
                        image['img'] = cv2.imread(image['path'])  # 加载图像
                        image['is_active'] = True
                        print('Load image:', image['path'])
                else:
                    if image['is_active']:
                        image['img'] = None  # 卸载图像
                        image['is_active'] = False

            # 渲染活动图像
            for image in self.images:
                if image['is_active']:
                    img = image['img']
                    pos = image['pos']
                    img_top_left = pos
                    img_bottom_right = pos + np.array([img.shape[1], img.shape[0]])

                    # 判断图像是否在视图范围内
                    if (img_bottom_right[0] > top_left[0] and img_top_left[0] < bottom_right[0] and
                        img_bottom_right[1] > top_left[1] and img_top_left[1] < bottom_right[1]):
                        view_pos = (pos - top_left) * self.zoom
                        view_pos = view_pos.astype(int)

                        x1 = max(0, view_pos[0])
                        y1 = max(0, view_pos[1])
                        x2 = min(self.window_size[0], view_pos[0] + int(img.shape[1] * self.zoom))
                        y2 = min(self.window_size[1], view_pos[1] + int(img.shape[0] * self.zoom))

                        img_x1 = max(0, -view_pos[0])
                        img_y1 = max(0, -view_pos[1])
                        img_x2 = img_x1 + (x2 - x1)
                        img_y2 = img_y1 + (y2 - y1)

                        img_resized = cv2.resize(img, (int(img.shape[1] * self.zoom), int(img.shape[0] * self.zoom)))

                        view[y1:y2, x1:x2] = img_resized[img_y1:img_y2, img_x1:img_x2]

            # 添加相机位置文本
            cv2.putText(view, f'Camera Pos: {self.camera_pos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # 添加相机位置文本
            cv2.putText(view, "press esc to exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # 显示图像
            cv2.imshow('Image Viewer', view)

            # 退出条件
            if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
                break
        cv2.destroyAllWindows()



if __name__ == "__main__":

    viewer = ImageViewer()


    viewer.add_image("./imgs/03_0001.JPG", np.array([0, 0]))
    viewer.add_image("./imgs/03_0002.JPG", np.array([-19.02805901*9, -98.53948212*9]))
    viewer.add_image("./imgs/03_0003.JPG", np.array([1000, -1000]))
    viewer.add_image("./imgs/03_0004.JPG", np.array([-10000, -1000]))

    # ============这里是一个加载文件夹中所有图像的例子================
    # image_dir = 'D:/Dataset/UAV_VisLoc_dataset/03/drone/'
    # image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.JPG', '.jpeg', '.png'))]
    # images = []
    # for path in image_paths:
    #     # 随机生成图像位置，确保每个图像最小相距1000以上
    #     pos = np.array([random.randint(0, 100000), random.randint(0, 100000)])
    #     viewer.add_image(path,pos)
    
    
    
    viewer.run()



class ImageViewer:
    def __init__(self, window_size=(800, 600), zoom=1.0):
        self.window_size = window_size
        self.zoom = zoom
        self.camera_pos = np.array([0, 0], dtype=np.float32)
        self.dragging = False
        self.last_mouse_pos = np.array([0, 0])
        self.images = []

        cv2.namedWindow('Image Viewer')
        cv2.setMouseCallback('Image Viewer', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse_pos = np.array([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            delta = (np.array([x, y]) - self.last_mouse_pos) / self.zoom
            self.camera_pos -= delta
            self.last_mouse_pos = np.array([x, y])
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            center_before_zoom = self.camera_pos + np.array([self.window_size[0], self.window_size[1]]) / (2 * self.zoom)
            if flags > 0:
                self.zoom *= 2
            else:
                self.zoom /= 1.1
            center_after_zoom = self.camera_pos + np.array([self.window_size[0], self.window_size[1]]) / (2 * self.zoom)
            self.camera_pos += center_before_zoom - center_after_zoom

    def add_image(self, path, pos):
        self.images.append({'path': path, 'pos': pos, 'is_active': False, 'img': None})

    def run(self):
        while True:
            view = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            top_left = self.camera_pos
            bottom_right = self.camera_pos + np.array([self.window_size[0], self.window_size[1]]) / self.zoom
            extended_top_left = top_left - np.array([self.window_size[0], self.window_size[1]]) / self.zoom
            extended_bottom_right = bottom_right + np.array([self.window_size[0], self.window_size[1]]) / self.zoom

            for image in self.images:
                pos = image['pos']
                img_top_left = pos
                img_bottom_right = pos + np.array([1000, 1000])

                if (img_bottom_right[0] > extended_top_left[0] and img_top_left[0] < extended_bottom_right[0] and
                    img_bottom_right[1] > extended_top_left[1] and img_top_left[1] < extended_bottom_right[1]):
                    if not image['is_active']:
                        image['img'] = cv2.imread(image['path'])
                        image['is_active'] = True
                        print('Load image:', image['path'])
                else:
                    if image['is_active']:
                        image['img'] = None
                        image['is_active'] = False

            for image in self.images:
                if image['is_active']:
                    img = image['img']
                    pos = image['pos']
                    img_top_left = pos
                    img_bottom_right = pos + np.array([img.shape[1], img.shape[0]])

                    if (img_bottom_right[0] > top_left[0] and img_top_left[0] < bottom_right[0] and
                        img_bottom_right[1] > top_left[1] and img_top_left[1] < bottom_right[1]):
                        view_pos = (pos - top_left) * self.zoom
                        view_pos = view_pos.astype(int)

                        x1 = max(0, view_pos[0])
                        y1 = max(0, view_pos[1])
                        x2 = min(self.window_size[0], view_pos[0] + int(img.shape[1] * self.zoom))
                        y2 = min(self.window_size[1], view_pos[1] + int(img.shape[0] * self.zoom))

                        img_x1 = max(0, -view_pos[0])
                        img_y1 = max(0, -view_pos[1])
                        img_x2 = img_x1 + (x2 - x1)
                        img_y2 = img_y1 + (y2 - y1)

                        img_resized = cv2.resize(image['img'], (int(image['img'].shape[1] * self.zoom), int(image['img'].shape[0] * self.zoom)))

                        view[y1:y2, x1:x2] = img_resized[img_y1:img_y2, img_x1:img_x2]

            cv2.putText(view, f'Camera Pos: {self.camera_pos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, "press esc to exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Image Viewer', view)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()