# import loguru
import yaml
import numpy as np
import cv2
import torch
import warnings


class OpenCVFisheyeModel(object):
    def __init__(self, device=None, file_name=None):
        assert device is not None, f"device should be cpu or one of cuda:?"
        self.device = device

        if file_name is not None:
            self.load_yaml(file_name)
        else:
            warnings.warn(f"need use load_yaml func to init camera parameters...")

    def load_yaml(self, file_name: str):
        with open(file_name, 'r') as f:
            data = yaml.safe_load(f)
        self.image_width = data["image_width"]
        self.image_height = data["image_height"]
        self.camera_matrix = np.array(data["camera_matrix"]["data"],
                                      dtype=np.float32).reshape(data["camera_matrix"]["rows"],
                                                                data["camera_matrix"]["cols"])
        self.distortion_model = data["distortion_model"]
        self.distortion_coefficients = np.array(data["distortion_coefficients"]["data"],
                                                dtype=np.float32).reshape(data["distortion_coefficients"]["rows"],
                                                                          data["distortion_coefficients"]["cols"])

        if "r_mat" in data.keys():
            pass
        else:
            self.r_mat = np.eye(3, dtype=np.float32)

        if "t_vec" in data.keys():
            pass
        else:
            self.t_vec = np.zeros((3, 1), dtype=np.float32)

        self.r_vec, _ = cv2.Rodrigues(self.r_mat)

        # loguru.logger.debug(f"K\n{self.camera_matrix}\n"
        #                     f"D\n{self.distortion_coefficients}\n"
        #                     f"R\n{self.r_mat}\n"
        #                     f"r\n{self.r_vec}\n"
        #                     f"t\n{self.t_vec}")

        self.image_width = torch.tensor(self.image_width, device=self.device)
        self.image_height = torch.tensor(self.image_height, device=self.device)
        self.camera_matrix = torch.from_numpy(self.camera_matrix).to(self.device)
        self.distortion_coefficients = torch.from_numpy(self.distortion_coefficients).to(self.device)
        self.r_mat = torch.from_numpy(self.r_mat).to(self.device)
        self.t_vec = torch.from_numpy(self.t_vec).to(self.device)
        self.r_vec = torch.from_numpy(self.r_vec).to(self.device)

    @torch.no_grad()
    def project_points(self, points):
        '''
        :param points: [..., 3] world points (torch tensor)
        :return pixels: [..., 2] pixel points on src fisheye image (torch tensor)
        '''
        ori_shape = points.shape
        points = points.to(self.device).reshape(-1, 3)
        points = self.r_mat @ points.T + self.t_vec
        a = points[0] / points[2]
        b = points[1] / points[2]
        r_2 = a ** 2 + b ** 2
        r = torch.sqrt(r_2)
        theta = torch.atan(r)
        theta_d = theta * (
                1 + self.distortion_coefficients[0, 0] * (theta ** 2) + self.distortion_coefficients[0, 1] * (
                theta ** 4) + self.distortion_coefficients[0, 2] * (theta ** 6) + self.distortion_coefficients[0, 3] * (
                        theta ** 8))
        x = theta_d / r * a
        y = theta_d / r * b
        pixels = torch.stack([x, y, torch.ones_like(x)])
        pixels = self.camera_matrix @ pixels
        pixels = pixels[:2, ...].T
        pixels = pixels.reshape(*ori_shape[:-1], 2)

        return pixels

    @torch.no_grad()
    def init_camera_map(self):
        v, u = torch.meshgrid([torch.arange(0, self.image_height), torch.arange(0, self.image_width)],
                              indexing="ij")
        world_pixels = torch.stack([u, v, torch.ones_like(v)], dim=-1).to(self.device).float().reshape(-1, 3)
        world_pixels = self.r_mat.T @ (torch.linalg.inv(self.camera_matrix) @ world_pixels.T - self.t_vec)
        world_pixels = world_pixels.T
        world_pixels = world_pixels.reshape(self.image_height, self.image_width, 3)

        map_pixels = self.project_points(world_pixels)
        map_pixels[..., 0] /= self.image_width
        map_pixels[..., 1] /= self.image_height
        map_pixels *= 2.
        map_pixels -= 1.
        self.undistort_map = map_pixels

    @torch.no_grad()
    def undistort_image(self, image):
        image = torch.from_numpy(image.astype(np.float32)).to(self.device)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)

        undistort_map = self.undistort_map.unsqueeze(0)
        image = torch.nn.functional.grid_sample(image, undistort_map, align_corners=True)
        image = image.squeeze().permute(1, 2, 0)

        return image.cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def undistort_images(self, images: list):
        images = torch.stack([torch.from_numpy(image.astype(np.float32)).to(self.device) for image in images], dim=0)
        images = images.permute(0, 3, 1, 2)
        undistort_map = self.undistort_map.unsqueeze(0).repeat(images.shape[0], 1, 1, 1)
        images = torch.nn.functional.grid_sample(images, undistort_map, align_corners=True)
        images = images.permute(0, 2, 3, 1)

        return [image.cpu().numpy().astype(np.uint8) for image in images]


if __name__ == "__main__":
    ost = "data/ost.yaml"
    fisheye_cam = OpenCVFisheyeModel(device="cpu", file_name=ost)
    # fisheye_cam.load_yaml(ost)
    fisheye_cam.init_camera_map()
    image = cv2.imread("data/test.jpg")
    undistort_image = fisheye_cam.undistort_image(image)
    cv2.namedWindow("src", cv2.WINDOW_NORMAL)
    cv2.imshow("src", image)
    cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
    cv2.imshow("dst", undistort_image)
    cv2.waitKey()
