from __future__ import division
import torch
import random
import numpy as np
#from scipy.misc import imresize
import scipy
import scipy.ndimage
import numbers
import collections
from itertools import permutations

from numpy.ma.core import indices


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images

class EnhancedCompose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.abc.Sequence):
                assert isinstance(img, collections.abc.Sequence) and len(img) == len(
                    t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    #传进来几个数据，就用几个Transform来处理
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img

class Merge(object):
    """Merge a group of images
    """
    def __init__(self, axis=-1):
        self.axis = axis
    def __call__(self, images):
        if isinstance(images, collections.Sequence) or isinstance(images, np.ndarray):
            assert all([isinstance(i, np.ndarray)
                        for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s == shapes[0] for s in shapes]
                       ), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, images):
        for tensor in images:
            # check non-existent file
            if _is_tensor_image is False:
                continue
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images

class ArrayToTensorNumpy(object):
    """Converts a list of numpy.ndarray (H x W x C) to torch.FloatTensor of shape (C x H x W) """
    def __call__(self, data_arrays):
        img=data_arrays[0]
        data_arrays[0]=torch.from_numpy(img.transpose((2, 0, 1)))
        data_arrays[1]=torch.from_numpy(data_arrays[1])
        data_arrays[2]=torch.from_numpy(data_arrays[2])
        return data_arrays

class RandomCropNumpy(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, random_state=np.random,fix_crop=False):
        self.fix_crop = fix_crop
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.random_state = random_state
    def __call__(self, data_array):
        #data_dict中包含了原始的图像数据以及相应的点和距离标签
        #对图像进行裁剪，并且筛掉不满足裁剪条件的关键点
        img=data_array[0]
        results  = []
        h,w = img.shape[:2]
        th, tw = self.size
        if w == tw and h == th:
            return data_array
        elif h == th:
            if self.fix_crop:
                x1 = (w-tw)//2#self.random_state.randint(0, w - tw)
                y1 = 0
            else:
                x1 = self.random_state.randint(0, w - tw)
                y1 = 0
        elif w == tw:
            if self.fix_crop:
                x1 = 0
                y1 = (h - th)//2
            else:
                x1 = 0
                y1 = self.random_state.randint(0, h - th)
        else:
            x1 = self.random_state.randint(0, w - tw)
            y1 = self.random_state.randint(0, h - th)
        x2=x1 + tw
        y2=y1 + th
        img=data_array[0]
        point_arrays=data_array[1]
        distance_arrays=data_array[2]
        visible_mask=data_array[3]
        depth_array=data_array[4]
        depth_mask=data_array[5]
        #origin_img=data_array[6]

        data_array[0]=img[y1:y2, x1:x2, :]
        data_array[4]=depth_array[y1:y2, x1:x2]
        data_array[5]=depth_mask[y1:y2, x1:x2]
        #data_array[6]=origin_img[y1:y2, x1:x2, :]
        for idx,point in enumerate(point_arrays):
            if (not (x1<=point[0]<=x2 and x1<=point[2]<=x2 )) or (not (y1<=point[1]<=y2 and y1<=point[3]<=y2 )):
                #如果说这个点的x,y坐标不在裁剪后的范围就丢弃
                point_arrays[idx]=point_arrays[idx]*0
                visible_mask[idx]=0
            else:
                #如果在裁剪后的范围，
                point[np.array([0,2])]-=x1
                point[np.array([1,3])]-=y1

        return data_array

class RandomColor(object):
    """Random brightness, gamma, color, channel on numpy.ndarray (H x W x C) globally"""
    def __init__(self, multiplier_range=(0.9, 1.1), brightness_mult_range=(0.9, 1.1), random_state=np.random, dataset = 'KITTI'):
        assert isinstance(multiplier_range, tuple)
        self.multiplier_range = multiplier_range
        self.brightness_mult_range = brightness_mult_range
        self.random_state = random_state
        self.indices = list(permutations(range(3),3))
        self.indices_len = len(self.indices)
        self.dataset = dataset
    def __call__(self, image):
        if self.dataset == 'KITTI':
            if random.random() < 0.5:
                gamma_mult = self.random_state.uniform(self.multiplier_range[0],
                                                 self.multiplier_range[1])
                imgOut = image**gamma_mult
                brightness_mult = self.random_state.uniform(self.brightness_mult_range[0],
                                                        self.brightness_mult_range[1])
                imgOut = imgOut*brightness_mult
                color_mult = self.random_state.uniform(self.multiplier_range[0],
                                                 self.multiplier_range[1], size=3)
                result = np.stack([imgOut[:,:,i]*color_mult[i] for i in range(3)],axis=2)
            else:
                result = image
        else:
            if random.random() < 0.5:
                gamma_mult = self.random_state.uniform(self.multiplier_range[0],
                                                 self.multiplier_range[1])
                imgOut = image**gamma_mult
                brightness_mult = self.random_state.uniform(self.brightness_mult_range[0],
                                                        self.brightness_mult_range[1])
                imgOut = imgOut*brightness_mult
                color_mult = self.random_state.uniform(self.multiplier_range[0],
                                                 self.multiplier_range[1], size=3)
                result = np.stack([imgOut[:,:,i]*color_mult[i] for i in range(3)],axis=2)
            else:
                result = image
        if random.random() < 0.5:
            ch_pair = self.indices[self.random_state.randint(1, self.indices_len - 1)]
            result = result[:,:,list(ch_pair)]
        if isinstance(image, np.ndarray):
            return np.clip(result, 0, 1)
        else:
            raise Exception('unsupported type')

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""
    def __call__(self, data_array):

        data_array[0]
        if random.random() < 0.5:
            #水平翻转图像，调整关键点位置同时
            data_array[0]=np.fliplr(data_array[0]).copy()
            #同时翻转掩膜图以及深度图
            data_array[4]=np.fliplr(data_array[4]).copy()
            data_array[5]=np.fliplr(data_array[5]).copy()
            #proper 相关位置点
            proper_point_array=data_array[1]
            #existed 相关位置点
            existed_point_array=data_array[-1]
            h,d,_=data_array[0].shape


            proper_point_array[:,[0,2]]=d-proper_point_array[:,[0,2]]-1
            existed_point_array[:,[0,2]]=d-existed_point_array[:,[0,2]]-1
            data_array[1]=proper_point_array
            data_array[-1]=existed_point_array
        return data_array

class RandomAffineZoom(object):
    def __init__(self, scale_range=(1.0, 1.5), random_state=np.random):
        assert isinstance(scale_range, tuple)
        self.scale_range = scale_range
        self.random_state = random_state

    def __call__(self, image):
        scale = self.random_state.uniform(self.scale_range[0],
                                          self.scale_range[1])
        if isinstance(image, np.ndarray):
            af = AffineTransform(scale=(scale, scale))
            image = warp(image, af.inverse)
            rgb = image[:, :, 0:3]
            depth = image[:, :, 3:4] / scale
            return np.concatenate([rgb, depth], axis=2)
        else:
            raise Exception('unsupported type')

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images):
        #print("images[1].shape: ",images[1].shape)
        in_h, in_w, _ = images[1].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]
        images[1]
        return cropped_images

class Resize(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    'nearest' or 'bilinear'
    """
    def __init__(self, interpolation='bilinear'):
        self.interpolation = interpolation
    def __call__(self, img,size, img_type = 'rgb'):
        assert isinstance(size, int) or isinstance(size, float) or \
               (isinstance(size, collections.Iterable) and len(size) == 2)
        if img_type == 'rgb':
            return scipy.misc.imresize(img, size, self.interpolation)
        elif img_type == 'depth':
            if img.ndim == 2:
                img = scipy.misc.imresize(img, size, self.interpolation, 'F')
            elif img.ndim == 3:
                img = scipy.misc.imresize(img[:,:,0], size, self.interpolation, 'F')
            img_tmp = np.zeros((img.shape[0], img.shape[1],1),dtype=np.float32)
            img_tmp[:,:,0] = img[:,:]
            img = img_tmp
            return img
        else:
            RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

class CenterCrop(object):
    """Crops the given ``numpy.ndarray`` at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for center crop.
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (H x W x C)): Image to be cropped.
        Returns:
            img (numpy.ndarray (H x W x C)): Cropped image.
        """
        i, j, h, w = self.get_params(img[0], self.size)

        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not(_is_numpy_image(img[0])):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img[1].ndim == 3:
            return [im[i:i+h, j:j+w, :] for im in img]
        elif img[1].ndim == 2:
            return [im[i:i+h, j:j+w] for im in img]
        else:
            raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

class RandomRotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) randomly
    """

    def __init__(self, angle_range=(0.0, 360.0), axes=(0, 1), mode='reflect', random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode

    def __call__(self, image):
        angle = self.random_state.uniform(
            self.angle_range[0], self.angle_range[1])
        if isinstance(image, np.ndarray):
            mi, ma = image.min(), image.max()
            image = scipy.ndimage.interpolation.rotate(
                image, angle, reshape=False, axes=self.axes, mode=self.mode)
            return np.clip(image, mi, ma)
        elif isinstance(image, Image.Image):
            return image.rotate(angle)
        else:
            raise Exception('unsupported type')

class Split(object):
    """Split images into individual arraies
    """

    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            if isinstance(s, collections.Sequence):
                slices_.append(slice(*s))
            else:
                slices_.append(s)
        assert all([isinstance(s, slice) for s in slices_]
                   ), 'slices must be consist of slice instances'
        self.slices = slices_
        self.axis = kwargs.get('axis', -1)

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                sl = [slice(None)] * image.ndim
                sl[self.axis] = s
                ret.append(image[tuple(sl)])
            return ret
        else:
            raise Exception("obj is not an numpy array")

class PointOffsetAug(object):

    def __init__(self, config):
        #偏移增强的圆半径范围
        self.radius=config.offset_radius
        self.config=config
        self.aug_prob=self.config.aug_prob #增强概率
        self.loss_for_proper=config.loss_for_proper
        self.loss_for_oneTooth=config.loss_for_oneTooth
        self.loss_for_existed=config.loss_for_existed
        self.loss_for_archLength=config.loss_for_archLength
        self.loss_for_crowd=config.loss_for_crowd
        self.img_w=config.img_size[1]
        self.img_h=config.img_size[0]
    def __call__(self, data_array):
        #以一定概率进行偏移增强
        if random.random() > self.aug_prob:
            return data_array

        #一般point_loca_array为(B,N,2)
        #一般augmessage_array(B,N)

        #根据是否开启各个训练来判断是否要将分配到一张图区
        proper_point_loca_match=data_array[1]
        existed_point_loca_match=data_array[-1]
        #现将4维点对进行按需合并
        proper_point_match_num=proper_point_loca_match.shape[0]
        existed_point_match_num=existed_point_loca_match.shape[0]
        total_point_loca_match=None
        if self.loss_for_proper or self.loss_for_oneTooth :
            total_point_loca_match=proper_point_loca_match.copy()
        if self.loss_for_archLength:
            if total_point_loca_match is None:
                total_point_loca_match=existed_point_loca_match[0:1]
            else:
                total_point_loca_match = np.concatenate((total_point_loca_match,existed_point_loca_match[0:1].copy()),axis=0)
        if self.loss_for_existed :
            if total_point_loca_match is None:
                total_point_loca_match = existed_point_loca_match
            else:
                total_point_loca_match = np.concatenate((total_point_loca_match,existed_point_loca_match[1:].copy()),axis=0)
        if self.loss_for_crowd and self.config.crowd_loss_type=="regression":
            if not self.loss_for_existed:
                total_point_loca_match = np.concatenate((total_point_loca_match,existed_point_loca_match[1:].copy(),),axis=0)
            if not self.loss_for_oneTooth and not self.loss_for_proper:
                total_point_loca_match = np.concatenate((proper_point_loca_match.copy(),total_point_loca_match),axis=0)

        #筛选match点对，看有没有无效的
        if not self.loss_for_oneTooth and (not  self.loss_for_proper) and (not self.loss_for_archLength ) and not (self.loss_for_crowd):
            return data_array
        '''
                        在这个过程中，我们
                        1、先剔除那些无效点对形成util_point_local_match
                        2、将util_point_local_match展开成point_p1
                        3、计算util_point_local_p1的相应重叠点，对于有重叠的计算重叠范围，没有重叠的设置重叠范围角度为0,0，这对后续计算很重要
                        4、对util_point_local_p1计算偏移，生成偏移后结果
                        5、将偏移后结果先形成点对格式如x1,y1,x2,y2
                        6、将最终的结果归还给one_total_point_loca_match
        '''

        # 先把一张图像中的无效点对全部剔除了
        mask = ~(total_point_loca_match == 0).all(axis=1)
        indices = np.where(mask)[0]
        util_point_local_match = total_point_loca_match[indices]
        # 将点对打散变成点,util_point_local_p1代表所有部位0
        util_point_local_p1 = np.concatenate((util_point_local_match[:, :2], util_point_local_match[:, 2:]), axis=0)
        nearest_point_index = self.get_nearest_neighbors(util_point_local_p1, self.radius)
        overlap_mask = nearest_point_index != -1
        #overlap_index_mask = nearest_point_index[nearest_point_index != -1]

        # 记录所有的重叠范围，尺度维N,2, 对于不重叠的就是0,0
        all_theta_range = np.zeros(util_point_local_p1.shape)
        overlap_theta_range = self.compute_overlap_theta_batch(util_point_local_p1[overlap_mask],
                                                               util_point_local_p1[nearest_point_index[overlap_mask]],
                                                               self.radius)

        # 只有重叠的才有角度范围
        all_theta_range[overlap_mask] = overlap_theta_range

        # 将nearest_point_index中的-1的值全部置为自己本身，方便索引
        nearest_point_index[nearest_point_index == -1] = np.where(nearest_point_index == -1)[0]
        util_point_local_p2 = util_point_local_p1[nearest_point_index]

        offset_operation = OffsetOperation(self.radius, util_point_local_p1, util_point_local_p2, all_theta_range,
                                           self.img_w, self.img_h)
        util_point_local_p1_auged = offset_operation.get_offset_random_aug()
        # 将增强后的点先重整成match
        util_point_local_match_auged = np.concatenate((util_point_local_p1_auged[:(util_point_local_p1_auged.shape[0] // 2)],
                                                       util_point_local_p1_auged[
                                                           (util_point_local_p1_auged.shape[0] // 2):]), axis=1)

        total_point_loca_match[indices] = util_point_local_match_auged
        #total_point_loca_match[b] = total_point_loca_match

        # 对整个total_point_loca_match进行增强后，要进行拆分
        if self.loss_for_crowd and self.config.crowd_loss_type == "regression":
            if not self.loss_for_existed:
                existed_point_loca_match_auged = total_point_loca_match[-1 * (existed_point_match_num - 1):]
                data_array[-1][1:] = existed_point_loca_match_auged
                total_point_loca_match = total_point_loca_match[:-1 * (existed_point_match_num - 1)]
            if not self.loss_for_oneTooth and not self.loss_for_proper:
                proper_point_loca_match_auged = total_point_loca_match[:proper_point_match_num]
                data_array[1] = proper_point_loca_match_auged
                total_point_loca_match=total_point_loca_match[proper_point_match_num:]
        if self.loss_for_existed:
            existed_point_loca_match_auged = total_point_loca_match[-1 * (existed_point_match_num-1):]
            data_array[-1][1:] = existed_point_loca_match_auged
            total_point_loca_match=total_point_loca_match[:-1 * (existed_point_match_num-1)]
        if self.loss_for_archLength:
            arch_point_loca_match_auged = total_point_loca_match[-1]
            data_array[-1][0] = arch_point_loca_match_auged
            total_point_loca_match=total_point_loca_match[:-1]
        if self.loss_for_oneTooth or self.loss_for_proper:
            proper_point_loca_match_auged = total_point_loca_match[:proper_point_match_num]
            data_array[1] = proper_point_loca_match_auged



        return data_array


    def get_nearest_neighbors(self,points, radius):
        """
        获取距离对应点最近的点的索引
        给每个点分配一个最近邻（排除自己）且距离 < 2 * radius
        points: numpy array of shape [N, 2]
        radius: float
        returns: list[int], 若无可选邻居则为 -1
        """
        dist = self.pairwise_distance(points)
        np.fill_diagonal(dist, np.inf)  # 排除自己
        dist[(dist >= 2 * radius) | (dist == 0)] = np.inf  # 超过范围的不考虑

        nearest = np.argmin(dist, axis=1)  # 每行中最小的索引
        nearest_val = np.min(dist, axis=1)

        # 如果没有任何可选的邻居，则设置为 -1
        nearest[nearest_val == np.inf] = -1

        #nearest[nearest==-1]=np.where(nearest==-1)[0]

        return nearest

    def pairwise_distance(self,points):
        diff = points[:, None, :] - points[None, :, :]  # [N, N, 2]
        dist = np.linalg.norm(diff, axis=-1)  # [N, N]
        return dist



    def compute_overlap_theta_batch(self,pi, pj, r):
        """
        给定 pi, pj: [N, 2] 的圆心数组，返回以 pi 为极坐标中心的两个交点角度 [theta1, theta2]。
        返回角度满足：theta2 - theta1 ∈ (0, π]，且 theta1 < theta2（可以为负数）。
        """
        vec = pj - pi  # [N, 2]
        d = np.linalg.norm(vec, axis=1)  # [N]
        v = vec / d[:, None]  # 单位方向向量 [N, 2]
        #v是两点连线的单位向量，d是两点连线的长度
        h = np.sqrt(r ** 2 - (d / 2) ** 2)  # [N]
        midpoint = (pi + pj) / 2
        #n是垂直平分线向量
        n = np.stack([-v[:, 1], v[:, 0]], axis=1)  # 垂直向量 [N, 2]

        inter1 = midpoint + n * h[:, None]
        inter2 = midpoint - n * h[:, None]

        def angle(center, pt):
            vec = pt - center
            return np.arctan2(vec[:, 1], vec[:, 0])  # 可为负值，范围 [-π, π]

        theta1 = angle(pi, inter1)
        theta2 = angle(pi, inter2)

        # 将两者调整为 theta1 < theta2
        mask = theta1 > theta2
        theta1[mask], theta2[mask] = theta2[mask], theta1[mask]

        # 判断差值是否大于 π，若是则将结果平移到另一侧
        delta = theta2 - theta1  # 差值在 [0, 2π]
        over_pi = delta > np.pi
        theta1[over_pi] += 2 * np.pi  # 平移到另一侧
        theta1[over_pi], theta2[over_pi] = theta2[over_pi], theta1[over_pi]

        theta_range = np.stack([theta1, theta2], axis=1)  # [N, 2]
        return theta_range


class OffsetOperation:
    #这个类主要用来保存每个点对应的 极坐标分段函数的角度范围，
    # 同时保存p1和p2用来计算极长
    def __init__(self,r,p1,p2,theta_range,img_w,img_h):
        self.img_h = img_h
        self.img_w = img_w
        self.r=r
        self.p1=p1
        self.p2=p2
        self.theta_range=theta_range
        self.theta_probability =self.get_probability_by_theta_range()

    def get_probability_by_theta_range(self):
        theta_reduce=np.abs(self.theta_range[:,1]-self.theta_range[:,0])
        triangle_h=np.cos(theta_reduce/2)*self.r
        triangle_w=np.sin(theta_reduce/2)*self.r*2
        triangle_area=triangle_w*triangle_h/2
        return triangle_area/(np.pi* self.r**2)
    def get_offset_random_aug(self):
        #先计算该点点对存在有意义的
        p1=self.p1.copy()
        p2=self.p2.copy()
        # used_mask=~(p1==0).all(axis=1)
        # used_index=np.where(used_mask)[0]
        # p1_used=p1[used_index]
        # p2_used=p2[used_index]
        theta_range=self.theta_range.copy()


        mask=~(theta_range==0).all(axis=1)
        index1=np.where(mask)[0] # 寻找有圆区域重叠的索引
        index2=np.where(~mask)[0] #寻找没有圆区域重叠的索引
        p1[index1]=p1[index1]+self.get_offset_in_line(p1[index1],p2[index1],theta_range[index1],self.theta_probability.copy()[index1])
        p1[index2]=p1[index2]+self.get_offset_in_ring(p1[index2])

        #超出边界范围的进行截断
        p1[index1][:, 0] = np.clip(p1[index1][:, 0], 0, self.img_w - 1).astype(np.int32)  # 限制 x 坐标
        p1[index1][:, 1] = np.clip(p1[index1][:, 1], 0, self.img_h - 1).astype(np.int32)  # 限制 y 坐标

        p1[index2][:, 0] = np.clip(p1[index2][:, 0], 0, self.img_w - 1).astype(np.int32)  # 限制 x 坐标
        p1[index2][:, 1] = np.clip(p1[index2][:, 1], 0, self.img_w - 1).astype(np.int32)  # 限制 x 坐标
        return p1

    def get_offset_in_line(self,p1,p2,theta_range,theta_range_probability):
        '''
        该方法用来随机圆区域重叠情况下的偏移
        Args:
            p1: 原始点
            p2: 对应原始点的重叠点
            theta_range: p1重叠线区域的角度范围
            theta_range_probability: 重叠区域面积占总面积的比重，用来衡量偏移点落在

        Returns:

        '''


        #先根据概率进行残缺区域的概率取值
        #当这个角度是在分割线范围内的，进行偏移

        sample_probability=np.random.rand(*theta_range_probability.shape)
        mask=sample_probability<=theta_range_probability
        line_range_index=np.where(mask)[0]
        ring_range_index=np.where(~mask)[0]

        line_random_theta=np.random.uniform(theta_range[line_range_index][:,0],theta_range[line_range_index][:,1])
        line_max_r_array = OffsetOperation.compute_r_theta_batch(p1[line_range_index], p2[line_range_index], line_random_theta)
        line_r_array_probability=line_max_r_array*np.random.rand(*line_max_r_array.shape)
        line_offset=np.stack((np.cos(line_random_theta)*line_r_array_probability,np.sin(line_random_theta)*line_r_array_probability),axis=1)


        abs_line_theta_range=np.abs(theta_range[ring_range_index][:,1]-theta_range[ring_range_index][:,0])
        ring_theta_range_end=theta_range[ring_range_index][:,1]+(np.pi*2-abs_line_theta_range)
        ring_theta_range=np.stack((theta_range[ring_range_index][:,1],ring_theta_range_end),axis=1)
        ring_random_theta=np.random.uniform(ring_theta_range[:,0],ring_theta_range[:,1])
        ring_r_array_probability=np.random.rand(*ring_random_theta.shape)
        ring_r_random_array=ring_r_array_probability*self.r
        ring_offset=np.stack((np.cos(ring_random_theta)*ring_r_random_array,np.sin(ring_random_theta)*ring_r_random_array),axis=1)

        all_offset=np.zeros(p1.shape)
        all_offset[line_range_index]=all_offset[line_range_index]+line_offset
        all_offset[ring_range_index]=all_offset[ring_range_index]+ring_offset

        return all_offset
    def get_offset_in_ring(self,p1):
        '''
        该方法用来计算随机圆没有重叠的情况
        Args:
            p1:原始点 (N,2)

        Returns:
            offset: Nx2 numpy array
        '''

        r_array=self.r*np.random.rand(p1.shape[0])
        theta_array= np.random.uniform(0, 2 * np.pi, size=r_array.shape)
        cos_vals = np.cos(theta_array)
        sin_vals = np.sin(theta_array)

        # 堆叠为 (N, 2) 形式：[cos, sin]
        cos_sin_scale = np.stack([cos_vals, sin_vals], axis=1)
        offset = r_array[:, np.newaxis] * cos_sin_scale
        return offset
    @staticmethod
    def compute_r_theta_batch(p1_batch, p2_batch, theta_array):
        v = p2_batch - p1_batch                      # 向量 v_i = p2_i - p1_i
        v_norm = np.linalg.norm(v, axis=1)  # [N]

        d = v_norm / 2                   # 中点距离原点 p1 的距离

        # 法向量方向 φ₀
        phi0 = np.arctan2(v[:, 1], v[:, 0])  # [N]

        r = d / (np.cos(theta_array - (phi0+2*np.pi)%(2*np.pi)) + 1e-8)  # 极坐标极长公式
        return np.abs(r)

    @staticmethod
    def compute_overlap_theta_batch(pi, pj, r):
        """
        给定 pi, pj: [N, 2] 的圆心数组，返回以 pi 为极坐标中心的两个交点角度 [theta1, theta2]。
        返回角度满足：theta2 - theta1 ∈ (0, π]，且 theta1 < theta2（可以为负数）。
        """
        vec = pj - pi  # [N, 2]
        d = np.linalg.norm(vec, axis=1)  # [N]
        v = vec / d[:, None]  # 单位方向向量 [N, 2]
        h = np.sqrt(r ** 2 - (d / 2) ** 2)  # [N]
        midpoint = (pi + pj) / 2
        n = np.stack([-v[:, 1], v[:, 0]], axis=1)  # 垂直向量 [N, 2]

        inter1 = midpoint + n * h[:, None]
        inter2 = midpoint - n * h[:, None]

        def angle(center, pt):
            vec = pt - center
            return np.arctan2(vec[:, 1], vec[:, 0])  # 可为负值，范围 [-π, π]

        theta1 = angle(pi, inter1)
        theta2 = angle(pi, inter2)

        # 将两者调整为 theta1 < theta2
        mask = theta1 > theta2
        theta1[mask], theta2[mask] = theta2[mask], theta1[mask]

        # 判断差值是否大于 π，若是则将结果平移到另一侧
        delta = theta2 - theta1  # 差值在 [0, 2π]
        over_pi = delta > np.pi
        theta1[over_pi] += 2 * np.pi  # 平移到另一侧
        theta1[over_pi], theta2[over_pi] = theta2[over_pi], theta1[over_pi]

        theta_range = np.stack([theta1, theta2], axis=1)  # [N, 2]
        return theta_range


