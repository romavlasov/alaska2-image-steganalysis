import copy
import numpy as np
import jpegio as jio


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetaData(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.state_dict = None
        self.loss = float("inf")
        self.accuracy = float("-inf")
        self.epoch = 0

    def update(self, state_dict, loss, accuracy, epoch):
        self.state_dict = copy.deepcopy(state_dict)
        self.loss = loss
        self.accuracy = accuracy
        self.epoch = epoch

    def __str__(self):
        return "epoch_{}_loss_{:.4f}_accuracy_{:.4f}".format(
            self.epoch, self.loss, self.accuracy
        )


def JPEGdecompressYCbCr(path):
    jpegStruct = jio.read(str(path))

    [col, row] = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    img_dims = np.array(jpegStruct.coef_arrays[0].shape)
    n_blocks = img_dims // 8
    broadcast_dims = (n_blocks[0], 8, n_blocks[1], 8)
    
    YCbCr = []
    for i, dct_coeffs, in enumerate(jpegStruct.coef_arrays):

        if i == 0:
            QM = jpegStruct.quant_tables[i]
        else:
            QM = jpegStruct.quant_tables[1]
        
        t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)
        qm = np.broadcast_to(QM.reshape(1, 8, 1, 8), broadcast_dims)
        dct_coeffs = dct_coeffs.reshape(broadcast_dims)
        
        a = np.transpose(t, axes=(0, 2, 3, 1))
        b = (qm * dct_coeffs).transpose(0,2,1,3)
        c = t.transpose(0,2,1,3)
                
        z = a @ b @ c
        z = z.transpose(0,2,1,3)
        YCbCr.append(z.reshape(img_dims))
                    
    return np.stack(YCbCr, -1).astype(np.float32)