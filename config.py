from easydict import EasyDict
config = EasyDict()
from run import dim
'''3Sources'''
config.input_features1 =3560
config.input_features2 =3631
config.enhidden_features = [2000, 320, 50,6]
config.dehidden_features1 = [50, 320, 2000,3560]
config.dehidden_features2 = [50, 320, 2000,3631]
config.classes = 6
'''BBCsports'''
# config.input_features1 =2582
# config.input_features2 =2544
# config.enhidden_features = [1500, 200, 50,5]
# config.dehidden_features1 = [50, 200, 1500,2582]
# config.dehidden_features2 = [50, 200, 1500,2544]
# config.classes = 5
'''Caltech101'''
# config.input_features1 =1984
# config.input_features2 =512
# config.enhidden_features = [500, 320, 50,10]
# config.dehidden_features1 = [50, 320, 500,1984]
# config.dehidden_features2 = [50, 320, 500,512]
# config.classes = 20
'''ORL_mtv'''
# config.input_features1 =400
# config.input_features2 =400
# config.enhidden_features = [300, 150, 50,10]
# config.dehidden_features1 = [50, 150, 300,400]
# config.dehidden_features2 = [50, 150, 300,400]
# config.classes = 40
'''Caltech101_7'''
# config.input_features1 =1984
# config.input_features2 =512
# config.enhidden_features = [500, 320, 50,5]
# config.dehidden_features1 = [50, 320, 500,1984]
# config.dehidden_features2 = [50, 320, 500,512]
# config.classes = 7
'''scene15'''
# config.input_features1 =20
# config.input_features2 =59
# config.enhidden_features = [20, 15, 15,10]
# config.dehidden_features1 = [15, 15, 20,20]
# config.dehidden_features2 = [15, 15, 20,59]
# config.classes = 10
'''Prokaryotic'''
# config.input_features1 =393
# config.input_features2 =438
# config.enhidden_features = [300, 150, 50,10]
# config.dehidden_features1 = [50, 150, 300,393]
# config.dehidden_features2 = [50, 150, 300,438]
# config.classes = 4
'''yale_mtv'''
# config.input_features1 =4096
# config.input_features2 =3304
# config.enhidden_features = [1500, 200, 50,5]
# config.dehidden_features1 = [50, 200, 1500,4096]
# config.dehidden_features2 = [50, 200, 1500,3304]
# config.classes = 15
'''flower17'''
# config.input_features1 =1360
# config.input_features2 =1360
# config.enhidden_features = [1000, 200, 50,5]
# config.dehidden_features1 = [50, 200, 1000,1360]
# config.dehidden_features2 = [50, 200, 1000,1360]
# config.classes = 17
'''100leaves'''
# config.input_features1 =64
# config.input_features2 =64
# config.enhidden_features = [200, 200, 50,10]
# config.dehidden_features1 = [50, 200, 200,64]
# config.dehidden_features2 = [50, 200, 200,64]
# config.classes = 100

config.lr = 1e-3
config.momentum = 0.9#SGD才有的参数，动量通过利用过去梯度的加权平均值来调整当前梯度的方向，避免震荡
config.weight_decay = 0
config.w_v = 0

config.print_step = 10
config.tensorboard_step = 100
config.load_iter = 0
config.train_iters = 5000
config.is_train = True
config.use_cuda = True