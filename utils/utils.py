import os

def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print ('use gpu:',gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()