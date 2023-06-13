import os, torch

from model.model import RELF_model

def load_model(args):
    # model = RotationEquivariantLocalDescriptor(args)
    model = RELF_model(args)

    print("The order of group: {} with fixparams: {} ResNet-{}".format( 
                    model.get_in_type(), 'fixparams' in os.environ, 18))

    if args.multi_gpu != '-1':
        devices = [int(i) for i in args.multi_gpu.split(',')]
        model = torch.nn.DataParallel(model, device_ids=devices)
        print("Multi-gpu training : {}".format(devices))

    if args.load_dir != '':
        model.train()
        checkpoint = torch.load(args.load_dir)
        model.load_state_dict(checkpoint)
        print("Successfully loaded: {}".format(args.load_dir))
        model.eval()

    model.cuda()

    print("Model loading is done. \n")

    return model
