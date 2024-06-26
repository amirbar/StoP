import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='input_ckp_path')
args = parser.parse_args()

input_ckp = args.path
output_ckp = input_ckp[:-6]

ckp2 = torch.load(input_ckp, map_location='cpu')
new_ckp = {'classy_state_dict':
               {'base_model':
                    {'model':
                         {'trunk': {k.replace('module.', ''): v for k, v in ckp2['target_encoder'].items()},
                          'heads': {}}}}
           }
new_ckp = {
     'target_encoder': ckp2['classy_state_dict']['base_model']['model']['trunk']
}
torch.save(new_ckp, output_ckp)