import os 
import argparse
import json

from douzero.evaluation.simulation_npc import record, evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Data Record')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_WP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/douzero_WP/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/douzero_WP/landlord_down.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='record_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    record_list = record(args.landlord,
                        args.landlord_up,
                        args.landlord_down,
                        args.eval_data,
                        args.num_workers)

    landlordtype = args.landlord.split('/')[1]
    farmertype = args.landlord_up.split('/')[1]
    # landlordtype_farmertype.json
    path = os.path.join("data_adviser", landlordtype+"_"+farmertype+".json")
    data = json.dumps(record_list, indent=1)
    with open(path, 'w', newline='\n') as f:
        f.write(data)