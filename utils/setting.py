import yaml
import argparse


def obj_dic(d):
    obj = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(obj, i, obj_dic(j))
        elif isinstance(j, seqs):
            setattr(obj, i,
                    type(j)(obj_dic(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(obj, i, j)
    return obj


def get_setting(args):
    m_s = {
        'rescan': './settings/rescan.yaml'
    }

    if args.model not in m_s:
        print("The model is not exist!!!")
        return

    with open(m_s[args.model], 'r', encoding='utf-8') as f:
        yam = yaml.load(f.read())
        obj = obj_dic(yam)
        return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, choices=['rescan'])
    args = parser.parse_args()
    obj = get_setting(args)
    print(obj.__dict__)
