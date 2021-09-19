from logging import error
import argparse
import os, sys
from forte.data.data_utils import maybe_download

a_model_url_dict = {'cnndm': 'https://drive.google.com/file/d/1XEJpvsZUdEqrcFxmVeEaK8-dKtXZx3gM/view?usp=sharing',
                    'cnndm_ref': 'https://drive.google.com/file/d/1Jd2axVY2lmkG0NDsj1pJ6ZhbVPoPNxad/view?usp=sharing',
                    'xsum': 'https://drive.google.com/file/d/1T0uhyhjYiCnWKlbPzc2QPHCkiolNVjYJ/view?usp=sharing',
                    'yelp': 'https://drive.google.com/file/d/1-5WufCq8faSlXcjkykkTR6UwVe-etFVA/view?usp=sharing',
                    'newsroom': 'https://drive.google.com/file/d/1PDShXZpn_1q6WIKCU3Vr1MNw15OWKMDh/view?usp=sharing'
                    }
persona_chat_model_url_dict = {
    'fact': 'https://drive.google.com/file/d/1-KoWzZwV2_8Abqo_d118cZL2MEUNlwmQ/view?usp=sharing',
    'history': 'https://drive.google.com/file/d/1-QQSfObtLHQdTxOYQvlMbSbhiYBx-YiM/view?usp=sharing',
    'fact_history': 'https://drive.google.com/file/d/1-V951a2uww08AZRbahnlaGpF2iNjyPUC/view?usp=sharing',
    'history_fact': 'https://drive.google.com/file/d/1-W3qE8q3TWOFsxkFQIiBq_AE6jpE76w-/view?usp=sharing'
}
topical_chat_model_url_dict = {
    'fact': 'https://drive.google.com/file/d/1-bAKZWgGiCVR6bF_oCeg5gOmif3qxLj4/view?usp=sharing',
    'history': 'https://drive.google.com/file/d/15_uOgZ_J6F9n0A6Iuc9OKhA2iv9nToGS/view?usp=sharing',
    'fact_history': 'https://drive.google.com/file/d/1-3j958vUIglSgUJEyS7E7sHh6AHPHxdd/view?usp=sharing',
    'history_fact': 'https://drive.google.com/file/d/1-8Z-2MGLsWCDVJ5eAwZ-7nCBI0-tRMEH/view?usp=sharing'
}
cons_data_url_dict = {'cnndm': 'https://drive.google.com/file/d/1fIDqCHt1D9Txt-5QbCKjg-tkpy6pb4Et/view?usp=sharing',
                      'xsum': 'https://drive.google.com/file/d/10-6nRvf0aC9havRfDGyMvnw5cNRDNggD/view?usp=sharing',
                      'yelp': 'https://drive.google.com/file/d/1q0rb3ClOY8KI8V5jsrG8IImhfl8QCUip/view?usp=sharing',
                      'persona_chat': 'https://drive.google.com/file/d/1nyH6wD7rIJVvP1OyuAlCBpO_rgmUJFue/view?usp=sharing',
                      'topical_chat': 'https://drive.google.com/file/d/1ZM9M81l8rB_DdAU_K17UoOwP7lfoObao/view?usp=sharing',
                      'newsroom': 'https://drive.google.com/file/d/1iRyWtPfF9-PJO0eap6hSkQk8uhV5m_I0/view?usp=sharing'
                      }


def download_prop_parser():
    parser = argparse.ArgumentParser('downloader')
    parser.add_argument(
        "--download_type", default="all", help="Download 'model', 'data' or 'all'"
    )
    parser.add_argument(
        "--model_name", default="all", help="model name"
    )
    parser.add_argument(
        "--model_path", default="ckpts/", help="Save path to checkpoints"
    )
    parser.add_argument(
        "--context", default="fact_history", help="Context of dialog datasets"
    )
    parser.add_argument(
        "--cons_data", default="all", help="Constructed data name"
    )
    parser.add_argument(
        "--data_path", default="constructed_data/", help="Save path to constructed data"
    )
    args = parser.parse_args()

    return args


def download_model_wcontext(model_name, save_pth, context):
    context_dict = persona_chat_model_url_dict if model_name == 'persona_chat' else topical_chat_model_url_dict

    maybe_download(
        urls=[context_dict[context]],
        path=os.path.join(save_pth, model_name),
        filenames=['disc_' + context + '.ckpt']
    )


def download_model(model_name, save_pth, context):
    if model_name == 'all':
        for md_name in a_model_url_dict.keys():
            maybe_download(
                urls=[a_model_url_dict[md_name]],
                path=os.path.join(save_pth, md_name),
                filenames=['disc.ckpt']
            )
        for md_name in ['persona_chat', 'topical_chat']:
            for ctx in ['fact', 'history', 'fact_history', 'history_fact']:
                download_model_wcontext(md_name, save_pth, ctx)
    elif model_name in a_model_url_dict.keys():
        maybe_download(
            urls=[a_model_url_dict[model_name]],
            path=os.path.join(save_pth, model_name),
            filenames=['disc.ckpt']
        )
    elif model_name in ['persona_chat', 'topical_chat']:
        download_model_wcontext(model_name, save_pth, context)
    else:
        error('Unrecognized model name: {}.'.format(model_name))


def download_data(data_name, save_pth):
    if data_name == 'all':
        for dt_name in cons_data_url_dict.keys():
            maybe_download(
                urls=[cons_data_url_dict[dt_name]],
                path=os.path.join(save_pth, dt_name),
                filenames=['example.json']
            )
    elif data_name in cons_data_url_dict.keys():
        maybe_download(
            urls=[cons_data_url_dict[data_name]],
            path=os.path.join(save_pth, data_name),
            filenames=['example.json']
        )
    else:
        error('Unrecognized dataset name: {}.'.format(data_name))


if __name__ == '__main__':
    args = download_prop_parser()
    if args.download_type == 'data':
        download_data(args.cons_data, args.data_path)
    elif args.download_type == 'model':
        download_model(args.model_name, args.model_path, args.context)
    elif args.download_type == 'all':
        download_data('all', args.data_path)
        download_model('all', args.model_path, args.context)
    else:
        error('Please input the correct download type')
