from modelscope.hub.snapshot_download import snapshot_download
import argparse


def download_modelscope(
        model_id='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
        cache_dir='./',
        revision=None
):
    snapshot_download(
        model_id,
        cache_dir=cache_dir,
        revision=revision
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_id', type=str, required=True, help='模型名称,如：qwen/Qwen-7B-Chat')
    parser.add_argument('-d', '--cache_dir', default='./', type=str, required=False, help='模型下载路径')
    parser.add_argument('-r', '--revision', default='./', type=str, required=False, help='revision 版本')
    args = parser.parse_args()

    download_modelscope(args.model_id, args.cache_dir)
