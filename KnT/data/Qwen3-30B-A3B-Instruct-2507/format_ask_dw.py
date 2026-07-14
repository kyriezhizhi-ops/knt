import os
from langcodes import Language
import argparse
from data_format import DIFWORDS, DW_SENTS
SUPPORT_LANGS = DW_SENTS.keys()

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', "--workspace", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
                        help="Workspace dir")
    parser.add_argument('-tn', "--test-name", type=str, required=True, help="wmt22/wmt21/...")
    parser.add_argument('-s', "--src", type=str, required=True, help='source lang')  # 仅读取此处传入的源语言
    parser.add_argument('-t', "--tgt", type=str, required=True, help='target lang')
    parser.add_argument('-m',"--modelname", type=str, required=True, help='model-name')
    return parser.parse_args()

def main(args):
    workspace = args.workspace
    data_dir = os.path.join(workspace, "data")
    raw_dir = os.path.join(data_dir, "raw")
    format_dir = os.path.join(data_dir, "format")
    os.makedirs(format_dir, exist_ok=True) 
    model_name = args.modelname
    test_name = args.test_name
    src = args.src  
    tgt = args.tgt
    if src not in SUPPORT_LANGS or tgt not in SUPPORT_LANGS:
        raise ValueError(f"不支持的语言，支持列表：{SUPPORT_LANGS}")
    if src == tgt:
        raise ValueError("源语言和目标语言不能相同")
    demo_dict = {}
    src_lng = src
    for tgt_lng in SUPPORT_LANGS:
        if src_lng == tgt_lng:
            continue
        demo_dict[(src_lng, tgt_lng)] = []
        for sent_idx in range(len(DW_SENTS[src_lng])):
            src_dw_lst = DIFWORDS[src_lng][sent_idx] 
            src_words = [dw["word"] for dw in src_dw_lst]
            tgt_explanations = [dw.get(tgt_lng, f"Explanation for {tgt_lng} not found") for dw in src_dw_lst]  
            demo_output_parts = [
                f'"{src_word}" explanation:{tgt_exp}'
                for src_word, tgt_exp in zip(src_words, tgt_explanations)
            ]
            demo_output = f"{', '.join(demo_output_parts)}"
            demo_dict[(src_lng, tgt_lng)].append(
                (DW_SENTS[src_lng][sent_idx], demo_output)
            )
    src_full = Language.make(language=src).display_name()
    tgt_full = Language.make(language=tgt).display_name()
    test_src_path = os.path.join(raw_dir, f"{test_name}.{src}-{tgt}.{src}")
    try:
        with open(test_src_path, 'r', encoding='utf-8') as f:
            test_src_lines = [l.strip() for l in f if l.strip()]  # 过滤空行
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到测试文件：{test_src_path}")
    try:
        demos = demo_dict[(src, tgt)]
    except KeyError:
        raise KeyError(f"无 {src}->{tgt} 的演示案例")
    if not demos:
        raise ValueError(f"{src}->{tgt} 无可用演示案例")
    folder_name = f"{test_name}.{src}-{tgt}"
    folder_path = os.path.join(format_dir, model_name, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    out_file_path = os.path.join(folder_path, f"{test_name}.{src}-{tgt}.{src}.dw")
    with open(out_file_path, 'w', encoding='utf-8') as out_f:
        for idx, src_line in enumerate(test_src_lines):
            prompt_parts = []
            num_demos = min(4, len(demos))
            selected_demos = demos[:num_demos]
            for demo_sent, demo_output in selected_demos:
                case = (
                    f"Extract difficult word or phrase from the {src_full} sentence and provide its interpretation in {tgt_full}.\n"
                    f"{src_full}: {demo_sent}\n"
                    f"Output:{demo_output}\n"
                )
                prompt_parts.append(case)
            task = (
                f"Extract difficult word or phrase from the {src_full} sentence and provide its interpretation in {tgt_full}.\n"
                f"{src_full}: {src_line}\n"
                f'Output:'
            )
            prompt_parts.append(task)
            prompt = "\n".join(prompt_parts)
            out_f.write(f"{idx:04}\n")
            out_f.write(f"{prompt}\n\n\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)