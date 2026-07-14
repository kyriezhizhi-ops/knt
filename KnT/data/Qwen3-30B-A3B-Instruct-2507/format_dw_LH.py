import os
from langcodes import Language
import argparse

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', "--workspace", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), help="Workspace dir")
    parser.add_argument('-tn', "--test-name", type=str, required=True, help="wmt22/wmt21/...")
    parser.add_argument('-m', "--model-name", type=str, required=True, help="model name")
    parser.add_argument('-s', "--src", type=str, required=True, help='source lang')
    parser.add_argument('-t', "--tgt", type=str, required=True, help='target lang')
    return parser.parse_args()

def main(args):
    if args.src.lower() in ['zh', 'en']:
        raise ValueError(f"错误：源语言 (src) 不应为 '{args.src}'。。")
    workspace = args.workspace
    data_dir=os.path.join(workspace, "data")
    raw_dir=os.path.join(data_dir, "raw")
    format_dir=os.path.join(data_dir, "format")
    test_name = args.test_name
    model_name = args.model_name
    src = args.src
    tgt = args.tgt
    src_full = Language.make(language=src).display_name()
    tgt_full = Language.make(language=tgt).display_name()
    lang_pair_folder = f"{test_name}.{src}-{tgt}"
    output_dir = os.path.join(format_dir,"with-knowledge",model_name,lang_pair_folder)
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(os.path.join(raw_dir, f"{test_name}.{src}-{tgt}.{src}")) as test_src_f:
            test_src_lines = [l.strip() for l in test_src_f.readlines()]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"未找到文件，请检查路径是否正确：\n{e.filename}")
    out_file_path = os.path.join(output_dir,f"{test_name}.{src}-{tgt}.{src}.dw")
    dws = []
    with open(out_file_path, 'w') as out_f:
        for id, src_line in enumerate(test_src_lines):
            all_items = dws + [(src_line, None, None)]
            prompt_lst = []
            for it in all_items:
                it_src, it_tgt, it_dw = it
                s = f"Instruction: Translate the following {src_full} text into {tgt_full}.\n" + \
                f"Preserve the original meaning and facts; avoid paraphrase or invention, especially for names and fixed expressions.Use standard, correct English spelling and conventional forms for names and abbreviations.\n" + \
                f"{src_full}: {it_src}\n" + \
                (f"{tgt_full}: {it_tgt}" if it_tgt else f"{tgt_full}:")
                prompt_lst.append(s)
            prompt = "\n\n".join(prompt_lst)
            out_f.write(
                f"{id:04}\n"
                f"{prompt}\n\n\n"
            )

if __name__ == "__main__":
    args = parse_args()
    main(args)