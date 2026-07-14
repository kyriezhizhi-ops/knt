import os
from langcodes import Language
import argparse
from data_format import TRIGGER_SENTS,DEMO_SENTS

SUPPORT_LANGS = TRIGGER_SENTS.keys()

demo_dict = {}
for src_lng in SUPPORT_LANGS:
    for tgt_lng in SUPPORT_LANGS:
        if src_lng == tgt_lng:
            continue
        else:
            demo_dict[(src_lng, tgt_lng)] = [
                (tri_sent, src_demo, tgt_demo)
                    for tri_sent, src_demo, tgt_demo in zip(TRIGGER_SENTS[src_lng], DEMO_SENTS[src_lng], DEMO_SENTS[tgt_lng])
            ]

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', "--workspace", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), help="Workspace dir")
    parser.add_argument('-tn', "--test-name", type=str, required=True, help="wmt22/wmt21/...")
    parser.add_argument('-s', "--src", type=str, required=True, help='source lang')
    parser.add_argument('-t', "--tgt", type=str, required=True, help='target lang')
    parser.add_argument('-m', "--modelname", type=str, required=True, help='model-name')
    return parser.parse_args()

def main(args):
    workspace = args.workspace
    data_dir=os.path.join(workspace, "data")
    raw_dir=os.path.join(data_dir, "raw")
    format_dir=os.path.join(data_dir, "format")
    test_name = args.test_name
    model_name = args.modelname
    src = args.src
    tgt = args.tgt
    src_full = Language.make(language=src).display_name()
    tgt_full = Language.make(language=tgt).display_name()
    with open(os.path.join(raw_dir, f"{test_name}.{src}-{tgt}.{src}")) as test_src_f:
        test_src_lines = [l.strip() for l in test_src_f.readlines()]
        folder_name = f"{test_name}.{src}-{tgt}"
        folder_path = os.path.join(format_dir, model_name, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        out_file_path = os.path.join(folder_path, f"{test_name}.{src}-{tgt}.{src}.demo")
        demos = demo_dict[(src, tgt)]
        with open(out_file_path, 'w') as out_f:
            for id, src_line in enumerate(test_src_lines):
                selected_demos = demos
                all_items = selected_demos + [(src_line, None, None)]
                prompt_lst = []
                for it in all_items:
                    it_src, it_demo_src, it_demo_tgt = it
                    s = f"Let's write {'an' if src_full == 'English' else 'a'} {src_full} sentence related to but different from the input {src_full} sentence and translate it into {tgt_full}.\n" + \
                    f"Input {src_full} sentence: {it_src}\n" + \
                    (f"Output {src_full}-{tgt_full} sentence pair: {it_demo_src}\t{it_demo_tgt}" if (it_demo_src and it_demo_tgt) else f"Output {src_full}-{tgt_full} sentence pair: ")
                    prompt_lst.append(s)

                prompt = "\n\n".join(prompt_lst)
                out_f.write(
                    f"{id:04}\n"
                    f"{prompt}\n\n\n"
                )
if __name__ == "__main__":
    args = parse_args()
    main(args)